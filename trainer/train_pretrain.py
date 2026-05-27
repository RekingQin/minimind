"""
MiniMind 预训练脚本
==================
本脚本实现了 MiniMind 语言模型的预训练流程，主要特性包括：
1. 支持单卡/多卡分布式训练（DDP）
2. 支持混合精度训练（FP16/BF16）加速
3. 支持梯度累积（模拟大 batch）
4. 支持断点续训（checkpoint resume）
5. 支持 MoE（Mixture of Experts）架构
6. 支持 torch.compile 编译加速
7. 支持 wandb/swanlab 实验日志记录

训练流程概览：
  数据加载 → 前向传播 → 计算损失 → 反向传播 → 梯度累积 → 梯度裁剪 → 参数更新 → 保存检查点
"""

import os
import sys

# 设置包名，确保相对导入正常工作
__package__ = "trainer"
# 将上级目录加入 sys.path，使得可以导入 model/ 和 dataset/ 下的模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 提前导入 datasets 库，解决 Windows 下 pyarrow/torch DLL 冲突问题（issue #771）
import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

# 导入模型配置类
from model.model_minimind import MiniMindConfig
# 导入预训练数据集类
from dataset.lm_dataset import PretrainDataset
# 导入训练工具函数：
#   get_lr: 余弦退火学习率调度
#   Logger: 仅主进程打印日志
#   is_main_process: 判断是否为主进程
#   lm_checkpoint: 检查点保存/加载
#   init_distributed_mode: 初始化分布式环境
#   setup_seed: 设置随机种子保证可复现
#   init_model: 初始化模型和分词器
#   SkipBatchSampler: 支持跳过前N个batch的采样器（用于断点续训）
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 忽略所有警告信息（如 deprecation warnings 等）
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练一个 epoch 的核心函数。

    Args:
        epoch (int): 当前 epoch 编号（从0开始）
        loader (DataLoader): 训练数据加载器
        iters (int): 当前 epoch 的总迭代步数（包含可能跳过的步数）
        start_step (int): 起始步数，用于断点续训时从中间恢复
        wandb: 实验日志记录工具实例（swanlab/wandb），为 None 则不记录
    """
    start_time = time.time()
    last_step = start_step

    # 遍历数据加载器，step 从 start_step+1 开始编号
    # input_ids: 输入 token 序列, labels: 目标 token 序列（通常是 input_ids 右移一位）
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 将数据移至训练设备（GPU/CPU）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # ===== 学习率调度 =====
        # 使用余弦退火策略动态调整学习率
        # 公式: lr * (0.1 + 0.45 * (1 + cos(π * current_step / total_steps)))
        # 效果: 从初始 lr 平滑衰减到 0.1*lr，中间先快后慢
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ===== 前向传播（混合精度上下文中执行）=====
        # autocast_ctx: 在 GPU 上启用自动混合精度（BF16/FP16），在 CPU 上为空上下文
        with autocast_ctx:
            # 模型前向传播，返回包含 loss 和 aux_loss 的结果
            res = model(input_ids, labels=labels)
            # loss: 语言模型主损失（交叉熵）
            # aux_loss: MoE 辅助损失（负载均衡损失），非 MoE 模型时为 0
            loss = res.loss + res.aux_loss
            # 梯度累积：将 loss 除以当前累积轮的实际步数，等效于增大 batch_size
            # 正常情况下 actual_accum = accumulation_steps
            # epoch 末尾不足一轮时，actual_accum = 剩余步数，保证梯度均值正确
            remaining = iters - step + 1  # 当前 step（含）到 epoch 结束还剩多少步
            steps_into_accum = (step - 1) % args.accumulation_steps  # 当前累积轮已走了几步（不含本步）
            actual_accum = min(args.accumulation_steps, steps_into_accum + remaining)
            loss = loss / actual_accum

        # ===== 反向传播 =====
        # 使用 GradScaler 缩放 loss 后反向传播（防止 FP16 下梯度下溢）
        # 注：BF16 不需要缩放，此时 scaler 的 enabled=False，相当于直接 backward
        scaler.scale(loss).backward()

        # ===== 梯度累积 & 参数更新 =====
        # 每累积 accumulation_steps 步后才真正更新一次参数
        if step % args.accumulation_steps == 0:
            # 将缩放后的梯度还原为真实梯度
            scaler.unscale_(optimizer)
            # 梯度裁剪：防止梯度爆炸，将所有参数梯度的 L2 范数限制在 grad_clip 以内
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 执行优化器更新（scaler 会检查梯度是否有 inf/nan，有则跳过此次更新）
            scaler.step(optimizer)
            # 更新 scaler 的缩放因子
            scaler.update()

            # 清零梯度，set_to_none=True 比 zero_grad() 更节省内存
            optimizer.zero_grad(set_to_none=True)

        # ===== 日志打印 =====
        # 每 log_interval 步或 epoch 最后一步打印训练信息
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 还原真实 loss（乘回当前实际累积步数，与前面的 loss/actual_accum 对应）
            current_loss = loss.item() * actual_accum
            # MoE 辅助损失（用于监控专家负载均衡情况）
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            # 纯语言模型损失 = 总损失 - 辅助损失
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算剩余时间（分钟）
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            # 记录到 wandb/swanlab
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ===== 模型保存 =====
        # 每 save_interval 步或 epoch 最后一步，且仅在主进程中保存
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()  # 切换到评估模式（关闭 dropout 等）
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'

            # 解包 DDP 和 torch.compile 的包装，获取原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()

            # 保存模型权重（转为 FP16 减少磁盘空间，移到 CPU 避免占用 GPU 显存）
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)

            # 保存完整的训练检查点（包含 optimizer、scaler 状态，用于断点续训）
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')

            model.train()  # 恢复训练模式
            del state_dict  # 释放内存

        # 手动删除本步变量，帮助 Python GC 及时回收显存
        del input_ids, labels, res, loss

    # ===== 处理 epoch 末尾未完成的梯度累积 =====
    # 如果最后一步不是 accumulation_steps 的整数倍，需要额外执行一次参数更新
    # 否则这部分累积的梯度会丢失
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # =====================================================================
    # 命令行参数定义
    # =====================================================================
    parser = argparse.ArgumentParser(description="MiniMind Pretraining")

    # --- 保存相关 ---
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='pretrain', type=str, help="保存权重的前缀名")

    # --- 训练超参数 ---
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=8, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")

    # --- 模型结构参数 ---
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=340, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")

    # --- 数据和权重路径 ---
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_t2t_mini.jsonl", help="预训练数据路径")
    parser.add_argument('--from_weight', default='none', type=str, help="基于哪个权重训练，为none则从头开始")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")

    # --- 实验记录 ---
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain", help="wandb项目名")

    # --- 加速选项 ---
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")

    args = parser.parse_args()

    # ========== 1. 初始化分布式环境和随机种子 ==========
    # init_distributed_mode: 检测环境变量 RANK，若存在则初始化 NCCL 后端的分布式通信
    # 返回 local_rank（当前进程在本机上的 GPU 编号）
    local_rank = init_distributed_mode()
    # 如果处于分布式模式，将设备设置为当前进程对应的 GPU
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子：基础种子42 + 进程 rank，确保不同进程数据打散不同但可复现
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 配置目录、模型参数、检查断点 ==========
    # 创建模型输出目录
    os.makedirs(args.save_dir, exist_ok=True)
    # 初始化模型配置（Transformer 架构参数）
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果启用了断点续训，尝试从 checkpoints 目录加载上次训练状态
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None

    # ========== 3. 设置混合精度训练上下文 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # BF16: 动态范围大，不易溢出，适合 A100/H100 等支持 BF16 的 GPU
    # FP16: 精度更高但范围小，需要 GradScaler 防止梯度下溢
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 不支持 autocast，使用空上下文; GPU 使用 torch.cuda.amp.autocast
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 配置实验日志（wandb/swanlab） ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 使用 swanlab 作为 wandb 的替代（接口兼容）
        import swanlab as wandb
        # 如果是断点续训，恢复到之前的 wandb run
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Pretrain-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)

    # ========== 5. 初始化模型、数据集、优化器 ==========
    # init_model: 创建 MiniMindForCausalLM 模型 + 加载分词器
    # 如果 from_weight != 'none'，会从指定路径加载预训练权重
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # PretrainDataset: 加载 JSONL 格式的预训练数据，tokenize 后截断到 max_seq_len
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 分布式采样器：确保多卡训练时每个 GPU 拿到不重叠的数据子集
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler: 仅在 FP16 模式下启用（BF16 不需要梯度缩放）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # AdamW 优化器：Adam + 权重衰减（weight decay），是 Transformer 训练的标准选择
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ========== 6. 从检查点恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重
        model.load_state_dict(ckp_data['model'])
        # 恢复优化器状态（包括动量 momentum 等）
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 恢复梯度缩放器状态
        scaler.load_state_dict(ckp_data['scaler'])
        # 恢复 epoch 和 step 位置
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 7. 模型编译和分布式包装 ==========
    # torch.compile: PyTorch 2.0+ 的图编译优化，可显著加速训练
    # 通过将动态图转为静态图，减少 Python 开销和 kernel launch 次数
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # DDP 包装：实现多卡数据并行，自动同步各 GPU 的梯度
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    # ========== 8. 训练主循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的 epoch（确保每个 epoch 数据 shuffle 不同）
        train_sampler and train_sampler.set_epoch(epoch)

        # 设置随机种子并生成打乱的索引（非分布式模式下使用）
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()

        # 计算需要跳过的 batch 数（仅在续训的第一个 epoch 生效）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0

        # SkipBatchSampler: 自定义采样器，跳过前 skip 个 batch，实现续训
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)

        # 创建数据加载器：pin_memory=True 加速 CPU→GPU 数据传输
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)

        if skip > 0:
            # 续训模式：打印跳过信息，传入总步数=loader长度+已跳过步数
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            # 正常训练
            train_epoch(epoch, loader, len(loader), 0, wandb)

    # ========== 9. 清理分布式进程 ==========
    # 训练结束后销毁进程组，释放 NCCL 通信资源
    if dist.is_initialized(): dist.destroy_process_group()