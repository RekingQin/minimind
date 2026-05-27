"""
MiniMind 全参数监督微调（Full SFT, Supervised Fine-Tuning）训练脚本
==================================================================
功能：在已有的预训练（pretrain）权重基础上，使用对话格式数据集对模型进行
全参数微调（区别于 LoRA 等参数高效微调），让模型学会按 chat 模板回答用户问题。

核心特性：
  - 支持单卡 / DDP 多卡分布式训练
  - 支持混合精度训练（bf16 / fp16）
  - 支持梯度累积、梯度裁剪
  - 支持余弦学习率调度
  - 支持断点续训（保存/加载 model + optimizer + scaler 等完整状态）
  - 支持 torch.compile 加速
  - 支持 wandb / swanlab 实验追踪
  - 支持 MoE 架构（Mixture-of-Experts）
"""
import os
import sys

# 把项目根目录加入 sys.path，便于以 `python trainer/train_full_sft.py` 直接运行时
# 仍能 import 到 model、dataset、trainer 等包
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 注意：必须在 torch 之前导入 datasets，规避 Windows 下 pyarrow / torch DLL 冲突（issue #771）
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
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

# 屏蔽训练过程中各种第三方库的 warning，保持日志整洁
warnings.filterwarnings('ignore')


def train_epoch(epoch, loader, iters, start_step=0, wandb=None):
    """
    训练单个 epoch。

    Args:
        epoch:      当前 epoch 索引（从 0 开始）
        loader:     当前 epoch 的 DataLoader（可能已通过 SkipBatchSampler 跳过前若干 step）
        iters:      该 epoch 的总 step 数（用于学习率调度和日志展示）
        start_step: 续训时的起始 step（从 0 开始训练时为 0）
        wandb:      wandb / swanlab 句柄，主进程才会传入；非主进程或未启用则为 None
    """
    start_time = time.time()
    last_step = start_step  # 记录最后一次执行的 step，用于 epoch 结束时处理"剩余未对齐的累积梯度"

    # enumerate 的 start 参数让 step 从 start_step+1 开始计数，
    # 这样无论是否续训，step 始终对应"全局 epoch 内的真实步数"
    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        # 数据搬到 GPU；input_ids 是 token 序列，labels 已在 SFTDataset 里把
        # 非 assistant 回复部分置为 -100（忽略 loss）
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        last_step = step

        # ===== 1. 余弦学习率调度 =====
        # 计算全局 step（跨 epoch），并按总训练步数计算余弦衰减后的 lr
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ===== 2. 前向传播（混合精度） =====
        # autocast_ctx：GPU 下为 bf16/fp16 autocast 上下文，CPU 下为 nullcontext
        with autocast_ctx:
            res = model(input_ids, labels=labels)
            # 总 loss = 主 logits CE loss + MoE 辅助负载均衡 loss（非 MoE 时 aux_loss=0）
            loss = res.loss + res.aux_loss
            # 梯度累积：将 loss 除以"本累积轮的实际步数"，使多次 backward 累加后等效于大 batch 的平均梯度
            # 正常累积轮：actual_accum == accumulation_steps
            # epoch 末尾不足一轮：actual_accum = 剩余真实步数，避免梯度被错误地缩小
            # 例：accumulation_steps=8 但 epoch 末尾只剩 3 步时，若仍除以 8，
            #     这一次"残余更新"的梯度量级会被错误地缩成 3/8，等效学习率被打折。
            remaining = iters - step + 1                              # 当前 step（含）到 epoch 结束还剩多少步
            steps_into_accum = (step - 1) % args.accumulation_steps   # 当前累积轮已走了几步（不含本步）
            actual_accum = min(args.accumulation_steps, steps_into_accum + remaining)
            loss = loss / actual_accum

        # ===== 3. 反向传播 =====
        # scaler 仅在 fp16 下生效；bf16 / cpu 下 scaler 是 no-op，等价于普通 backward
        scaler.scale(loss).backward()

        # ===== 4. 梯度累积：到达累积步数才真正更新参数 =====
        if step % args.accumulation_steps == 0:
            # 先 unscale 把梯度还原到正常量级，才能正确做梯度裁剪
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # 优化器更新参数
            scaler.step(optimizer)
            # 动态调整 GradScaler 的缩放系数（fp16 下根据是否 inf/nan 进行调整）
            scaler.update()

            # 清零梯度；set_to_none=True 比置零张量更省显存且略快
            optimizer.zero_grad(set_to_none=True)

        # ===== 5. 日志打印（间隔 log_interval 步或 epoch 末尾） =====
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            # 还原真实 loss（前面除过 actual_accum，这里乘回去打印未缩放的真实 loss）
            current_loss = loss.item() * actual_accum
            current_aux_loss = res.aux_loss.item() if res.aux_loss is not None else 0.0
            current_logits_loss = current_loss - current_aux_loss
            current_lr = optimizer.param_groups[-1]['lr']
            # 估算本 epoch 剩余时间（分钟）：平均每 step 耗时 × 剩余 step 数
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, logits_loss: {current_logits_loss:.4f}, aux_loss: {current_aux_loss:.4f}, lr: {current_lr:.8f}, epoch_time: {eta_min:.1f}min')
            if wandb: wandb.log({"loss": current_loss, "logits_loss": current_logits_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ===== 6. 保存 checkpoint（仅主进程，避免 DDP 多卡重复写文件） =====
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()  # 切到 eval 模式，关闭 dropout 等
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 单独保存一份纯权重（FP16，CPU），便于推理部署
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 解包 DDP 包装，获取真正的模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            # 解包 torch.compile 包装（compile 后会有 _orig_mod 指向原模型）
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存为半精度 + CPU，减少磁盘占用
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 同时保存"完整训练状态"（含 optimizer、epoch、step、scaler 等），用于断点续训
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scaler=scaler)
            model.train()  # 切回 train 模式继续训练
            del state_dict

        # 显式释放本 step 的中间变量，缓解长时间训练的显存增长
        del input_ids, labels, res, loss

    # ===== 7. epoch 末尾：处理"未对齐 accumulation_steps 的剩余梯度" =====
    # 若总 step 数不能被 accumulation_steps 整除，最后那段累积的梯度还没被 step 出去，
    # 这里补一次更新，避免最后几个 batch 的训练信号被丢弃。
    # 注：上面前向 loss 已经按 actual_accum（不足一轮时取剩余步数）做归一化，
    #     所以这里累积的梯度量级是正确的"该轮均值梯度"，可以直接 step。
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ====================== 命令行参数解析 ======================
    parser = argparse.ArgumentParser(description="MiniMind Full SFT")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_sft', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=768, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    parser.add_argument('--from_weight', default='pretrain', type=str, help="基于哪个权重训练，为none则不基于任何权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Full-SFT", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化分布式环境和随机种子 ==========
    # init_distributed_mode 内部根据环境变量 RANK 判断是否启用 DDP；
    # 单卡模式直接返回 0，DDP 模式返回当前进程的 local_rank
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"  # DDP 下每张卡绑定自己的设备
    # 不同 rank 用不同种子，避免数据增强等环节多卡完全同步而失去多样性
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型超参数、检测续训 checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 构建模型配置（从命令行注入），决定模型容量与是否启用 MoE
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 仅当用户指定续训时才尝试加载 *_resume.pth；否则 ckp_data 为 None
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度上下文 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 下不启用 autocast；GPU 下使用指定 dtype（bf16 推荐：不需要 GradScaler，稳定性好）
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置 wandb / swanlab 实验追踪（只在主进程） ==========
    wandb = None
    if args.use_wandb and is_main_process():
        # 项目里使用 swanlab 但接口与 wandb 兼容，因此别名为 wandb 直接用
        import swanlab as wandb
        # 续训时复用保存在 checkpoint 中的 run id，让指标曲线连续
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Full-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 构建模型、数据集、采样器、优化器 ==========
    # init_model 内部会从 ../out/{from_weight}_{hidden_size}{_moe}.pth 加载预训练权重
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    # SFT 数据集：负责把多轮对话样本转换成 (input_ids, labels) 张量对。
    # ----------------------------------------------------------------
    # 这是 SFT 与 Pretrain 的关键区别所在：
    #   - Pretrain (PretrainDataset)：labels = input_ids（除 pad 置 -100），全 token 都参与 loss
    #   - SFT      (SFTDataset)    ：labels 默认全为 -100，只把 "<|im_start|>assistant\n ... <|im_end|>\n"
    #                                 区间内的位置还原为 input_ids，其余位置（system / user / 特殊 token）
    #                                 保持 -100
    # 由于 nn.CrossEntropyLoss 默认 ignore_index=-100，label=-100 的位置不会贡献 loss 也不会回传梯度。
    # 因此模型只学 "看到 system+user 上下文之后，assistant 应该怎么回答"，
    # 不会学着去复述用户问题或 system prompt，训练信号更干净。
    #
    # 内部主要工作（见 dataset/lm_dataset.py 的 SFTDataset）：
    #   1) load_dataset 加载 jsonl 中的 conversations 字段
    #   2) pre_processing_chat 概率性补一条 system 消息，提升泛化
    #   3) tokenizer.apply_chat_template 把多轮对话拼成符合 chat 模板的纯文本
    #   4) post_processing_chat 概率性去除空 <think>\n\n</think> 标签
    #   5) tokenize → 截断 / pad 到 max_length
    #   6) generate_labels：扫描 input_ids，定位每段 assistant 回复区间，
    #      把这些位置的 label 设为 input_ids[j]（其它位置保持 -100）
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # DDP 模式用 DistributedSampler 自动按 rank 切分数据；单卡为 None
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 仅 fp16 需要，bf16 / cpu 下 enabled=False（变成 no-op）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从 checkpoint 恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 恢复模型权重（注意：此时 model 还没被 DDP / compile 包装，state_dict key 是干净的）
        model.load_state_dict(ckp_data['model'])
        # 恢复优化器动量等内部状态，保证续训等价于不中断
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 恢复 GradScaler 的缩放系数等
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. torch.compile 加速 + DDP 包装（顺序很重要） ==========
    # 先 compile 再 DDP 是官方推荐顺序：compile 后保留 _orig_mod 引用，方便后续解包
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 主训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # DDP 下必须每个 epoch 调用 set_epoch，保证 shuffle 在各 rank 间一致且每 epoch 不同
        train_sampler and train_sampler.set_epoch(epoch)
        # 单卡模式下手动 shuffle：固定基于 epoch 的种子让所有 rank 生成相同的 indices
        # （这样配合 SkipBatchSampler 才能正确"跳过已训练过的 batch"）
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 仅当处于"续训的那个起始 epoch"时才 skip，后续 epoch 从头训练
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        # SkipBatchSampler：按 batch_size 分组并跳过前 skip 个 batch（断点续训关键）
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            # 续训情景：iters 仍按完整 epoch 长度计算（len(loader)+skip），
            # 这样学习率调度和"剩余时间估算"才与未中断时一致
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, start_step, wandb)
        else:
            train_epoch(epoch, loader, len(loader), 0, wandb)
    
    # ========== 9. 清理分布式进程组，释放 NCCL 资源 ==========
    if dist.is_initialized(): dist.destroy_process_group()
