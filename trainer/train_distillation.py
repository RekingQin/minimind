"""
====================================================================================
MiniMind 知识蒸馏训练脚本 (Knowledge Distillation Training)
====================================================================================

【核心思想】
知识蒸馏(Knowledge Distillation)是一种模型压缩技术，由 Hinton 等人在 2015 年提出。
其核心思想是：用一个大模型（Teacher，教师模型）的"软标签"(soft labels)来指导
一个小模型（Student，学生模型）的训练，使小模型在参数量更少的情况下尽可能接近大模型的性能。

【本脚本的蒸馏策略】
- Teacher 模型：可以是 MoE(混合专家)模型 或 更大hidden_size的dense模型，参数冻结，不参与梯度更新
- Student 模型：较小的dense模型，是实际被训练的目标模型
- 损失函数：总损失 = alpha * CE_loss + (1 - alpha) * KL_distill_loss
  - CE_loss: 学生模型输出 vs 真实标签 (hard label)
  - KL_distill_loss: 学生模型的软概率分布 vs 教师模型的软概率分布 (soft label)

【温度(Temperature)的作用】
- 温度T用于"软化"概率分布：softmax(logits / T)
- T=1 时是标准softmax；T越大，分布越平滑（越"软"），暴露更多token间的相对关系
- 这使学生模型能从教师的"暗知识"(dark knowledge)中学习，而不仅仅是最高概率的那个token

【典型使用场景】
1. MoE教师 → Dense学生：用MoE模型的丰富表达能力指导小模型
2. 大模型教师 → 小模型学生：用大hidden_size模型蒸馏到小hidden_size模型
====================================================================================
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import time
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from model.model_minimind import MiniMindConfig
from dataset.lm_dataset import SFTDataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def distillation_loss(student_logits, teacher_logits, temperature=1.0, reduction='batchmean'):
    """
    计算知识蒸馏的 KL 散度损失。

    【原理】
    KL散度衡量学生模型的概率分布 Q 与教师模型的概率分布 P 之间的差异：
        KL(P || Q) = Σ P(x) * log(P(x) / Q(x))

    通过最小化 KL 散度，让学生模型的输出分布逐渐逼近教师模型的输出分布。

    【温度缩放】
    - 将 logits 除以温度 T 再做 softmax，使概率分布更加平滑
    - T > 1 时，原本概率很小的 token 也会有更大的概率值，暴露出 token 之间的相似性关系
    - 最终 loss 乘以 T^2 进行补偿（因为梯度在 T 缩放后会变小，需要放大回来）

    【参数说明】
    Args:
        student_logits: 学生模型的原始输出 logits, shape: [N, vocab_size]
                        N 是有效 token 的数量（已用 loss_mask 过滤了 padding）
        teacher_logits: 教师模型的原始输出 logits, shape: [N, vocab_size]
                        教师模型是冻结的，只提供"软标签"指导
        temperature:    蒸馏温度 T，T 越大分布越平滑，暴露更多"暗知识"
                        推荐范围 1.0 ~ 2.0
        reduction:      KL 散度的归约方式，'batchmean' 表示对 batch 求平均

    Returns:
        (T^2) * KL_div: 经过温度补偿的蒸馏损失标量

    【为什么要乘 T^2？】
    因为 softmax(z/T) 的梯度相对于 z 会缩小 1/T 倍，
    所以 KL 散度的梯度也会缩小 1/T^2，乘以 T^2 恢复正确的梯度量级。
    """
    # 教师模型的软概率分布（不需要梯度，仅作为"目标分布"）
    with torch.no_grad():
        # teacher_probs: 教师模型经温度软化后的概率分布 P = softmax(teacher_logits / T)
        # 这就是"软标签"，包含了教师模型对各个 token 相对可能性的认知
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1).detach()

    # 学生模型的 log 概率分布（需要梯度，因为要通过反向传播更新学生模型参数）
    # student_log_probs = log(softmax(student_logits / T))
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

    # 计算 KL 散度: KL(teacher_probs || student_probs)
    # 注意: F.kl_div 的输入是 (log_Q, P)，计算的是 KL(P || Q) = Σ P * (log P - log Q)
    kl = F.kl_div(
        student_log_probs,   # 学生的 log 概率 (要被优化的)
        teacher_probs,       # 教师的概率 (目标分布，固定不变)
        reduction=reduction  # 'batchmean': 对所有样本的 KL 散度求平均
    )
    # 乘以 T^2 补偿温度缩放导致的梯度缩小
    return (temperature ** 2) * kl


def train_epoch(epoch, loader, iters, teacher_model, lm_config_student, start_step=0, wandb=None, alpha=0.0, temperature=1.0):
    """
    执行一个 epoch 的知识蒸馏训练。

    【训练流程】
    对于每个 batch:
    1. 学生模型前向传播 → 得到 student_logits
    2. 教师模型前向传播（no_grad）→ 得到 teacher_logits（仅提供软标签，不更新参数）
    3. 计算 CE Loss（学生 vs 真实标签）
    4. 计算 KL Distill Loss（学生分布 vs 教师分布）
    5. 总损失 = alpha * CE + (1-alpha) * KL
    6. 反向传播，仅更新学生模型参数

    Args:
        epoch:            当前 epoch 索引
        loader:           数据加载器
        iters:            本 epoch 的总迭代步数
        teacher_model:    教师模型（参数冻结，eval模式，不参与梯度计算）
        lm_config_student: 学生模型配置
        start_step:       断点续训的起始步数
        wandb:            可视化日志工具
        alpha:            CE 损失的权重，alpha=1 时退化为纯 SFT，alpha=0 时退化为纯蒸馏
        temperature:      蒸馏温度
    """
    start_time = time.time()
    last_step = start_step
    
    # 确保教师模型处于推理模式，且所有参数不计算梯度
    # 教师模型的角色是"知识的来源"，其参数在蒸馏过程中保持不变
    if teacher_model is not None:
        teacher_model.eval()           # 设为评估模式（关闭 dropout 等）
        teacher_model.requires_grad_(False)  # 冻结所有参数，不计算梯度

    for step, (input_ids, labels) in enumerate(loader, start=start_step + 1):
        last_step = step
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        # loss_mask: 标记哪些位置是有效的预测位置（非 padding，非 -100 的位置）
        # labels 中 -100 表示该位置不参与损失计算（通常是 prompt 部分或 padding）
        loss_mask = (labels[..., 1:] != -100).float()
        # 使用余弦退火等策略动态调整学习率
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # ==================== 学生模型前向传播 ====================
        # 学生模型是被训练的目标，其参数会通过反向传播更新
        with autocast_ctx:
            res = model(input_ids)
            # student_logits: 学生模型对每个位置的下一个 token 的预测分数
            # [..， :-1, :] 去掉最后一个位置（因为没有对应的"下一个token"标签）
            # shape: [batch_size, seq_len - 1, vocab_size]
            student_logits = res.logits[..., :-1, :].contiguous()

        # ==================== 教师模型前向传播 ====================
        # 教师模型只做推理，提供"软标签"指导学生模型
        # 不需要梯度（torch.no_grad），不更新参数，只贡献知识
        if teacher_model is not None:
            with torch.no_grad():
                # 教师模型对相同输入产生预测，得到更"丰富"的概率分布
                # 这个分布包含了教师模型对各 token 相似性的理解（暗知识）
                teacher_logits = teacher_model(input_ids).logits[..., :-1, :].contiguous()
                # 如果教师模型和学生模型的词表大小不同（如 MoE 模型可能更大），
                # 截取教师 logits 到学生词表大小，确保维度匹配
                vocab_size_student = student_logits.size(-1)
                teacher_logits = teacher_logits[..., :vocab_size_student]

        # ==================== 计算损失 ====================
        # 蒸馏训练的损失由两部分组成：
        # (1) CE Loss: 学生预测 vs 真实标签（保证学生学会正确答案）
        # (2) KL Loss: 学生分布 vs 教师分布（让学生模仿教师的"思维方式"）

        # ---------- 1) Ground-Truth CE Loss（交叉熵损失）----------
        # 这是标准的语言模型训练损失：预测下一个 token
        shift_labels = labels[..., 1:].contiguous()  # 对齐标签：第 i 个位置的 logit 预测第 i+1 个 token
        loss_mask_flat = loss_mask.view(-1)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),  # [batch*seq_len, vocab_size]
            shift_labels.view(-1),                              # [batch*seq_len]
            ignore_index=-100,    # 忽略标记为 -100 的位置（如 prompt 部分）
            reduction='none'      # 不自动求平均，手动用 loss_mask 加权
        )
        # 只对有效位置（loss_mask==1）的损失求平均
        ce_loss_raw = torch.sum(ce_loss * loss_mask_flat) / (loss_mask_flat.sum() + 1e-8)
        # 如果学生模型使用 MoE，还需要加上辅助负载均衡损失
        if lm_config_student.use_moe: ce_loss = ce_loss_raw + res.aux_loss
        else: ce_loss = ce_loss_raw

        # ---------- 2) Distillation Loss（蒸馏 KL 散度损失）----------
        # 让学生模型的输出概率分布接近教师模型的输出概率分布
        # 只在有效位置（loss_mask==1）上计算蒸馏损失，忽略 padding 和 prompt 位置
        if teacher_model is not None:
            distill_loss = distillation_loss(
                student_logits.view(-1, student_logits.size(-1))[loss_mask_flat == 1],  # 学生的有效位置 logits
                teacher_logits.view(-1, teacher_logits.size(-1))[loss_mask_flat == 1],  # 教师的有效位置 logits
                temperature=temperature  # 蒸馏温度，控制软标签的"柔软程度"
            )
        else:
            distill_loss = torch.tensor(0.0, device=args.device)

        # ---------- 3) 总损失 = alpha * CE + (1-alpha) * Distill ----------
        # alpha 控制两种损失的平衡：
        #   - alpha=1.0: 纯 CE 训练（完全忽略教师），退化为标准 SFT
        #   - alpha=0.0: 纯蒸馏（完全忽略真实标签），只模仿教师
        #   - alpha=0.5: 两者平衡（推荐值，既学正确答案又模仿教师的分布）
        # 除以 accumulation_steps 实现梯度累积（等效于更大的 batch size）
        loss = (alpha * ce_loss + (1 - alpha) * distill_loss) / args.accumulation_steps

        # 反向传播：只有学生模型的参数会被更新（教师模型已冻结）
        scaler.scale(loss).backward()

        # 梯度累积：每累积 accumulation_steps 步才真正更新一次参数
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # 日志打印
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_ce_loss = ce_loss_raw.item()
            current_aux_loss = res.aux_loss.item() if lm_config_student.use_moe else 0.0
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, ce: {current_ce_loss:.4f}, aux_loss: {current_aux_loss:.4f}, distill: {distill_loss.item():.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if wandb:
                wandb.log({
                    "loss": current_loss,
                    "ce_loss": current_ce_loss,
                    "aux_loss": current_aux_loss,
                    "distill_loss": distill_loss.item() if teacher_model is not None else 0.0,
                    "learning_rate": current_lr,
                    "epoch_time": eta_min
                })

        # 定期保存学生模型的 checkpoint
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config_student.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config_student.hidden_size}{moe_suffix}.pth'
            # 处理 DDP 包装和 torch.compile 包装，获取原始模型
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存为 fp16 格式，节省存储空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config_student, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # 及时释放显存，避免 OOM
        del input_ids, labels, loss_mask, res, student_logits, ce_loss, distill_loss, loss

    # 处理最后一批未满 accumulation_steps 的梯度
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ====================================================================================
    # 知识蒸馏主流程
    # 典型场景：用 MoE 教师模型蒸馏 Dense 学生模型，
    #          或用更大 hidden_size 的教师模型蒸馏更小 hidden_size 的学生模型
    # ====================================================================================
    parser = argparse.ArgumentParser(description="MiniMind Knowledge Distillation")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='full_dist', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=6, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument("--max_seq_len", type=int, default=340, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument("--data_path", type=str, default="../dataset/sft_t2t_mini.jsonl", help="训练数据路径")
    # ----- 学生模型配置（被训练的小模型）-----
    parser.add_argument('--student_hidden_size', default=768, type=int, help="学生模型隐藏层维度")
    parser.add_argument('--student_num_layers', default=8, type=int, help="学生模型隐藏层数量")
    # ----- 教师模型配置（提供知识的大模型/MoE模型）-----
    parser.add_argument('--teacher_hidden_size', default=768, type=int, help="教师模型隐藏层维度")
    parser.add_argument('--teacher_num_layers', default=8, type=int, help="教师模型隐藏层数量")
    parser.add_argument('--student_use_moe', default=0, type=int, choices=[0, 1], help="学生模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--teacher_use_moe', default=1, type=int, choices=[0, 1], help="教师模型是否使用MoE（0=否，1=是）")
    parser.add_argument('--from_student_weight', default='full_sft', type=str, help="学生模型基于哪个预训练权重初始化")
    parser.add_argument('--from_teacher_weight', default='full_sft', type=str, help="教师模型加载哪个权重（通常是训练好的大模型）")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    # ----- 蒸馏超参数 -----
    parser.add_argument('--alpha', default=0.5, type=float, help="CE损失权重，总损失=alpha*CE+(1-alpha)*KL")
    parser.add_argument('--temperature', default=1.5, type=float, help="蒸馏温度（推荐范围1.0-2.0）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Distillation", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查 checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 学生模型配置：较小的 dense 模型（被训练的目标）
    lm_config_student = MiniMindConfig(hidden_size=args.student_hidden_size, num_hidden_layers=args.student_num_layers, use_moe=bool(args.student_use_moe))
    # 教师模型配置：较大的模型或 MoE 模型（提供知识，参数不更新）
    lm_config_teacher = MiniMindConfig(hidden_size=args.teacher_hidden_size, num_hidden_layers=args.teacher_num_layers, use_moe=bool(args.teacher_use_moe))
    # 检查是否有可恢复的断点
    ckp_data = lm_checkpoint(lm_config_student, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度训练（AMP）==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置可视化日志（swanlab/wandb）==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-Distill-S{args.student_hidden_size}T{args.teacher_hidden_size}-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义学生模型和教师模型 ==========
    # -------- 学生模型（Student）--------
    # 学生模型是训练目标，参数会被更新
    # 通常是较小的 dense 模型，希望通过蒸馏获得接近教师模型的性能
    model, tokenizer = init_model(lm_config_student, args.from_student_weight, device=args.device)
    Logger(f'学生模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # -------- 教师模型（Teacher）--------
    # 教师模型只做推理，提供"软标签"（softened probability distribution）
    # 其参数完全冻结，不参与梯度计算，不会被更新
    # 教师模型通常更大/更强（如 MoE 模型），已经在相同数据上训练好
    teacher_model, _ = init_model(lm_config_teacher, args.from_teacher_weight, device=args.device)
    teacher_model.eval()              # 永久设为评估模式
    teacher_model.requires_grad_(False)  # 冻结所有参数，节省显存和计算
    Logger(f'教师模型总参数量：{sum(p.numel() for p in teacher_model.parameters()) / 1e6:.3f} M')
    
    # 数据集和训练工具
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 用于 fp16 混合精度训练的梯度缩放（bf16 不需要，故 enabled=False）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 只优化学生模型的参数（教师模型不在优化器中）
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从 checkpoint 恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. torch.compile 加速 & 分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始蒸馏训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 支持断点续训：跳过已训练的 step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, teacher_model, lm_config_student, start_step, wandb, args.alpha, args.temperature)
        else:
            train_epoch(epoch, loader, len(loader), teacher_model, lm_config_student, 0, wandb, args.alpha, args.temperature)
    
    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()