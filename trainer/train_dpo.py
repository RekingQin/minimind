"""
MiniMind DPO (Direct Preference Optimization) 训练脚本

DPO 是一种无需训练奖励模型(Reward Model)的RLHF替代方案。
核心思想：直接利用人类偏好数据（chosen/rejected对）优化策略模型，
通过最大化策略模型对"优选回答"与"劣选回答"之间的对数概率差值来实现对齐。

DPO 的损失函数:
    L_DPO = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

其中:
    - π_θ: 策略模型（正在训练的模型）
    - π_ref: 参考模型（冻结的基础模型，作为正则化锚点防止模型偏离太远）
    - y_w: chosen（优选回答）
    - y_l: rejected（劣选回答）
    - β: 温度参数，控制偏离参考模型的程度（β越大，惩罚偏离越强）
    - σ: sigmoid 函数

相比传统RLHF流程（训练RM -> PPO优化），DPO更加简洁稳定，只需一个阶段即可完成对齐。
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
from dataset.lm_dataset import DPODataset
from trainer.trainer_utils import get_lr, Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, init_model, SkipBatchSampler

warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    将模型输出的 logits 转换为每个位置上“标签 token”的对数概率。
    
    这是 DPO 计算中的关键步骤：我们需要知道模型在每个时间步对目标 token
    （也就是 labels 中给出的下一个 token）赋予了多大的概率。
    
    Args:
        logits: 模型输出的原始分数，shape: (batch_size, seq_len, vocab_size)
        labels: 目标 token id，shape: (batch_size, seq_len)
    
    Returns:
        log_probs_per_token: 每个位置上目标 token 的对数概率，shape: (batch_size, seq_len)
    
    计算过程:
        1. 对 vocab 维度做 log_softmax，将 logits 转为对数概率分布
        2. 用 gather 按 labels 索引，取出每个位置上目标 token 的对数概率
    
    示例（假设 batch_size=1, seq_len=3, vocab_size=5）:
        logits = [[[2.0, 1.0, 0.5, 0.1, 0.3],    # 位置0: 模型对5个token的打分
                   [0.1, 3.0, 0.2, 0.5, 0.1],    # 位置1: token_id=1 得分最高
                   [0.3, 0.2, 0.1, 4.0, 0.5]]]   # 位置2: token_id=3 得分最高
        labels = [[0, 1, 3]]                       # 目标token分别是 token_0、token_1、token_3
        
        Step 1 - 对最后一维做 log_softmax 后，得到每个位置上的词表对数概率分布
        （注意：softmax 后的概率和为1；log_softmax 后的“对数概率”本身不要求和为0）:
        log_probs ≈ [[[-0.65, -1.65, -2.15, -2.55, -2.35],   # 位置0
                      [-3.13, -0.23, -3.03, -2.73, -3.13],   # 位置1
                      [-3.79, -3.89, -3.99, -0.09, -3.59]]]  # 位置2
        
        Step 2 - 按 labels=[0, 1, 3] 从每个位置的词表分布中取值:
        log_probs_per_token ≈ [[-0.65, -0.23, -0.09]]
        # 含义：
        #   位置0取 token_0 的 log_prob = -0.65
        #   位置1取 token_1 的 log_prob = -0.23
        #   位置2取 token_3 的 log_prob = -0.09
        
        对数概率越接近 0，说明模型对该 token 越确信；越负说明越不确信。
        在 DPO 中，会先对这些逐 token 对数概率按 mask 求和，得到整条 chosen/rejected
        回复的序列级对数概率，再比较策略模型与参考模型的偏好差异。
    """
    # Step 1: 在词表维度(dim=2)上计算 log_softmax，得到每个 token 的对数概率分布
    log_probs = F.log_softmax(logits, dim=2)
    # Step 2: 根据 labels 索引，取出每个位置上目标 token 对应的 log_prob
    # labels.unsqueeze(2) 将 shape 从 (B, S) -> (B, S, 1)，用于 gather 索引
    # gather 后 shape 为 (B, S, 1)，squeeze(-1) 去掉最后一维变为 (B, S)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算 DPO (Direct Preference Optimization) 损失函数。
    
    DPO 的核心公式:
        L = -log σ(β * ((log π_θ(y_w|x) - log π_θ(y_l|x)) - (log π_ref(y_w|x) - log π_ref(y_l|x))))
    
    等价形式:
        L = -log σ(β * (pi_logratios - ref_logratios))
    
    其中:
        pi_logratios  = log π_θ(y_w|x) - log π_θ(y_l|x)   (策略模型对chosen和rejected的对数概率之差)
        ref_logratios = log π_ref(y_w|x) - log π_ref(y_l|x) (参考模型对chosen和rejected的对数概率之差)
    
    直觉理解:
        - 当策略模型相对于参考模型更偏好 chosen 而非 rejected 时，损失减小
        - β 控制正则化强度：β 越大，模型越不愿意偏离参考模型的行为
        - 参考模型提供了一个"基线"，避免策略模型过度优化偏好数据
    
    Args:
        ref_log_probs: 参考模型的逐 token 对数概率，shape: (batch_size, seq_len)
                       其中 batch_size = 2 * actual_batch（前半是 chosen，后半是 rejected）
        policy_log_probs: 策略模型的逐 token 对数概率，shape 同上
        mask: 有效 token 的 mask（padding 位置为 0），shape 同上
        beta: DPO 温度参数，控制偏离参考模型的惩罚力度
    
    Returns:
        loss: 标量，DPO 损失值
    
    数据组织方式:
        输入的 batch 是 chosen 和 rejected 拼接在一起的:
        [chosen_1, chosen_2, ..., chosen_n, rejected_1, rejected_2, ..., rejected_n]
        前半部分是 chosen（优选），后半部分是 rejected（劣选）
    """
    # Step 1: 对每个序列，将逐 token 的对数概率求和，得到整条序列的对数概率
    # mask 确保只对有效 token（非 padding）求和
    # 求和后 shape: (batch_size,)，即每条序列一个标量值
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)

    # Step 2: 将 chosen 和 rejected 数据分开
    # 输入数据的前半部分是 chosen，后半部分是 rejected
    batch_size = ref_log_probs.shape[0]
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]       # 参考模型对 chosen 的对数概率
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]       # 参考模型对 rejected 的对数概率
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2] # 策略模型对 chosen 的对数概率
    reject_policy_log_probs = policy_log_probs[batch_size // 2:] # 策略模型对 rejected 的对数概率

    # Step 3: 计算策略模型的对数比率 (policy log-ratio)
    # pi_logratios = log π_θ(y_chosen) - log π_θ(y_rejected)
    # 表示策略模型对 chosen 相对于 rejected 的偏好程度
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    
    # Step 4: 计算参考模型的对数比率 (reference log-ratio)
    # ref_logratios = log π_ref(y_chosen) - log π_ref(y_rejected)
    # 表示参考模型对 chosen 相对于 rejected 的偏好程度（作为基线）
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    
    # Step 5: 计算 DPO 的核心 logits
    # logits = pi_logratios - ref_logratios
    # 表示策略模型相对于参考模型，额外增加了多少对 chosen 的偏好
    # 如果 logits > 0，说明策略模型比参考模型更偏好 chosen，这是我们期望的方向
    logits = pi_logratios - ref_logratios
    
    # Step 6: 计算最终的 DPO 损失
    # loss = -log σ(β * logits)
    # 当 logits 越大（策略模型越偏好 chosen），σ(β*logits) 越接近 1，loss 越小
    # β 放大了 logits 的影响：β 越大，对偏离参考模型的惩罚越重
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """
    DPO 训练的单个 epoch 逻辑。
    
    训练流程:
        1. 从 DataLoader 获取 chosen 和 rejected 的输入/标签/mask
        2. 将 chosen 和 rejected 拼接成一个大 batch，同时送入模型（效率更高）
        3. 用冻结的参考模型(ref_model)计算参考对数概率（不计算梯度）
        4. 用策略模型(model)计算策略对数概率（计算梯度）
        5. 计算 DPO 损失并反向传播
        6. 梯度累积 + 梯度裁剪 + 优化器更新
    
    Args:
        epoch: 当前 epoch 编号
        loader: DataLoader，提供 DPO 训练数据
        iters: 总迭代步数
        ref_model: 冻结的参考模型
        lm_config: 模型配置
        start_step: 起始步数（用于断点续训）
        wandb: wandb/swanlab 日志记录器
        beta: DPO 温度超参数
    """
    start_time = time.time()
    last_step = start_step

    for step, batch in enumerate(loader, start=start_step + 1):
        last_step = step
        # ---- 数据准备：获取 chosen 和 rejected 的输入、标签、mask ----
        # 这些 key 来自 DPODataset.__getitem__()，原始数据格式为:
        #   {"chosen": [{"role":"user","content":"..."},{"role":"assistant","content":"好回答"}],
        #    "rejected": [{"role":"user","content":"..."},{"role":"assistant","content":"差回答"}]}
        #
        # DPODataset 的处理逻辑：
        #   1. 将 chosen/rejected 对话用 chat_template 转为完整文本
        #   2. tokenize 后生成 input_ids
        #   3. x = input_ids[:-1]（输入），y = input_ids[1:]（标签，即下一个 token 预测目标）
        #   4. mask = generate_loss_mask()[1:]，只对 assistant 回复区间标记为 1
        #      （即只计算模型"生成"部分的概率，忽略 prompt/system 部分）
        #
        # mask 的定义与作用（关键）：
        #   generate_loss_mask() 扫描 input_ids，找到 <bos>assistant\n 到 <eos>\n 之间的区间标记为1，
        #   其余（user prompt、system、padding）标记为 0。示例：
        #
        #   input_ids = [<bos>user\n, 你, 好, <eos>\n, <bos>assistant\n, 你好, ！, 帮你, <eos>\n, <pad>]
        #   loss_mask = [0,          0,  0,  0,       0,                1,    1,   1,    1,       0    ]
        #   拆分后：x = input_ids[:-1], y = input_ids[1:], mask = loss_mask[1:]
        #
        #   mask 在 dpo_loss() 中的用途：
        #     ref_log_probs = (ref_log_probs * mask).sum(dim=1)
        #     policy_log_probs = (policy_log_probs * mask).sum(dim=1)
        #   即：只对 assistant 回复部分的 token 累加对数概率，忽略 prompt 和 padding。
        #   原因：
        #     - prompt 部分对 chosen/rejected 完全相同（同一个用户提问），计算概率无区分意义
        #     - padding 不是真实内容，不应参与概率计算
        #     - 如果不用 mask，prompt 的高概率会淹没回复差异，导致 DPO 损失信号极弱
        x_chosen = batch['x_chosen'].to(args.device)       # chosen 序列的输入 token ids, shape: (B, seq_len-1)
        x_rejected = batch['x_rejected'].to(args.device)   # rejected 序列的输入 token ids
        y_chosen = batch['y_chosen'].to(args.device)       # chosen 的目标标签（input_ids 右移一位）
        y_rejected = batch['y_rejected'].to(args.device)   # rejected 的目标标签
        mask_chosen = batch['mask_chosen'].to(args.device)  # chosen 的 loss mask（仅 assistant 回复为1）
        mask_rejected = batch['mask_rejected'].to(args.device)  # rejected 的 loss mask
        
        # 将 chosen 和 rejected 拼接成一个大 batch
        # 这样只需要一次前向传播，效率更高
        # 拼接后: [chosen_samples, rejected_samples]
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # ---- 学习率调度：使用余弦退火策略动态调整学习率 ----
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with autocast_ctx:
            # ---- 参考模型前向传播（不计算梯度，节省显存） ----
            # 参考模型是冻结的，提供"基线"对数概率，防止策略模型偏离太远
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            # 计算参考模型对每个 token 的对数概率
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            
            # ---- 策略模型前向传播（计算梯度） ----
            outputs = model(x)
            logits = outputs.logits
            # 计算策略模型对每个 token 的对数概率
            policy_log_probs = logits_to_log_probs(logits, y)
            
            # ---- 计算 DPO 损失 ----
            # dpo_loss_val 是核心对齐损失
            # outputs.aux_loss 是 MoE 架构的辅助负载均衡损失（非 MoE 时为 0）
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            loss = dpo_loss_val + outputs.aux_loss
            # 梯度累积：将损失除以累积步数，等效于增大 batch size
            loss = loss / args.accumulation_steps

        # 混合精度反向传播
        scaler.scale(loss).backward()

        # ---- 梯度累积达标后，执行优化器步骤 ----
        if step % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)  # 反缩放梯度（用于梯度裁剪前）
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)  # 梯度裁剪，防止梯度爆炸
            scaler.step(optimizer)      # 执行参数更新
            scaler.update()             # 更新 scaler 的缩放因子
            optimizer.zero_grad(set_to_none=True)  # 清零梯度（set_to_none 比 zero_grad 更省显存）

        # ---- 日志输出 ----
        if step % args.log_interval == 0 or step == iters:
            spend_time = time.time() - start_time
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            eta_min = spend_time / max(step - start_step, 1) * (iters - step) // 60
            
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            if wandb: wandb.log({"loss": current_loss, "dpo_loss": current_dpo_loss, "aux_loss": current_aux_loss, "learning_rate": current_lr, "epoch_time": eta_min})

        # ---- 定期保存模型检查点 ----
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 获取原始模型（去除 DDP/compile 包装）
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存为半精度以减小文件体积
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, scaler=scaler, epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints')
            model.train()
            del state_dict

        # ---- 手动释放显存，避免 OOM ----
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss

    # ---- 处理最后未凑齐累积步数的残留梯度 ----
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    # ==================== 命令行参数定义 ====================
    # DPO 训练的关键超参数说明：
    # - learning_rate: 建议极低(<=5e-8)，因为 DPO 是在 SFT 基础上微调，学习率过大会导致灾难性遗忘
    # - beta: DPO 温度参数，控制模型偏离参考模型的程度。典型值 0.1~0.5
    #         beta 越大 -> 模型越保守，越不敢偏离参考模型
    #         beta 越小 -> 模型越激进，更容易过拟合偏好数据
    # - from_weight: 通常基于 SFT 后的权重进行 DPO 训练
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.15, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境（如果启用了多GPU）
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子确保可复现性，不同 rank 使用不同种子保证数据多样性
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果开启断点续训，尝试加载之前保存的检查点
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # bfloat16 数值范围更大更稳定（推荐），float16 在某些旧显卡上兼容性更好
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义策略模型和参考模型 ==========
    # 注意：model 和 ref_model 初始化时加载的是同一份权重（from_weight，默认为 'full_sft'），
    # 但它们在训练中的角色完全不同：
    #   - model（策略模型）：参数会随训练不断更新，学习人类偏好
    #   - ref_model（参考模型）：参数永远冻结在初始状态，作为正则化锚点
    # 随着训练进行，model 和 ref_model 的参数差异越来越大，
    # DPO 损失函数通过 ref_logratios 项隐式约束这种偏离（等价于 KL 散度惩罚），
    # β 参数控制惩罚力度。如果没有 ref_model 的约束，model 可能过拟合偏好数据导致生成退化。
    
    # 策略模型(policy model): 正在训练的模型，参数会被梯度更新
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    # 参考模型(reference model): 加载相同的 from_weight 权重，但设为 eval 模式且冻结所有参数
    # 它在整个训练过程中保持不变，提供"原始 SFT 模型会怎么做"的基线概率
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model.eval()
    ref_model.requires_grad_(False)
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    # 加载 DPO 偏好数据集（包含 chosen/rejected 配对）
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # GradScaler 仅在 float16 时启用（bfloat16 不需要动态缩放）
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 6. 从ckp恢复状态（断点续训） ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scaler.load_state_dict(ckp_data['scaler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    # torch.compile 通过 JIT 编译加速模型推理和训练
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    # 分布式数据并行（DDP）：多GPU训练时同步梯度
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        # 分布式采样器设置 epoch 以确保每个 epoch 数据顺序不同
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 断点续训时跳过已训练的步数
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)
    
    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()