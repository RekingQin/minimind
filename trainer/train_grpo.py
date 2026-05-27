"""
===================================================================================
MiniMind GRPO (Group Relative Policy Optimization) 训练脚本
===================================================================================

GRPO 是一种基于组相对优势的策略优化算法，源自 DeepSeek 的论文。
与传统 PPO 不同，GRPO 不需要 Critic（价值网络），而是通过对同一 prompt 生成
多个候选回复（一组），计算组内相对优势（advantage）来替代价值函数的估计。

核心流程：
1. 对每个 prompt 生成 num_generations 个候选回复
2. 使用 Reward Model 对每个回复打分
3. 在同组（同一 prompt 的多个回复）内，将 reward 标准化为 advantage
4. 使用 PPO-clip 或 CISPO 风格的策略梯度损失来更新策略
5. 加入 KL 散度惩罚，防止策略偏离参考模型太远

支持特性：
- 分布式训练 (DDP)
- 混合精度训练 (bf16/fp16)
- 梯度累积与梯度裁剪
- 两种损失类型：标准 GRPO (PPO-clip) 和 CISPO
- 可插拔的 Rollout 引擎（原生 PyTorch / SGLang 加速推理）
- 断点续训（checkpoint resume）
- torch.compile 加速
- WandB / SwanLab 实验跟踪
===================================================================================
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """
    计算文本的重复惩罚分数。
    
    通过检测文本中 n-gram 的重复程度来衡量生成质量：
    - 将文本分词为 token 列表
    - 统计所有 n-gram 中重复的数量
    - 重复越多，惩罚越大（最大不超过 cap）
    
    Args:
        text: 待检测的文本
        n: n-gram 的长度，默认为 3（trigram）
        cap: 惩罚的上限值，默认为 0.5
    
    Returns:
        float: 重复惩罚值，范围 [0, cap]
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    """
    计算所有回复的综合奖励分数。
    
    奖励由以下几部分组成：
    1. 长度奖励：回复长度在 [20, 800] 范围内得 +0.5，否则 -0.5
    2. 思考格式奖励（如果包含 </think> 标签）：
       - 思考内容长度在 [20, 300] 范围内得 +1.0，否则 -0.5
       - 只有一个 </think> 标签得 +0.25，否则 -0.25
    3. 重复惩罚：基于 n-gram 重复度的扣分
    4. Reward Model 打分：使用外部奖励模型对回复质量进行评估
    
    Args:
        prompts: 原始 prompt 列表，长度为 B
        responses: 所有生成的回复列表，长度为 B * num_generations
        reward_model: 奖励模型实例（如 InternLM2-Reward）
    
    Returns:
        rewards: 形状为 [B * num_generations] 的 tensor，每个回复对应一个分数
    """
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                # 从 prompt 中解析出对话消息列表（ChatML 格式）
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response

                # 规则1: 回复长度奖励 —— 鼓励适中长度的回复
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

                # 规则2: 思考格式奖励 —— 鼓励模型产出结构化的思考过程
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    # 思考内容的合理长度奖励
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    # 格式正确性奖励（只有一个 </think> 标签）
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()

                # 规则3: 重复惩罚 —— 惩罚重复冗余的内容
                rewards[response_idx] -= rep_penalty(answer)

                # 规则4: Reward Model 评分 —— 使用外部模型评估回复质量
                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        # 将 Reward Model 的分数加到最终奖励中
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    """
    执行一个 epoch 的 GRPO 训练。
    
    核心算法流程（每个 step）：
    1. [Rollout] 对 batch 中每个 prompt 生成 num_generations 个候选回复
    2. [Reward] 对所有候选回复计算奖励
    3. [Advantage] 在组内标准化奖励，得到相对优势 advantage
    4. [Policy Loss] 计算 PPO-clip / CISPO 策略损失 + KL 惩罚
    5. [Update] 反向传播并更新模型参数
    
    Args:
        epoch: 当前 epoch 编号
        loader: 数据加载器
        iters: 总迭代步数
        rollout_engine: 推理引擎（用于生成候选回复）
        ref_model: 参考模型（用于计算 KL 散度）
        reward_model: 奖励模型
        start_step: 起始步数（用于断点续训）
        wandb: wandb/swanlab 实例（可选）
        use_sglang: 是否使用 SGLang 引擎
    """
    for step, batch in enumerate(loader, start=start_step + 1):
        # =====================================================================
        # Step 1: 准备 prompt 输入
        # =====================================================================
        prompts = batch['prompt']  # list[str], length B
        # 对 prompt 进行 tokenize，左填充以适应批量生成
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        # 截断过长的 prompt，保留末尾（左截断）
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        # =====================================================================
        # Step 2: Rollout —— 使用当前策略为每个 prompt 生成多个候选回复
        # =====================================================================
        # rollout_engine 会为每个 prompt 生成 num_generations 个回复
        # 并计算 old policy 下每个 token 的 log probability
        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids           # [B*num_gen, P+R] 完整序列（prompt + completion）
        completion_ids = rollout_result.completion_ids # [B*num_gen, R] 仅生成部分的 token ids
        completions = rollout_result.completions       # list[str] 解码后的回复文本
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()  # [B*num_gen, R] 旧策略的 log prob
        prompt_lens = rollout_result.prompt_lens.to(args.device)  # [B*num_gen] 每条 prompt 的实际长度
        # 构建完整序列的 attention mask（非 pad token 位置为 1）
        full_mask = (outputs != tokenizer.pad_token_id).long()
        # 计算 completion 部分在完整序列中对应的 logits 位置索引
        # logp_pos[i, j] = prompt_lens[i] - 1 + j，即第 i 个样本第 j 个生成 token 的位置
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        # =====================================================================
        # Step 3: 计算 Reward
        # =====================================================================
        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        # =====================================================================
        # Step 4: 前向传播 —— 计算当前策略的 log prob
        # =====================================================================
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            # 用当前策略模型对完整序列进行前向推理
            res = model_unwrapped(outputs, attention_mask=full_mask)
            # MoE 辅助损失（如果使用 MoE 架构）
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            # 计算当前策略在 completion token 位置的 log probability
            # log_softmax -> gather(选取实际 token 的 logp) -> 取 completion 对应位置
            per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # =====================================================================
        # Step 5: 计算参考模型的 log prob（用于 KL 惩罚）
        # =====================================================================
        with torch.no_grad():
            # 参考模型不参与梯度计算，仅用于计算 KL 散度
            ref_per_token_logps = F.log_softmax(ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)

        # =====================================================================
        # [可选] Debug 模式：打印采样信息
        # =====================================================================
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        # =====================================================================
        # Step 6: 计算组内相对优势 (Group Relative Advantage)
        # =====================================================================
        # GRPO 的核心思想：不使用 Critic 网络，而是在同一 prompt 的多个回复组内
        # 做标准化，用组内均值和标准差将 reward 转化为 advantage
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        # 每组的均值，重复展开到每个样本
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 每组的标准差（有偏），重复展开
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)  # [B*num_gen]
        # 标准化得到 advantage：(reward - mean) / (std + eps)
        advantages = (rewards - mean_r) / (std_r + 1e-4)  # [B*num_gen]

        # =====================================================================
        # Step 7: 构建 completion mask（截止到 EOS token）
        # =====================================================================
        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        # 找到每个序列中 EOS token 的位置
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # 最终的 completion_mask：从开始到 EOS 位置（含）为 1，之后为 0
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]

        # =====================================================================
        # Step 8: 计算策略损失
        # =====================================================================
        # KL 散度惩罚项：使用 k3 estimator (exp(x) - x - 1)，比朴素 KL 更稳定
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1  # [B*num_gen, R]

        # 重要性采样比率 (importance sampling ratio)
        # ratio = π_θ(a|s) / π_old(a|s) = exp(log π_θ - log π_old)
        ratio = torch.exp(per_token_logps - old_per_token_logps)  # [B*num_gen, R]

        if args.loss_type == "cispo":
            # CISPO (Clipped Importance Sampling Policy Optimization):
            # 只 clamp ratio 的上界，并且 detach ratio（不通过 ratio 传梯度）
            # 梯度仅通过 per_token_logps 传递，更稳定
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # 标准 GRPO (PPO-clip 风格):
            # 对 ratio 做双侧 clip [1-ε, 1+ε]，取 min 作为保守更新
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)

        # 对有效 token 求平均损失，再对 batch 求平均
        policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
        # 加上 MoE 辅助损失，除以梯度累积步数
        loss = (policy_loss + aux_loss) / args.accumulation_steps  # scalar
        loss.backward()

        # =====================================================================
        # Step 9: 参数更新（梯度累积达到步数时执行）
        # =====================================================================
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # =====================================================================
        # Step 10: 日志记录
        # =====================================================================
        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = loss.item() * args.accumulation_steps
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            # KL 散度：衡量当前策略与参考模型的偏离程度
            kl_ref_val = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(completion_mask.sum().item(), 1)
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Reward: {avg_reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss: {policy_loss_val:.4f}, Avg Response Len: {avg_len_val:.2f}, Learning Rate: {current_lr:.8f}')

            if wandb and is_main_process():
                wandb.log({
                    "reward": avg_reward_val,
                    "kl_ref": kl_ref_val,
                    "advantages_std": advantages_std_val,
                    "advantages_mean": advantages_mean_val,
                    "policy_loss": policy_loss_val,
                    "avg_response_len": avg_len_val,
                    "learning_rate": current_lr
                })

        # =====================================================================
        # Step 11: 定期保存模型 checkpoint
        # =====================================================================
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            # 保存为半精度以节省空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # 同步 rollout 引擎中的策略模型权重
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # =====================================================================
        # Step 12: 清理显存
        # =====================================================================
        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos

    # epoch 结束时，如果还有未消耗的累积梯度，执行最后一次参数更新
    if step > start_step and step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    # ========== 命令行参数定义 ==========
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境（如使用 torchrun 启动则自动配置 DDP）
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    # 设置随机种子，不同 rank 使用不同种子以确保数据多样性
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    # 配置模型参数，max_seq_len 设为 prompt + generation 的总长度
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    # 如果开启续训，尝试加载已有 checkpoint
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU 不支持 autocast，使用 nullcontext 作为占位
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验跟踪（WandB/SwanLab） ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight

    # Policy 模型（待训练的策略模型）
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)

    # Reference 模型（冻结参数，用于计算 KL 散度惩罚）
    # 参考模型保持 SFT 阶段的权重不变，防止策略偏离太远导致 reward hacking
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # Reward 模型（外部奖励模型，如 InternLM2-1.8B-Reward）
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)

    # Rollout 引擎（可插拔设计，负责策略推理/生成）
    # - torch: 原生 PyTorch 推理，简单直接
    # - sglang: 使用 SGLang 推理服务器，支持更高效的批量推理
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    # 数据集和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 使用 AdamW 优化器，适合 Transformer 模型
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 先创建一个临时 loader 计算总步数
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    # 使用余弦退火调度器，学习率从 lr 衰减到 lr/10
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从 checkpoint 恢复训练状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译优化和分布式包装 ==========
    if args.use_compile == 1:
        # torch.compile 可以加速模型推理和训练（需要 PyTorch 2.0+）
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        # 用 DDP 包装模型以支持多 GPU 数据并行训练
        model = DistributedDataParallel(model, device_ids=[local_rank])
    # 确保 rollout 引擎持有最新的策略模型引用
    rollout_engine.update_policy(model)
    
    # ========== 8. 开始训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # 分布式训练中每个 epoch 设置不同的 sampler seed，确保数据打乱方式不同
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        # 如果是续训，跳过已经训练过的 step
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
