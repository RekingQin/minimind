"""
===================================================================================
MiniMind PPO (Proximal Policy Optimization) 训练脚本
===================================================================================

PPO 是 OpenAI 提出的经典强化学习策略优化算法，广泛用于 RLHF（基于人类反馈的
强化学习）中对语言模型进行对齐训练。

与 GRPO 的核心区别：
- PPO 使用一个独立的 Critic（价值网络）来估计状态价值 V(s)，
  并通过 GAE（广义优势估计）计算 advantage
- GRPO 则通过组内相对排序来估计 advantage，不需要 Critic

PPO-RLHF 的四个模型：
1. Actor（策略模型）：待训练的生成模型，负责生成回复
2. Critic（价值模型）：估计每个状态的期望回报 V(s)，用于计算 advantage
3. Reference Model（参考模型）：冻结的 SFT 模型，用于 KL 惩罚防止策略漂移
4. Reward Model（奖励模型）：对生成的回复进行质量评分

核心算法流程：
1. [Rollout] 用 Actor 对 prompt 生成回复
2. [Reward] 用 Reward Model 对回复打分
3. [GAE] 用 Critic 估计价值，通过 GAE 计算时序差分优势
4. [PPO Update] 多轮 mini-batch 更新 Actor 和 Critic：
   - Actor: PPO-clip 策略梯度 + KL 惩罚
   - Critic: Value function regression loss（也用 clip）
5. [Early Stop] 当 KL 散度过大时提前停止更新

支持特性：
- Actor/Critic 分离训练（独立优化器和学习率调度）
- GAE (Generalized Advantage Estimation) 时序差分优势估计
- PPO-clip 双侧裁剪（Actor 和 Critic 都有 clip）
- KL early stopping 防止策略更新过猛
- Mini-batch 更新 + 多轮 PPO epoch
- 分布式训练 (DDP)、混合精度、梯度累积
- 可插拔 Rollout 引擎（PyTorch / SGLang）
- 断点续训
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
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    """
    计算文本的 n-gram 重复惩罚。
    
    通过统计 trigram 的重复比例来衡量生成文本的冗余度：
    - 重复 n-gram 越多，惩罚越大
    - 惩罚值上限为 cap
    
    Args:
        text: 待检测文本
        n: n-gram 长度，默认为 3
        cap: 惩罚上限，默认为 0.5
    
    Returns:
        float: 重复惩罚值 [0, cap]
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


class CriticModel(MiniMindForCausalLM):
    """
    Critic 价值模型（Value Network）。
    
    继承自 MiniMindForCausalLM（与 Actor 共享相同的 Transformer 骨干架构），
    但将语言模型头（lm_head）替换为一个输出标量价值的线性层（value_head）。
    
    Critic 的作用：
    - 对每个 token 位置输出一个标量值 V(s_t)，表示从该位置开始的期望累积回报
    - 用于计算 GAE 优势估计：A_t = δ_t + γλδ_{t+1} + ...
      其中 δ_t = r_t + γV(s_{t+1}) - V(s_t)
    
    初始化策略：
    - 使用与 Actor 相同的预训练权重初始化骨干网络（共享语言理解能力）
    - value_head 随机初始化
    """
    
    def __init__(self, params):
        super().__init__(params)
        # 替换语言模型头为价值输出头：hidden_size -> 1
        self.value_head = nn.Linear(params.hidden_size, 1)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        """
        前向推理：输入 token 序列，输出每个位置的价值估计。
        
        Args:
            input_ids: [B, seq_len] 输入 token ids
            attention_mask: [B, seq_len] 注意力掩码
        
        Returns:
            values: [B, seq_len] 每个位置的状态价值 V(s_t)
        """
        # 使用基础 Transformer 模型获取隐藏状态
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        # 对最后一层隐藏状态做 RMSNorm
        hidden_states = self.model.norm(outputs[0])
        # 通过 value_head 映射为标量价值
        values = self.value_head(hidden_states).squeeze(-1)  # [B, seq_len]
        return values


def calculate_rewards(prompts, responses, reward_model):
    """
    计算每个回复的综合奖励分数。
    
    奖励组成：
    1. 长度奖励：回复长度在 [20, 800] 范围得 +0.5，否则 -0.5
    2. 思考格式奖励（含 </think> 标签时）：
       - 思考内容长度合适 [20, 300] 得 +1.0，否则 -0.5
       - 只有一个 </think> 得 +0.25，否则 -0.25
    3. 重复惩罚：基于 n-gram 重复度扣分
    4. Reward Model 评分：外部奖励模型的质量评估
    
    注意：PPO 中每个 prompt 只生成 1 个回复（num_generations=1），
    而 GRPO 中每个 prompt 生成多个回复用于组内比较。
    
    Args:
        prompts: prompt 文本列表 [B]
        responses: 回复文本列表 [B]（PPO 中 B 即 batch_size）
        reward_model: 奖励模型实例
    
    Returns:
        rewards: [B] 每个回复的奖励分数
    """
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            # 从 ChatML 格式的 prompt 中解析出对话消息列表
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            answer = response

            # 规则1: 长度奖励
            rewards[i] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5

            # 规则2: 思考格式奖励
            if '</think>' in response:
                thinking_content, answer_content = response.split('</think>', 1)
                rewards[i] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                rewards[i] += 0.25 if response.count('</think>') == 1 else -0.25
                answer = answer_content.strip()

            # 规则3: 重复惩罚
            rewards[i] -= rep_penalty(answer)

            # 规则4: Reward Model 评分
            score = reward_model.get_score(messages, answer)
            reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def ppo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step=0, wandb=None, use_sglang=False):
    """
    执行一个 epoch 的 PPO 训练。
    
    PPO 训练的完整流程（每个 outer step）：
    ┌──────────────────────────────────────────────────────┐
    │ Phase 1: Rollout（数据收集）                          │
    │   - Actor 生成回复                                    │
    │   - 记录 old_logp（旧策略概率）                       │
    │   - Reward Model 打分                                │
    │   - Critic 估计 V(s)                                 │
    │   - GAE 计算 advantage 和 returns                    │
    ├──────────────────────────────────────────────────────┤
    │ Phase 2: PPO Update（多轮策略更新）                   │
    │   for ppo_epoch in range(ppo_update_iters):          │
    │     for mini_batch in shuffle(data):                  │
    │       - 计算新策略的 logp 和 ratio                    │
    │       - PPO-clip Actor loss                           │
    │       - Clipped Value loss                            │
    │       - KL early stopping                             │
    │       - 梯度更新 Actor + Critic                       │
    └──────────────────────────────────────────────────────┘
    
    Args:
        epoch: 当前 epoch 编号
        loader: 数据加载器
        iters: 总步数
        rollout_engine: 推理引擎
        ref_model: 参考模型（冻结）
        actor_scheduler: Actor 学习率调度器
        critic_scheduler: Critic 学习率调度器
        reward_model: 奖励模型
        start_step: 起始步数（续训用）
        wandb: 实验跟踪实例
        use_sglang: 是否使用 SGLang 引擎
    """
    actor_model.train()
    critic_model.train()
    grad_accum_step = 0  # 梯度累积计数器

    for step, batch in enumerate(loader, start=start_step + 1):
        # =================================================================
        # Phase 1: Rollout —— 数据收集阶段
        # =================================================================
        prompts = batch["prompt"]  # list[str], length B

        # 对 prompt 进行 tokenize（左填充对齐）
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_seq_len,
                        padding_side="left").to(args.device)  # input_ids: [B, P], attention_mask: [B, P]

        # 使用 Rollout 引擎生成回复（PPO 中 num_generations=1，每个 prompt 只生成一个回复）
        rollout_result = rollout_engine.rollout(
            prompt_ids=enc.input_ids,
            attention_mask=enc.attention_mask,
            num_generations=1,          # PPO 只需要一个回复（不需要组内比较）
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        gen_out = rollout_result.output_ids          # [B, P+R] 完整序列
        completion_ids = rollout_result.completion_ids # [B, R] 生成部分
        prompt_lens = rollout_result.prompt_lens.to(args.device)  # [B] prompt 长度
        responses_text = rollout_result.completions   # list[str] 解码后的回复文本
        old_resp_logp = rollout_result.per_token_logps.to(args.device)  # [B, R] 旧策略 log prob

        # 计算奖励
        rewards = calculate_rewards(prompts, responses_text, reward_model)  # [B]

        # [可选] Debug 模式打印采样信息
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                Logger(f"[DEBUG] prompt_len={prompt_lens[i].item()}, response_len={len(responses_text[i])}")
                Logger(f"{'=' * 28} [DEBUG] sample[{i}] RESPONSE_BEGIN {'=' * 28}")
                Logger(responses_text[i])
                Logger(f"{'=' * 29} [DEBUG] sample[{i}] RESPONSE_END {'=' * 29}")
                Logger(f"[DEBUG] reward={rewards[i].item():.4f}")
                Logger('='*100)

        # =================================================================
        # 构建各种 mask 和索引
        # =================================================================
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # [B, P+R] 完整序列有效掩码
        labels = gen_out[:, 1:].clone()  # [B, P+R-1] 用于计算 next-token logp 的目标
        B = len(prompts)
        resp_labels = completion_ids  # [B, R]
        resp_idx = torch.arange(resp_labels.size(1), device=gen_out.device).unsqueeze(0)  # [1, R]
        # logp_pos: completion 部分在完整序列中对应的 logits 位置索引
        logp_pos = prompt_lens.unsqueeze(1) - 1 + resp_idx  # [B, R]

        # 构建 response 有效位置掩码（截止到 EOS token）
        resp_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        resp_lengths = resp_pad_mask.sum(dim=1)  # [B] 每个回复的原始有效长度
        valid_resp = resp_lengths > 0             # [B] 标记非空回复
        eos_mask = resp_labels.eq(tokenizer.eos_token_id) & resp_pad_mask  # EOS 位置
        has_eos = eos_mask.any(dim=1)             # [B] 是否包含 EOS
        eos_pos = torch.argmax(eos_mask.int(), dim=1)  # [B] 第一个 EOS 的位置
        # 最终有效长度：有 EOS 则截止到 EOS（含），否则取原始长度
        resp_lengths = torch.where(has_eos, eos_pos + 1, resp_lengths).long().clamp(min=1)
        # resp_policy_mask: Actor 策略损失的有效 token 掩码
        resp_policy_mask = ((resp_idx < resp_lengths.unsqueeze(1)) & resp_pad_mask).float()  # [B, R]
        # resp_value_mask: Critic 价值损失的有效 token 掩码（与 policy_mask 相同）
        resp_value_mask = resp_policy_mask.clone()

        # =================================================================
        # GAE (Generalized Advantage Estimation) 计算
        # =================================================================
        # 以下计算都不需要梯度（属于 Rollout 阶段的统计量）
        with torch.no_grad():
            # --- Critic 前向推理，获取 old_values ---
            critic_for_rollout = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model
            values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)  # [B, P+R]
            # 只取 completion 部分对应位置的价值估计
            old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask  # [B, R]
            
            # --- Reference Model 前向推理，获取 ref_logp（用于 KL 惩罚）---
            ref_resp_logp = F.log_softmax(ref_model(input_ids=gen_out, attention_mask=full_mask).logits[:, :-1], dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)  # [B, R]

            # --- 构造 token-level rewards ---
            # PPO 中奖励是稀疏的：只在回复的最后一个 token 位置给予外部奖励
            token_rewards = torch.zeros_like(old_resp_logp)  # [B, R] 全零
            last_idx = resp_lengths - 1  # [B] 每个回复的最后有效位置
            # 在最后一个有效 token 处添加外部奖励
            token_rewards[torch.arange(B, device=args.device)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]

            # --- GAE 逆序递推计算优势 ---
            # GAE 公式：
            #   δ_t = r_t + γ * V(s_{t+1}) - V(s_t)         （TD 误差）
            #   A_t = δ_t + γλ * A_{t+1}                     （GAE 递推）
            # 其中 γ 是折扣因子，λ 是 GAE lambda 平滑参数
            gen_len = old_resp_values.size(1)  # R
            lastgaelam = torch.zeros(B, device=args.device)  # GAE 递推的末尾项
            advs_rev = []  # 逆序存储 advantage
            for t in reversed(range(gen_len)):
                # 下一时刻的价值（最后一步之后为 0）
                nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
                # TD 误差: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
                delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
                # GAE 递推: A_t = δ_t + γ * λ * A_{t+1}
                lastgaelam = delta + args.gamma * args.lam * lastgaelam
                advs_rev.append(lastgaelam)
            # 翻转得到正序的 advantage
            advantages = torch.stack(advs_rev[::-1], dim=1)  # [B, R]
            # Returns = Advantage + Value（用作 Critic 的回归目标）
            returns = advantages + old_resp_values  # [B, R]

            # --- Advantage 标准化（减均值除标准差）---
            # 只在有效 token 位置上计算统计量
            adv_mean = (advantages * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            adv_var = ((advantages - adv_mean) ** 2 * resp_policy_mask).sum() / resp_policy_mask.sum().clamp(min=1)
            advantages = (advantages - adv_mean) * torch.rsqrt(adv_var + 1e-8) * resp_policy_mask

        # =================================================================
        # Phase 2: PPO Update —— 多轮 Mini-batch 策略更新
        # =================================================================
        # PPO 的特点：同一批 rollout 数据可以重复利用多次（ppo_update_iters 轮）
        # 每轮内对数据随机打乱并按 mini_batch_size 切分更新
        mb_size = max(1, min(args.mini_batch_size, B))
        stop_ppo = False  # KL early stopping 标志

        # 统计量累加器
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        kl_sum = 0.0
        kl_ref_sum = 0.0
        clipfrac_sum = 0.0
        aux_loss_sum = 0.0
        log_count = 0

        # 解包 DDP 获取原始模型
        actor_unwrapped = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
        critic_unwrapped = critic_model.module if isinstance(critic_model, DistributedDataParallel) else critic_model

        for ppo_epoch in range(args.ppo_update_iters):
            if stop_ppo:
                break
            # 在每个 PPO epoch 内随机打乱样本顺序
            b_inds = torch.randperm(B, device=args.device)

            for i in range(0, B, mb_size):
                inds = b_inds[i:i + mb_size]  # 当前 mini-batch 的索引
                
                # --- Critic 前向推理（需要梯度）---
                # 用当前 Critic 重新估计 mini-batch 样本的价值
                mb_values_seq = critic_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                mb_resp_values = mb_values_seq.gather(1, logp_pos[inds])  # [mb, R]

                # --- Actor 前向推理（需要梯度）---
                with autocast_ctx:
                    res = actor_unwrapped(input_ids=gen_out[inds], attention_mask=full_mask[inds])
                    # MoE 辅助损失
                    aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)

                # 计算当前策略在 completion token 位置的 log probability
                mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1).gather(2, labels[inds].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos[inds])  # [mb, R]
                
                # --- 计算重要性采样比率和近似 KL 散度 ---
                log_ratio = mb_resp_logp - old_resp_logp[inds]  # log(π_θ / π_old)
                # 近似 KL 散度（使用二阶近似 0.5 * (log_ratio)^2）
                approx_kl = (0.5 * (log_ratio ** 2) * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                
                # 多卡同步 KL 值，防止某张卡 break 而其他卡继续（导致 DDP 死锁）
                approx_kl_val = approx_kl.detach().clone()
                if dist.is_initialized():
                    dist.all_reduce(approx_kl_val, op=dist.ReduceOp.AVG)
                    
                # KL Early Stopping：当策略变化过大时停止更新
                if approx_kl_val > args.early_stop_kl:
                    stop_ppo = True
                
                # --- 计算 Actor 策略损失（PPO-clip）---
                ratio = torch.exp(log_ratio)  # π_θ / π_old
                # 统计被 clip 的比例（用于监控训练状态）
                clipfrac = ((((ratio - 1.0).abs() > args.clip_epsilon).float() * resp_policy_mask[inds]).sum()
                            / resp_policy_mask[inds].sum().clamp(min=1))

                # KL 惩罚项：使用 k3 estimator (exp(x) - x - 1)
                # 惩罚当前策略偏离参考模型的程度
                kl_ref_penalty = ((torch.exp(ref_resp_logp[inds] - mb_resp_logp) - (ref_resp_logp[inds] - mb_resp_logp) - 1.0)
                                  * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)

                # PPO-clip 策略损失：
                # L = max(-A * ratio, -A * clip(ratio, 1-ε, 1+ε))
                # 注意这里用 max 而非 min，因为 advantage 已经考虑了符号
                policy_loss = ((torch.max(-advantages[inds] * ratio,
                                          -advantages[inds] * torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon))
                               * resp_policy_mask[inds]).sum() / resp_policy_mask[inds].sum().clamp(min=1)
                               + args.kl_coef * kl_ref_penalty)  # 加上 KL 惩罚

                # --- 计算 Critic 价值损失（Clipped Value Loss）---
                # 双重 clip：防止 Critic 的价值估计变化过大
                # L_V = 0.5 * max((V - R)^2, (clip(V, V_old±ε) - R)^2)
                value_loss = 0.5 * (torch.max((mb_resp_values - returns[inds]) ** 2,
                                              (torch.clamp(mb_resp_values, old_resp_values[inds] - args.cliprange_value,
                                                           old_resp_values[inds] + args.cliprange_value) - returns[inds]) ** 2)
                                    * resp_value_mask[inds]).sum() / resp_value_mask[inds].sum().clamp(min=1)

                kl = approx_kl_val
                kl_ref = kl_ref_penalty.detach()

                # 如果 early stop 触发，仍然做一次 forward-backward 以维持 DDP 通信
                # 但将 loss 乘以 0，不产生实际梯度
                if stop_ppo:
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0
                else:
                    # 总损失 = Actor损失 + vf_coef * Critic损失 + MoE辅助损失
                    loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
                
                loss.backward()

                # 累加统计量
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                kl_sum += kl.item()
                kl_ref_sum += kl_ref.item()
                clipfrac_sum += clipfrac.item()
                aux_loss_sum += aux_loss.item()
                log_count += 1

                grad_accum_step += 1

                # 梯度累积达到步数时，执行参数更新
                if grad_accum_step % args.accumulation_steps == 0:
                    clip_grad_norm_(actor_model.parameters(), args.grad_clip)
                    clip_grad_norm_(critic_model.parameters(), args.grad_clip)
                    actor_optimizer.step()
                    critic_optimizer.step()
                    actor_scheduler.step()
                    critic_scheduler.step()
                    actor_optimizer.zero_grad()
                    critic_optimizer.zero_grad()

        # PPO epoch 结束后，如果还有未消耗的累积梯度，执行最后一次更新
        if grad_accum_step % args.accumulation_steps != 0:
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()
            critic_scheduler.step()
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
        
        # 同步最新策略权重到 Rollout 引擎
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(actor_model)

        # =================================================================
        # 日志记录
        # =================================================================
        if is_main_process():
            critic_loss_val = value_loss_sum / max(log_count, 1)
            reward_val = rewards.mean().item()
            approx_kl_val = kl_sum / max(log_count, 1)
            kl_ref_val = kl_ref_sum / max(log_count, 1)
            clipfrac_val = clipfrac_sum / max(log_count, 1)
            avg_len_val = resp_lengths.float().mean().item()
            actor_lr, critic_lr = actor_optimizer.param_groups[0]['lr'], critic_optimizer.param_groups[0]['lr']

            if wandb is not None:
                wandb.log({
                    "reward": reward_val,
                    "kl_ref": kl_ref_val,
                    "approx_kl": approx_kl_val,
                    "clipfrac": clipfrac_val,
                    "critic_loss": critic_loss_val,
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                    "critic_lr": critic_lr,
                })

            Logger(f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                   f"Reward: {reward_val:.4f}, KL_ref: {kl_ref_val:.4f}, Approx KL: {approx_kl_val:.4f}, "
                   f"ClipFrac: {clipfrac_val:.4f}, Critic Loss: {critic_loss_val:.4f}, "
                   f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}")

        # =================================================================
        # 模型保存
        # =================================================================
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            # 保存 Actor 权重为半精度
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 保存完整训练状态（含 Critic、优化器等，支持断点续训）
            lm_checkpoint(lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                         scheduler=actor_scheduler, critic_model=critic_model, 
                         critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler)
            actor_model.train()
            del actor_state

        # =================================================================
        # 清理显存
        # =================================================================
        del enc, gen_out, completion_ids, responses_text, rewards, full_mask, values_seq, advantages
        del labels, resp_labels, resp_idx, resp_pad_mask, valid_resp, eos_mask, has_eos, eos_pos, resp_lengths, resp_policy_mask, resp_value_mask, old_resp_logp, ref_resp_logp
        del kl, kl_ref, policy_loss, value_loss, loss, token_rewards, returns, old_resp_values, prompt_lens, logp_pos


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    # ========== 命令行参数定义 ==========
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    # --- 基础训练参数 ---
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=5e-7, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    # --- 模型架构参数 ---
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    # --- 数据和奖励 ---
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    # --- PPO 核心超参数 ---
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO裁剪参数（策略比率的 clip 范围）")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function损失系数（平衡 Actor 和 Critic 的 loss 权重）")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数（控制策略偏离参考模型的惩罚强度）")
    parser.add_argument("--gamma", type=float, default=1.0, help="GAE折扣因子（1.0 表示无折扣，适合对话场景）")
    parser.add_argument("--lam", type=float, default=0.95, help="GAE lambda参数（权衡 bias 和 variance）")
    parser.add_argument("--cliprange_value", type=float, default=0.2, help="Value function裁剪范围（防止价值估计剧变）")
    parser.add_argument("--ppo_update_iters", type=int, default=2, help="同一批rollout数据重复更新次数")
    parser.add_argument("--early_stop_kl", type=float, default=0.25, help="PPO early stop 的 KL 阈值")
    parser.add_argument("--mini_batch_size", type=int, default=2, help="PPO每次更新的minibatch大小")
    # --- 权重和续训 ---
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    # --- 实验跟踪 ---
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    # --- 加速和调试 ---
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    # --- Rollout 引擎参数 ---
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_ppo", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查 checkpoint ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=bool(args.use_moe))
    # 如果开启续训，尝试加载已有 checkpoint
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置实验跟踪 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化四个模型 ==========
    base_weight = args.from_weight

    # ──────────────────────────────────────────────────────────────────
    # PPO 四个模型的 DDP 并行策略：
    #
    #   模型              是否 DDP    原因
    #   ─────────────────────────────────────────────────────────────
    #   Actor Model       ✅ DDP     需要训练，梯度需跨卡同步
    #   Critic Model      ✅ DDP     需要训练，梯度需跨卡同步
    #   Reference Model   ❌ 不DDP   冻结参数，仅前向推理，无梯度同步需求
    #   Reward Model      ❌ 不DDP   外部模型，仅推理打分，无梯度同步需求
    #
    # Q: 非 DDP 的模型也是每张卡都加载一份吗？
    # A: 是的。DDP 是数据并行（Data Parallel），每张卡都持有完整的模型副本。
    #    区别在于：
    #    - DDP 模型：每张卡加载一份，训练时各卡计算不同 mini-batch 的梯度，
    #      然后通过 all-reduce 通信同步梯度，保证各卡参数一致。
    #    - 非 DDP 模型：每张卡也加载一份（相同权重），但只做前向推理，
    #      不需要 backward 也不需要卡间通信。各卡独立推理各自负责的数据子集，
    #      互不干扰，天然并行。
    #
    #    所以 4 个模型 × N 张卡 = 4N 个模型副本同时存在于显存中，
    #    这也是 PPO 显存占用远高于 GRPO（仅 3 个模型）的原因。
    #
    # DDP 包装在后续第7步完成（需在 optimizer 创建之后、训练开始之前）。
    # ──────────────────────────────────────────────────────────────────

    # Actor 模型（策略模型，待训练）→ 后续会被 DDP 包装
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)

    # Reference 模型（冻结的 SFT 基线，用于 KL 惩罚）→ 不做 DDP，仅推理
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # Critic 模型（价值网络）→ 后续会被 DDP 包装
    # 使用 SFT 模型权重初始化骨干，然后加载到 CriticModel 中
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    critic_model = CriticModel(lm_config)
    # strict=False: 因为 CriticModel 多了 value_head，少了 lm_head
    critic_model.load_state_dict(state_dict, strict=False)
    critic_model = critic_model.to(args.device)

    # Reward 模型（外部奖励模型，如 InternLM2-1.8B-Reward）→ 不做 DDP，仅推理打分
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)

    # Rollout 引擎（负责 Actor 的推理生成）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=actor_model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )

    # ========== 6. 数据集和优化器 ==========
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len), thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None

    # Actor 和 Critic 使用独立的优化器（不同学习率）
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)

    # 计算总优化步数（考虑 PPO 多轮更新和 mini-batch）
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    mb_factor = max(1, math.ceil(args.batch_size / args.mini_batch_size))  # 每个 rollout step 内的 mini-batch 数
    total_optimizer_steps = math.ceil(iters * args.epochs * args.ppo_update_iters * mb_factor / args.accumulation_steps)

    # 余弦退火学习率调度
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)

    # 从 checkpoint 恢复训练状态
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(actor_model)
    if dist.is_initialized():
        # DDP 包装 Actor 和 Critic
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
    rollout_engine.update_policy(actor_model)
    
    # ========== 8. 开始训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            ppo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, actor_scheduler, critic_scheduler, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布式进程 ==========
    if dist.is_initialized(): dist.destroy_process_group()
