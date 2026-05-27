"""
===================================================================================
Rollout Engine - GRPO/PPO 训练中的策略推理引擎
===================================================================================

本模块实现了 RLHF 训练中 "Rollout"（策略采样/推理）环节的可插拔引擎架构。

在 GRPO/PPO 等强化学习训练中，每一步都需要：
1. 用当前策略模型对 prompt 进行采样生成（rollout）
2. 记录生成过程中每个 token 的 log probability（用于后续计算重要性采样比率）
3. 训练更新策略后，同步最新权重到推理引擎

本模块提供两种引擎实现：
- TorchRolloutEngine: 使用 PyTorch 原生 generate() 进行推理，简单可靠
- SGLangRolloutEngine: 通过 HTTP API 调用 SGLang 推理服务器，支持更高效的
  批量推理（continuous batching、PagedAttention 等优化）

架构设计采用策略模式（Strategy Pattern）+ 工厂模式（Factory Pattern），
使得训练代码可以无缝切换推理后端。

如果使用 SGLang 加速，需通过以下命令首先启动（transformers格式）模型：
python -m sglang.launch_server --model-path ./minimind-3 --attention-backend triton --host 0.0.0.0 --port 8998
===================================================================================
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import torch
import torch.distributed as dist
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer


def compute_per_token_logps(model, input_ids: Tensor, n_keep: int, attention_mask: Optional[Tensor] = None) -> Tensor:
    """
    计算序列中最后 n_keep 个 token 的 log probability。
    
    这是 GRPO/PPO 中的关键计算：在 rollout 阶段，我们需要记录当前策略
    生成每个 token 时的概率（即 π_old(a_t|s_t)），用于后续训练时计算
    重要性采样比率 ratio = π_θ / π_old。
    
    计算过程：
    1. 将完整序列（prompt + completion）输入模型，获取 logits
    2. 对 logits 做 log_softmax 得到 log probability 分布
    3. 用 gather 操作取出实际生成 token 对应的 log prob
    
    Args:
        model: 策略模型（可能被 DDP 包装）
        input_ids: 完整输入序列 [B, seq_len]，包含 prompt + completion
        n_keep: 需要计算 logp 的 token 数量（即 completion 长度）
        attention_mask: 注意力掩码 [B, seq_len]，标记有效 token 位置
    
    Returns:
        per_token_logps: [B, n_keep]，每个 completion token 的 log probability
    
    ==================== 详细示例 ====================
    
    假设 vocab_size=5（词表只有5个token），B=1，序列如下：
    
      input_ids = [[10, 20, 30, 40, 50]]   # 完整序列，长度 seq_len=5
                    |-- prompt --|-- completion --|
                    [10, 20, 30]  [40, 50]
      
      n_keep = 2（只需要计算 completion 部分 [40, 50] 的 logp）
    
    Step 1: 模型前向推理（logits_to_keep=3）
    ─────────────────────────────────────────
      模型输入整个序列 [10, 20, 30, 40, 50]，但只返回最后 3 个位置的 logits：
      
      logits = model(input_ids, logits_to_keep=3).logits
      # 形状: [1, 3, vocab_size] = [1, 3, 5]
      # 对应位置 2, 3, 4 的 logits（即预测 token 30, 40, 50 之后的下一个 token）
      
      然后 [:, :-1, :] 去掉最后一个位置（位置4预测的是 token 50 之后的内容，没有 target）：
      logits = logits[:, :-1, :]
      # 形状: [1, 2, 5]
      # logits[0, 0, :] → 位置2的logits → 预测位置3的token（即 token 40）
      # logits[0, 1, :] → 位置3的logits → 预测位置4的token（即 token 50）
    
    Step 2: 取出 completion 部分的 token ids 作为索引
    ─────────────────────────────────────────────────
      ids_row = input_ids[:, -n_keep:] = input_ids[:, -2:] = [40, 50]
      
      这就是我们需要查询 logp 的目标 token。
    
    Step 3: log_softmax + gather 取出对应 logp
    ───────────────────────────────────────────
      假设 logits[0] 为（简化数值）：
        位置0: [2.0, 0.5, 1.0, 3.0, 0.1]  → log_softmax → [-1.5, -3.0, -2.5, -0.5, -3.4]
        位置1: [0.2, 1.0, 0.5, 0.3, 2.5]  → log_softmax → [-2.8, -2.0, -2.5, -2.7, -0.5]
      
      gather 操作：用 ids_row 作为索引从 log_softmax 结果中取值
        - 位置0，目标 token=40（假设 token id 40 对应 vocab 索引 3）→ logp = -0.5
        - 位置1，目标 token=50（假设 token id 50 对应 vocab 索引 4）→ logp = -0.5
      
      最终结果: per_token_logps = [[-0.5, -0.5]]  形状 [1, 2]
    
    ==================== 关键理解 ====================
    
    为什么是 "位置 i 的 logits 对应预测 token i+1"？
    
      因为语言模型是自回归的：给定 [t0, t1, t2, t3, t4]，
      位置 i 的输出 logits 是在看到 [t0, ..., ti] 后预测 t_{i+1} 的概率分布。
      
      所以：
        logits[位置2] 预测的是 token[位置3] = 40 的概率
        logits[位置3] 预测的是 token[位置4] = 50 的概率
      
      这正是我们要计算的：模型生成 token 40 和 50 时的概率。
    
    为什么要用 logits_to_keep 参数？
    
      如果序列很长（如 prompt=768 + completion=1024 = 1792），但我们只需要
      completion 部分（1024个位置）的 logp。使用 logits_to_keep=1025 可以让
      模型只输出最后 1025 个位置的 logits，而不是全部 1792 个位置，
      从而节省约 (1792-1025)/1792 ≈ 43% 的 logits 显存。
    ==================================================
    """
    # 如果不需要保留任何 token 的 logp，直接返回空 tensor
    if n_keep <= 0:
        return input_ids.new_empty((input_ids.size(0), 0), dtype=torch.float32)

    # 解包 DDP 包装，获取原始模型
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model

    # 处理 inference mode 下的 tensor（inference mode 不允许原地操作）
    input_ids = input_ids.detach().clone() if input_ids.is_inference() else input_ids

    # 前向推理，logits_to_keep 参数只保留最后 n_keep+1 个位置的 logits 以节省显存
    # 为什么是 n_keep+1？因为要预测 n_keep 个 completion token，
    # 需要从 completion 前一个位置开始的 n_keep+1 个 logits，去掉最后一个后正好 n_keep 个
    # 示例: n_keep=2, logits_to_keep=3 → 得到 [1, 3, V] → [:, :-1, :] → [1, 2, V]
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]

    # 逐样本计算每个 token 的 log probability
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        # logits_row: [n_keep, vocab_size] —— 每个位置预测下一个 token 的概率分布
        # ids_row: [n_keep] —— completion 部分的实际 token ids（作为 gather 索引）
        ids_row = ids_row.detach().clone() if ids_row.is_inference() else ids_row
        # log_softmax: 将 logits 归一化为 log 概率分布
        # gather: 从每个位置的 log 概率分布中，取出实际生成的那个 token 对应的 logp
        # ids_row.unsqueeze(1): [n_keep] → [n_keep, 1]（作为 dim=1 的索引）
        # gather 后 squeeze: [n_keep, 1] → [n_keep]
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)  # [B, n_keep]


@dataclass
class RolloutResult:
    """
    Rollout 结果的数据容器。
    
    封装了一次 rollout（策略采样）的所有输出，供训练循环使用。
    
    Attributes:
        output_ids: [B*num_gen, P+R] 完整序列（prompt + completion）的 token ids
        completion_ids: [B*num_gen, R] 仅 completion（生成部分）的 token ids
        per_token_logps: [B*num_gen, R] 生成时每个 token 的 log probability（旧策略）
        completions: list[str] 解码后的回复文本，长度为 B*num_gen
        prompt_lens: [B*num_gen] 每条样本的 prompt 实际长度
        completion_mask: [B*num_gen, R] completion 的有效位置掩码（1=有效, 0=padding）
    
    其中：
        B = batch_size（prompt 数量）
        num_gen = num_generations（每个 prompt 生成的候选数）
        P = prompt 长度
        R = completion 最大长度（右侧 padding 对齐）
    """
    output_ids: Tensor
    completion_ids: Tensor
    per_token_logps: Tensor
    completions: List[str]
    prompt_lens: Tensor
    completion_mask: Tensor


class RolloutEngine(ABC):
    """
    Rollout 引擎抽象基类。
    
    定义了所有推理引擎必须实现的接口：
    - rollout(): 对给定 prompt 生成多个候选回复并返回相关统计量
    - update_policy(): 训练更新后同步最新策略权重到推理引擎
    
    这种抽象设计使得训练代码与具体推理实现解耦，
    可以轻松替换推理后端而不影响训练逻辑。
    """
    tokenizer = None
    
    @abstractmethod
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """
        执行策略采样（rollout）。
        
        Args:
            prompt_ids: [B, P] prompt 的 token ids（可能含左侧 padding）
            attention_mask: [B, P] prompt 的 attention mask
            num_generations: 每个 prompt 生成的候选回复数量
            max_new_tokens: 每个回复最多生成的 token 数
            temperature: 采样温度，越高越随机
        
        Returns:
            RolloutResult 数据容器
        """
        pass
    
    @abstractmethod
    def update_policy(self, model: torch.nn.Module):
        """
        将训练更新后的策略模型权重同步到推理引擎。
        
        对于 TorchRolloutEngine：直接更新模型引用
        对于 SGLangRolloutEngine：保存权重到磁盘，通知 SGLang 服务器重新加载
        """
        pass


class TorchRolloutEngine(RolloutEngine):
    """
    PyTorch 原生推理引擎。
    
    直接使用 HuggingFace 风格的 model.generate() 方法进行自回归采样。
    优点：简单可靠，无需额外部署推理服务
    缺点：不支持 continuous batching 等推理优化，大规模生成时速度较慢
    
    适用场景：小模型、单卡训练、快速原型验证
    """
    
    def __init__(self, policy_model: torch.nn.Module, tokenizer, device: str = "cuda", autocast_ctx=None):
        """
        Args:
            policy_model: 策略模型实例（可能已被 DDP 包装）
            tokenizer: 分词器实例
            device: 计算设备
            autocast_ctx: 混合精度上下文（如 torch.cuda.amp.autocast）
        """
        self.policy_model = policy_model
        self.tokenizer = tokenizer
        self.device = device
        self.autocast_ctx = autocast_ctx
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """
        使用 PyTorch 原生 generate 进行策略采样。
        
        流程：
        1. 将每个 prompt 复制 num_generations 份（repeat_interleave）
        2. 调用 model.generate() 进行自回归采样生成
        3. 分离出 completion 部分
        4. 调用 compute_per_token_logps 计算每个生成 token 的 log prob
        5. 解码生成文本并打包返回
        """
        # 解包 DDP 获取原始模型
        model = self.policy_model.module if isinstance(self.policy_model, DistributedDataParallel) else self.policy_model
        ctx = self.autocast_ctx if self.autocast_ctx else nullcontext()

        with torch.no_grad(), ctx:
            # 将每个 prompt 重复 num_generations 次
            # [B, P] -> [B*num_gen, P]，每个 prompt 对应 num_gen 个生成
            output_ids = model.generate(
                input_ids=prompt_ids.repeat_interleave(num_generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(num_generations, dim=0),
                max_new_tokens=max_new_tokens,
                do_sample=True,          # 启用随机采样（非贪心）
                temperature=temperature,  # 采样温度
                num_return_sequences=1,   # 每次调用只返回1个序列（已通过 repeat 扩展）
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            ).clone()  # [B*num_gen, P+R]，clone() 防止内存引用问题

            # 分离 completion 部分（去掉 prompt 前缀）
            prompt_len = prompt_ids.size(1)
            completion_ids = output_ids[:, prompt_len:]  # [B*num_gen, R]

            # 构建完整序列的 attention mask（非 pad 位置为 1）
            # 注意：full_mask 同时覆盖了两种 padding：
            #   1. 输入（prompt）部分的左侧 padding（因为 batch 内 prompt 长度不同，短的左填充对齐）
            #   2. 输出（completion）部分的右侧 padding（因为各序列生成长度不同，短的右填充对齐）
            # 示例：
            #   output_ids = [PAD, PAD, t0, t1, t2, g0, g1, EOS, PAD, PAD]
            #   full_mask  = [0,   0,   1,  1,  1,  1,  1,  1,   0,   0  ]
            # 后续传给模型做 attention 时，必须屏蔽这两端的 pad，否则 logprobs 计算会不准确
            full_mask = (output_ids != self.tokenizer.pad_token_id).long()

            # 计算旧策略下每个 completion token 的 log probability
            # 这是后续计算 importance sampling ratio 的关键
            per_token_logps = compute_per_token_logps(self.policy_model, output_ids, completion_ids.size(1), attention_mask=full_mask)

        # 将 completion token ids 解码为文本字符串
        completions = self.tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

        return RolloutResult(
            output_ids,                    # 完整序列
            completion_ids,                # completion 部分
            per_token_logps,               # 旧策略 log prob
            completions,                   # 解码后的文本
            # prompt_lens: 所有样本的 prompt 长度相同（因为已左填充对齐）
            prompt_ids.new_full((output_ids.size(0),), prompt_len),
            # completion_mask: 全 1（PyTorch generate 已通过 eos 停止，无需额外 mask）
            attention_mask.new_ones(output_ids.size(0), completion_ids.size(1))
        )
    
    def update_policy(self, model: torch.nn.Module):
        """
        更新策略模型引用。
        
        TorchRolloutEngine 直接持有模型引用，训练时 model 参数已原地更新，
        所以这里只需更新引用指针即可（处理 DDP 重新包装等情况）。
        """
        self.policy_model = model


class SGLangRolloutEngine(RolloutEngine):
    """
    SGLang HTTP API 推理引擎。
    
    通过 HTTP 请求调用独立部署的 SGLang 推理服务器进行采样生成。
    
    优点：
    - 支持 continuous batching、PagedAttention 等推理优化
    - 推理和训练可以使用不同 GPU，实现计算资源解耦
    - 支持更大的并发生成吞吐量
    
    缺点：
    - 需要额外部署和维护推理服务
    - 权重同步需要通过磁盘传输，有额外 I/O 开销
    - 网络通信引入延迟
    
    适用场景：大规模 GRPO 训练、多卡/多机环境
    
    权重同步流程：
    1. 训练进程将最新权重保存到共享磁盘路径
    2. 通过 HTTP API 通知 SGLang 服务器从磁盘加载新权重
    3. 分布式环境中通过 broadcast + barrier 确保所有 rank 同步
    """
    
    def __init__(self, base_url: str, model_path: str, shared_ckpt_path: str = "./sglang_ckpt", timeout: int = 120):
        """
        Args:
            base_url: SGLang 推理服务器的 URL（如 http://localhost:8998）
            model_path: tokenizer 所在路径（用于解码生成结果）
            shared_ckpt_path: 训练与推理共享的权重存储路径（用于权重同步）
            timeout: HTTP 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.shared_ckpt_path = shared_ckpt_path
        self.timeout = timeout
        # 加载 tokenizer 用于解码生成结果
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.http = requests
    
    def rollout(self, prompt_ids: Tensor, attention_mask: Tensor, num_generations: int, max_new_tokens: int, temperature: float = 0.8) -> RolloutResult:
        """
        通过 SGLang HTTP API 进行策略采样。
        
        流程：
        1. 去除左侧 padding，提取有效 token ids
        2. 构造 HTTP 请求 payload（含采样参数和 logprob 请求）
        3. 调用 SGLang /generate 接口进行批量生成
        4. 解析返回结果，提取 completion ids 和 log probabilities
        5. 统一 padding 并封装为 RolloutResult
        """
        # =====================================================================
        # Step 1: 去除左侧 padding，只保留有效 token
        # =====================================================================
        # prompt_ids 可能经过左填充对齐，需要还原每个 prompt 的实际 token
        input_ids_list = []
        for ids, mask in zip(prompt_ids, attention_mask):
            valid_ids = ids[mask.bool()].tolist()  # 只保留 mask=1 的位置
            input_ids_list.append(valid_ids)
        # 每个 prompt 复制 num_generations 份
        all_input_ids = [ids for ids in input_ids_list for _ in range(num_generations)]
        
        # =====================================================================
        # Step 2: 构造 SGLang HTTP 请求
        # =====================================================================
        payload = {
            "input_ids": all_input_ids,          # 直接传 token ids（避免重复 tokenize）
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else [],
            },
            "return_logprob": True,              # 请求返回每个 token 的 log probability
        }
        
        # =====================================================================
        # Step 3: 发送请求并获取结果
        # =====================================================================
        resp = self.http.post(f"{self.base_url}/generate", json=payload, timeout=self.timeout)
        resp.raise_for_status()  # 如果 HTTP 状态码非 2xx，抛出异常
        
        results = resp.json()
        if not isinstance(results, list):
            results = [results]  # 单条结果也统一为列表
        
        # =====================================================================
        # Step 4: 解析每个生成结果
        # =====================================================================
        all_output_ids, all_completion_ids, all_logprobs = [], [], []
        completions = []
        
        for i, result in enumerate(results):
            # 从返回的 meta_info 中提取 completion token ids 和 logprobs
            meta = result.get("meta_info", {})
            completion_ids = meta.get("output_ids", result.get("output_ids", []))
            raw_logprobs = meta.get("output_token_logprobs", [])
            
            # 解析 logprobs（SGLang 可能返回 list/tuple 或纯数值格式）
            logprobs = []
            for item in raw_logprobs:
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    logprobs.append(item[0])   # (logp, token_id) 格式，取第一个
                elif isinstance(item, (int, float)):
                    logprobs.append(item)      # 直接是数值
            
            # 对齐 logprobs 和 completion_ids 的长度
            if len(logprobs) < len(completion_ids):
                # logprobs 不足时，前面补 0.0
                logprobs = [0.0] * (len(completion_ids) - len(logprobs)) + logprobs
            elif len(logprobs) > len(completion_ids):
                # logprobs 过多时，只保留最后 len(completion_ids) 个
                logprobs = logprobs[-len(completion_ids):] if completion_ids else []

            # 拼接完整序列 = prompt + completion
            prompt = all_input_ids[i]
            full_output = prompt + completion_ids
            all_output_ids.append(full_output)
            all_completion_ids.append(completion_ids)
            all_logprobs.append(logprobs)
            # 解码 completion 为文本
            completions.append(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
        
        # =====================================================================
        # Step 5: 统一 padding 并封装结果
        # =====================================================================
        device = prompt_ids.device
        # 找到最大 completion 长度和最大完整序列长度，用于 padding 对齐
        max_comp_len = max(1, max(len(ids) for ids in all_completion_ids))
        max_out_len = max(len(ids) for ids in all_input_ids) + max_comp_len
        
        def pad_to_tensor(seqs, max_len, pad_val=0):
            """将不等长的序列列表右侧 padding 到统一长度，转为 tensor。"""
            return torch.tensor([s + [pad_val] * (max_len - len(s)) for s in seqs], device=device)
        
        pad_id = self.tokenizer.pad_token_id
        return RolloutResult(
            output_ids=pad_to_tensor(all_output_ids, max_out_len, pad_val=pad_id),
            completion_ids=pad_to_tensor(all_completion_ids, max_comp_len, pad_val=pad_id),
            per_token_logps=pad_to_tensor(all_logprobs, max_comp_len, pad_val=0.0),
            completions=completions,
            prompt_lens=torch.tensor([len(ids) for ids in all_input_ids], device=device),
            # completion_mask: 有效 token 位置为 1，padding 位置为 0
            completion_mask=torch.tensor([[1] * len(ids) + [0] * (max_comp_len - len(ids)) for ids in all_completion_ids], device=device),
        )
    
    def update_policy(self, model: torch.nn.Module):
        """
        将训练更新后的策略权重同步到 SGLang 推理服务器。
        
        同步流程：
        1. [Rank 0] 将模型权重保存到共享磁盘（半精度以节省空间和 I/O）
        2. [Rank 0] 同时保存 tokenizer（确保推理端配置一致）
        3. [Rank 0] 通过 HTTP POST 调用 SGLang 的 /update_weights_from_disk 接口
        4. [All Ranks] 通过 broadcast + barrier 同步操作结果
        5. 如果失败则抛出 RuntimeError
        
        Args:
            model: 当前训练后的策略模型（可能被 DDP 或 compile 包装）
        
        Returns:
            bool: 权重更新是否成功
        
        Raises:
            RuntimeError: 当权重同步失败时抛出
        """
        ok = True
        # 只在 rank 0 执行实际的保存和通知操作
        if not dist.is_initialized() or dist.get_rank() == 0:
            try:
                # 解包 DDP 和 torch.compile 的包装层
                unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
                unwrapped = getattr(unwrapped, '_orig_mod', unwrapped)  # 解包 compile
                abs_path = os.path.abspath(self.shared_ckpt_path)

                # 保存模型权重为半精度（fp16）到共享路径
                state_dict = {k: v.detach().half().cpu() for k, v in unwrapped.state_dict().items()}
                unwrapped.save_pretrained(abs_path, state_dict=state_dict, safe_serialization=False)
                self.tokenizer.save_pretrained(abs_path)

                # 通知 SGLang 服务器从磁盘加载新权重
                resp = self.http.post(f"{self.base_url}/update_weights_from_disk", json={"model_path": abs_path}, timeout=self.timeout)
                if resp.status_code != 200: print(f"[SGLANG WARNING] update_weights 失败: {resp.status_code}, {resp.text}")
                ok = resp.status_code == 200
            except Exception as e:
                print(f"[SGLANG WARNING] update_weights 异常: {e}"); ok = False

        # 分布式环境中，将 rank 0 的操作结果广播到所有 rank
        if dist.is_initialized():
            ok_t = torch.tensor(int(ok), device=next(model.parameters()).device)
            dist.broadcast(ok_t, src=0)  # rank 0 广播结果
            dist.barrier()                # 所有 rank 等待同步完成
            ok = bool(ok_t.item())

        if not ok: raise RuntimeError("SGLang update_policy failed")
        return ok
    
    def flush_cache(self) -> bool:
        """
        清空 SGLang 服务器的 KV cache。
        
        在权重更新后调用，确保旧权重的缓存不会影响新推理。
        
        Returns:
            bool: 操作是否成功
        """
        resp = self.http.post(f"{self.base_url}/flush_cache", timeout=30)
        return resp.status_code == 200
    
    def health(self) -> bool:
        """
        检查 SGLang 推理服务器的健康状态。
        
        Returns:
            bool: 服务器是否可用
        """
        try:
            resp = self.http.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except:
            return False


def create_rollout_engine(
    engine_type: str = "torch",
    policy_model: torch.nn.Module = None,
    tokenizer = None,
    device: str = "cuda",
    autocast_ctx = None,
    sglang_base_url: str = None,
    sglang_model_path: str = None,
    sglang_shared_path: str = None,
) -> RolloutEngine:
    """
    工厂函数：根据引擎类型创建对应的 Rollout 引擎实例。
    
    采用工厂模式，使训练代码与具体引擎实现解耦：
    - "torch": 创建 TorchRolloutEngine，使用 PyTorch 原生推理
    - "sglang": 创建 SGLangRolloutEngine，使用 SGLang 推理服务
    
    Args:
        engine_type: 引擎类型，"torch" 或 "sglang"
        policy_model: 策略模型实例（torch 引擎需要）
        tokenizer: 分词器实例（torch 引擎需要）
        device: 计算设备（torch 引擎需要）
        autocast_ctx: 混合精度上下文（torch 引擎需要）
        sglang_base_url: SGLang 服务器 URL（sglang 引擎需要）
        sglang_model_path: SGLang tokenizer 路径（sglang 引擎需要）
        sglang_shared_path: 权重共享磁盘路径（sglang 引擎需要）
    
    Returns:
        RolloutEngine 实例
    
    Raises:
        ValueError: 不支持的引擎类型
    """
    if engine_type == "torch":
        return TorchRolloutEngine(policy_model, tokenizer, device, autocast_ctx)
    elif engine_type == "sglang":
        return SGLangRolloutEngine(sglang_base_url, sglang_model_path, sglang_shared_path)
    else:
        raise ValueError(f"不支持的引擎类型: {engine_type}")
