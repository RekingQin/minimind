"""
===================================================================================
MiniMind Agent RL 训练脚本
===================================================================================

本脚本使用 GRPO/CISPO 算法训练语言模型的 Agent（工具调用）能力。

与普通 GRPO 训练的关键区别：
1. **多轮交互 Rollout**：Agent 需要在"生成 → 调用工具 → 获取结果 → 继续生成"
   的多轮循环中完成任务，而非单次生成
2. **工具调用奖励**：除了通用的格式/长度/RM 奖励外，还有工具调用正确性奖励
3. **混合 Mask**：response 中只有模型生成的部分参与梯度计算，
   工具返回结果（环境观察）不参与梯度

训练目标：让模型学会：
- 正确识别何时需要调用工具
- 生成格式正确的 <tool_call>...</tool_call> 标签
- 正确传递工具参数
- 在获取工具结果后给出准确的最终回答

支持的工具：
- calculate_math: 数学表达式计算
- unit_converter: 单位换算
- get_current_weather: 天气查询
- get_current_time: 时间查询
- get_exchange_rate: 汇率查询
- translate_text: 文本翻译

算法选择：GRPO/CISPO（与 train_grpo.py 相同的策略优化算法）
===================================================================================
"""

import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import re
import gc
import json
import math
import random
import signal
import argparse
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import AgentRLDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine, compute_per_token_logps

warnings.filterwarnings('ignore')

# ================================ 工具与 Reward = Start ================================

def rep_penalty(text, n=3, cap=0.5):
    """
    计算文本的 n-gram 重复惩罚。
    重复 trigram 越多惩罚越大，上限为 cap。
    """
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0

# ======== 工具定义 ========
# 定义 Agent 可调用的工具列表（符合 OpenAI function calling 格式）
# 这些工具定义会通过 tokenizer.apply_chat_template(tools=...) 注入到 system prompt 中
TOOLS = [
    {"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}}},
    {"type": "function", "function": {"name": "unit_converter", "description": "单位换算", "parameters": {"type": "object", "properties": {"value": {"type": "number"}, "from_unit": {"type": "string"}, "to_unit": {"type": "string"}}, "required": ["value", "from_unit", "to_unit"]}}},
    {"type": "function", "function": {"name": "get_current_weather", "description": "获取天气", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}},
    {"type": "function", "function": {"name": "get_current_time", "description": "获取时间", "parameters": {"type": "object", "properties": {"timezone": {"type": "string", "default": "Asia/Shanghai"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_exchange_rate", "description": "查询汇率", "parameters": {"type": "object", "properties": {"from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, "required": ["from_currency", "to_currency"]}}},
    {"type": "function", "function": {"name": "translate_text", "description": "翻译文本", "parameters": {"type": "object", "properties": {"text": {"type": "string"}, "target_language": {"type": "string"}}, "required": ["text", "target_language"]}}},
]

# ======== 模拟数据 ========
# 训练时不真正调用外部 API，而是使用预定义的模拟数据来模拟工具执行结果
# 这样训练环境是确定性的，不依赖外部服务
WEATHER_DATA = {"北京": ("28°C", "晴"), "上海": ("15°C", "多云"), "广州": ("32°C", "闷热"), "深圳": ("30°C", "晴"), "杭州": ("22°C", "阴"), "成都": ("18°C", "小雨"), "武汉": ("25°C", "多云"), "南京": ("20°C", "晴"), "西安": ("16°C", "大风"), "重庆": ("26°C", "阴"), "Tokyo": ("12°C", "晴"), "New York": ("8°C", "多云"), "London": ("5°C", "小雨"), "Paris": ("10°C", "阴"), "Sydney": ("25°C", "晴朗")}
TIME_DATA = {"Asia/Shanghai": "2025-03-07 14:30:00", "America/New_York": "2025-03-07 01:30:00", "Europe/London": "2025-03-07 06:30:00", "Asia/Tokyo": "2025-03-07 15:30:00", "Europe/Paris": "2025-03-07 07:30:00", "Australia/Sydney": "2025-03-07 17:30:00"}
EXCHANGE_DATA = {("USD", "CNY"): 7.21, ("EUR", "CNY"): 7.85, ("GBP", "CNY"): 9.12, ("JPY", "CNY"): 0.048, ("USD", "EUR"): 0.92, ("USD", "GBP"): 0.79, ("CNY", "JPY"): 20.83, ("AUD", "CNY"): 4.72}
TRANSLATE_DATA = {("你好世界", "english"): "Hello World", ("Good morning", "chinese"): "早上好", ("今天天气真好", "english"): "The weather is nice today", ("I love programming", "chinese"): "我喜欢编程", ("机器学习很有趣", "english"): "Machine learning is interesting", ("Happy birthday", "chinese"): "生日快乐"}
UNIT_DATA = {"km_miles": 0.621371, "miles_km": 1.60934, "kg_pounds": 2.20462, "pounds_kg": 0.453592, "meters_feet": 3.28084, "feet_meters": 0.3048, "celsius_fahrenheit": 1.8, "fahrenheit_celsius": 0.5556}

# ======== 模拟执行 ========
# 每个工具对应一个 lambda，接收参数 dict，返回结果 dict
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**").replace("×", "*").replace("÷", "/").replace("−", "-").replace("（", "(").replace("）", ")"), {"__builtins__": {}, "math": math}))},
    "unit_converter": lambda args: {"result": round(float(args.get("value", 0)) * UNIT_DATA.get(f"{args.get('from_unit', '').lower()}_{args.get('to_unit', '').lower()}", 1), 4)},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], "humidity": "65%", "condition": w[1]})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    "get_current_time": lambda args: {"datetime": TIME_DATA.get(args.get("timezone", "Asia/Shanghai"), "2025-03-07 14:30:00"), "timezone": args.get("timezone", "Asia/Shanghai")},
    "get_exchange_rate": lambda args: {"from": args.get("from_currency"), "to": args.get("to_currency"), "rate": EXCHANGE_DATA.get((args.get("from_currency"), args.get("to_currency")), 1.0)},
    "translate_text": lambda args: {"translated_text": TRANSLATE_DATA.get((args.get("text"), args.get("target_language")), args.get("text", ""))},
}

# ======== 参数校验 ========
# 校验工具调用的参数是否完整有效
CHECK_ARGS = {
    "calculate_math": lambda a: bool(a.get("expression")),
    "unit_converter": lambda a: a.get("value") is not None and a.get("from_unit") and a.get("to_unit"),
    "get_current_weather": lambda a: bool(a.get("location")),
    "get_current_time": lambda a: True,  # timezone 有默认值，参数可选
    "get_exchange_rate": lambda a: bool(a.get("from_currency")) and bool(a.get("to_currency")),
    "translate_text": lambda a: bool(a.get("text")) and bool(a.get("target_language")),
}

# ======== 工具调用解析与执行 ========
def parse_tool_calls(text):
    """
    从模型生成的文本中解析 <tool_call>...</tool_call> 标签内的 JSON。
    
    示例输入：
        '我来帮你查一下天气<tool_call>{"name": "get_current_weather", "arguments": {"location": "北京"}}</tool_call>'
    
    返回：
        [{"name": "get_current_weather", "arguments": {"location": "北京"}}]
    """
    calls = []
    for m in re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        try: calls.append(json.loads(m.strip()))
        except: pass
    return calls

def execute_tool(name, args):
    """
    执行指定工具并返回结果（使用模拟数据）。
    设置 1 秒超时，防止 eval 等操作挂死（如死循环表达式）。
    
    Args:
        name: 工具名称
        args: 工具参数字典
    
    Returns:
        dict: 工具执行结果，失败返回 None
    """
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        # 设置 1 秒超时保护（仅 Linux 有效）
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
        signal.alarm(1)
        return fn(args)
    except:
        return None
    finally:
        try: signal.alarm(0)
        except: pass

# ======== 多轮 Rollout ========
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """
    对单个样本执行多轮 Agent Rollout。
    
    这是 Agent RL 训练的核心：模型可能需要多轮"生成 → 工具调用 → 观察结果 → 继续生成"
    的交互循环才能完成任务。
    
    多轮交互流程：
    ┌─────────────────────────────────────────────────────────────┐
    │ Turn 1: 模型生成回复（可能包含 <tool_call>）                  │
    │   ↓ 如果包含工具调用                                         │
    │ 环境执行工具，返回结果 → 追加到对话历史                       │
    │   ↓                                                          │
    │ Turn 2: 模型看到工具结果后继续生成                            │
    │   ↓ 如果还需要调用工具                                       │
    │ ...重复直到无工具调用或达到 max_turns                         │
    └─────────────────────────────────────────────────────────────┘
    
    关键设计 —— 混合 Mask：
    - 模型生成的 token → mask=1（参与梯度计算）
    - 环境注入的 token（工具结果、模板标记）→ mask=0（不参与梯度）
    
    Args:
        rollout_engine: 推理引擎
        tokenizer: 分词器
        messages: 初始对话消息列表
        tools: 可用工具列表
        max_turns: 最大交互轮数
        max_new_tokens: 每轮最大生成长度
        thinking_ratio: 开启思考模式的概率
        device: 计算设备
    
    Returns:
        final_output: 最后一轮生成的文本
        final_context: 完整的对话上下文文本
        prompt_ids: 初始 prompt 的 token ids
        response_ids: 所有 response 部分的 token ids（含模型生成 + 环境注入）
        response_mask: 对应的 mask（1=模型生成, 0=环境注入）
        response_old_logps: 旧策略的 log prob（环境注入部分为 0）
        all_outputs: 各轮模型生成的文本列表
        unfinished: 是否因达到 max_turns 而未完成
    """
    all_outputs = []
    prompt_ids = None
    response_ids = []
    response_mask = []
    response_old_logps = []
    final_context = ""
    unfinished = False
    # 随机决定是否开启思考模式（<think>...</think>）
    open_thinking = random.random() < thinking_ratio

    for turn in range(max_turns):
        # 将当前对话历史渲染为模型输入格式
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        inputs = tokenizer(context, return_tensors="pt", add_special_tokens=False).to(device)
        context_ids = inputs["input_ids"][0].tolist()

        # 第一轮时记录原始 prompt ids
        if prompt_ids is None:
            prompt_ids = context_ids

        # 调用 rollout 引擎生成回复
        rollout_result = rollout_engine.rollout(
            prompt_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            num_generations=1,
            max_new_tokens=max_new_tokens,
            temperature=0.8,
        )
        new_ids = rollout_result.completion_ids[0].tolist()
        new_logps = rollout_result.per_token_logps[0].tolist()

        if len(new_ids) != len(new_logps): Logger(f"rollout token/logprob length mismatch: {len(new_ids)} vs {len(new_logps)}")

        # 过滤掉 pad 和 eos token（只保留有效生成内容）
        pairs = [(t, lp) for t, lp in zip(new_ids, new_logps) if t != tokenizer.pad_token_id and t != tokenizer.eos_token_id]
        new_ids = [t for t, _ in pairs]
        new_logps = [lp for _, lp in pairs]
        new_text = rollout_result.completions[0]

        all_outputs.append(new_text)
        # 模型生成的部分：mask=1，参与梯度计算
        response_ids.extend(new_ids)
        response_mask.extend([1] * len(new_ids))
        response_old_logps.extend(new_logps)
        final_context = context + new_text

        # 解析模型是否发起了工具调用
        calls = parse_tool_calls(new_text)
        if not calls:
            # 没有工具调用 → 任务完成，退出循环
            break

        # 如果已是最后一轮但还有工具调用 → 标记为未完成
        unfinished = turn == max_turns - 1

        # 将模型回复和工具执行结果追加到对话历史
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            name, raw = call.get("name", ""), call.get("arguments", {})
            if isinstance(raw, str):
                try: raw = json.loads(raw)
                except: raw = {}
            result = execute_tool(name, raw)
            # 截断过长的工具结果，防止撑爆 tokenizer
            result_str = (json.dumps(result, ensure_ascii=False) if result else '{"error": "tool not found"}')[:2048]
            messages.append({"role": "tool", "content": result_str})

        # 将工具结果渲染后的新增 token 作为"环境观察"，mask=0（不参与梯度）
        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        observe_ids = tokenizer(observe_context, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        # 计算新增的 observation token（当前总长度之后的部分）
        current_len = len(prompt_ids) + len(response_ids)
        obs_delta = observe_ids[current_len:]
        # 环境注入的部分：mask=0，logp=0，不参与策略梯度
        response_ids.extend(obs_delta)
        response_mask.extend([0] * len(obs_delta))
        response_old_logps.extend([0.0] * len(obs_delta))
        final_context = observe_context

    final_output = all_outputs[-1] if all_outputs else ""
    prompt_ids = prompt_ids or []
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, list(all_outputs), unfinished

def rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, num_gen, max_turns=3, max_new_tokens=256, thinking_ratio=0.5, device="cuda"):
    """
    对一个 batch 的样本执行多轮 Agent Rollout。
    
    对每个样本生成 num_gen 个独立的多轮交互轨迹。
    注意：这里是逐条串行执行的（因为多轮交互是顺序依赖的），
    不同于 GRPO 中可以一次性并行生成所有候选。
    
    Args:
        messages_batch: batch 中每个样本的初始消息列表
        tools_batch: batch 中每个样本可用的工具列表
        num_gen: 每个样本生成的候选轨迹数（用于 GRPO 组内比较）
        其他参数同 rollout_single
    
    Returns:
        各字段的 batch 列表，长度均为 B * num_gen
    """
    all_completions = []
    all_contexts = []
    all_prompt_ids = []
    all_response_ids = []
    all_response_masks = []
    all_response_old_logps = []
    all_turn_outputs = []
    all_unfinished = []
    for messages, tools in zip(messages_batch, tools_batch):
        for _ in range(num_gen):
            # 深拷贝消息列表，避免多次生成之间互相污染
            msgs_copy = [dict(m) for m in messages]
            completion, context, prompt_ids, response_ids, response_mask, response_old_logps, turn_outputs, unfinished = rollout_single(rollout_engine, tokenizer, msgs_copy, tools, max_turns, max_new_tokens, thinking_ratio, device)
            all_completions.append(completion)
            all_contexts.append(context)
            all_prompt_ids.append(prompt_ids)
            all_response_ids.append(response_ids)
            all_response_masks.append(response_mask)
            all_response_old_logps.append(response_old_logps)
            all_turn_outputs.append(turn_outputs)
            all_unfinished.append(unfinished)
    return all_completions, all_contexts, all_prompt_ids, all_response_ids, all_response_masks, all_response_old_logps, all_turn_outputs, all_unfinished

# ======== Reward 计算 ========
def validate_gt_in_text(text, gt_list):
    """
    验证文本中是否包含 ground truth 答案。
    
    支持两种匹配方式：
    1. 字符串包含匹配（不区分大小写）
    2. 数值近似匹配（容差 1e-6，处理逗号分隔的数字格式）
    
    Args:
        text: 模型最终输出的文本
        gt_list: ground truth 答案列表（可以是字符串或数字）
    
    Returns:
        set: 在文本中找到的 GT 答案集合
    """
    text, text_num = str(text), str(text).replace(',', '')
    nums = [float(x) for x in re.findall(r'(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w.])', text_num)]
    return {g for g in gt_list if ((s := str(g).strip()) and s.lower() in text.lower()) or (re.fullmatch(r'[-+]?\d+(?:\.\d+)?', str(g).strip().replace(',', '')) and any(abs(float(str(g).strip().replace(',', '')) - n) < 1e-6 for n in nums))}

def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, reward_model=None, device="cuda", turn_outputs_batch=None, unfinished_batch=None):
    """
    计算 Agent RL 的综合奖励。
    
    奖励分为两大类：
    
    一、无工具调用的情况（模型直接回答）：
       - 长度奖励：[5, 800] 范围得 +0.5
       - 思考格式奖励：thinking 合理得 +1.0
       - Reward Model 评分
       - 重复惩罚
    
    二、有工具调用的情况（Agent 模式）：
       - 标签格式分：<tool_call> 和 </tool_call> 必须配对
       - 工具对齐分：调用的工具数量是否与 GT 期望一致
       - 参数校验分：工具参数是否完整有效
       - GT 答案分：最终回答中是否包含正确答案（最高 +2.5）
       - 未完成扣分：达到 max_turns 但任务未完成 -0.5
       - 重复惩罚
    
    所有奖励最终 clip 到 [-3.0, 3.0] 范围。
    
    Args:
        prompts: 原始 prompt 列表 [B]
        completions: 最终回复文本列表 [B*num_gen]
        gt_batch: ground truth 答案列表 [B]
        tools_batch: 每个样本的工具列表 [B]
        num_gen: 每个样本的生成数
        reward_model: 奖励模型（可选）
        turn_outputs_batch: 各轮输出列表 [B*num_gen]
        unfinished_batch: 是否未完成列表 [B*num_gen]
    
    Returns:
        rewards: [B*num_gen] 奖励 tensor
    """
    rewards = torch.zeros(len(completions), device=device)
    for idx, response in enumerate(completions):
        reward, answer = 0.0, response
        sample_idx = idx // num_gen
        tools = tools_batch[sample_idx]
        turn_outputs = turn_outputs_batch[idx] if turn_outputs_batch is not None else [response]
        unfinished = unfinished_batch[idx] if unfinished_batch is not None else False

        # 提取每轮的实际回答内容（去掉 <think> 部分）
        turn_answers = [turn.split('</think>', 1)[-1].strip() if '</think>' in turn else turn.strip() for turn in turn_outputs]
        answer = turn_answers[-1] if turn_answers else response.strip()

        # 获取有效工具名集合
        valid_names = {t['function']['name'] for t in tools} if tools else set()

        # 解析所有轮次中的工具调用
        tool_calls = []
        for turn_answer in turn_answers: tool_calls.extend(parse_tool_calls(turn_answer))

        # 通用扣分：<tool_call> 和 </tool_call> 标签不配对时扣分
        reward -= 0.5 * sum(abs(turn.count('<tool_call>') - turn.count('</tool_call>')) for turn in turn_answers)

        # -------- 无工具调用：格式+reward 奖励 --------
        if not tool_calls:
            reward += 0.5 if 5 <= len(response.strip()) <= 800 else -0.5  # 长度分
            if '</think>' in response:
                think, answer = response.split('</think>', 1)
                reward += 1.0 if 20 <= len(think.strip()) <= 300 else -0.5  # 思考长度分
                reward += 0.25 if response.count('</think>') == 1 else -0.25  # 思考闭合分
                answer = answer.strip()
            # Reward Model 评分
            if reward_model is not None:
                prompt = prompts[sample_idx]
                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                score = reward_model.get_score(messages, answer)
                reward += score
            reward -= rep_penalty(answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)

        # -------- 有工具调用：执行结果奖励 --------
        else:
            gt = gt_batch[sample_idx]
            # 统计有效的工具调用数量（工具名合法 + 参数校验通过）
            valid_call_count = 0
            for tool_call in tool_calls:
                name, raw = tool_call.get("name", ""), tool_call.get("arguments", {})
                if isinstance(raw, str):
                    try: raw = json.loads(raw)
                    except: raw = {}
                check = CHECK_ARGS.get(name)
                valid_call_count += int(bool(name in valid_names and check and check(raw)))

            # 工具对齐分：有效调用数与 GT 期望数的差距
            tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)
            reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap
            
            # 提取最终回答文本（工具调用结束后的部分）
            final_text = "" if unfinished else (answer.split('</tool_call>')[-1] if '</tool_call>' in answer else answer)
            # GT 验证：检查最终回答是否包含正确答案
            verified = validate_gt_in_text(final_text, gt) if gt else set()
            if gt: reward += 2.5 * len(verified) / len(gt)  # 按正确比例给分（满分 +2.5）
            if unfinished: reward -= 0.5  # 未完成扣分
            reward -= rep_penalty(final_text if final_text else answer)
            rewards[idx] = max(min(reward, 3.0), -3.0)

    return rewards

# ================================ 工具与 Reward = End ================================


def rl_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model=None, start_step=0, wandb=None, use_sglang=False):
    """
    执行一个 epoch 的 Agent RL 训练。
    
    与 train_grpo.py 的 grpo_train_epoch 的主要区别：
    1. Rollout 是多轮交互的（rollout_batch），而非单次生成
    2. 序列由 prompt + response 拼接而成，response 中包含混合 mask
       （模型生成=1，环境注入=0）
    3. 需要将变长的多轮序列 padding 对齐后再做批量前向
    
    算法核心与 GRPO 相同：
    - 组内标准化 advantage
    - CISPO / PPO-clip 策略损失
    - KL 散度惩罚
    """
    last_step = start_step
    for step, batch in enumerate(loader, start=start_step + 1):
        messages_batch = batch['messages']   # list[list[dict]]，每个样本的初始消息
        tools_batch = batch['tools']         # list[list[dict]]，每个样本可用的工具
        gt_batch = batch['gt']               # list[list]，每个样本的 ground truth 答案
        last_step = step

        # =================================================================
        # Step 1: 多轮 Rollout —— 对每个样本生成 num_generations 条交互轨迹
        # =================================================================
        with torch.no_grad():
            completions, contexts, prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch, turn_outputs_batch, unfinished_batch = rollout_batch(rollout_engine, tokenizer, messages_batch, tools_batch, args.num_generations, max_turns=3, max_new_tokens=args.max_gen_len, thinking_ratio=args.thinking_ratio, device=args.device)

        # =================================================================
        # Step 2: 将变长序列 padding 对齐，构建训练 tensor
        # =================================================================
        prompts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True, tools=t) for m, t in zip(messages_batch, tools_batch)]

        # 将 prompt_ids + response_ids 拼接，并构建对应的 mask 和 old_logps
        packed_samples = []
        for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
            ids = p + r                        # 完整序列 = prompt + response
            mask = [0] * len(p) + m            # prompt 部分 mask=0，response 按实际 mask
            old_logps = [0.0] * max(len(p) - 1, 0) + old_lp  # 对齐到 logits 维度（偏移1）
            # 截断过长的序列
            if len(ids) > args.max_total_len:
                ids = ids[-args.max_total_len:]
                mask = mask[-args.max_total_len:]
                old_logps = old_logps[-(len(ids) - 1):]
            # 找到第一个 mask=1 的位置作为 prompt_len
            prompt_len = next((i for i, v in enumerate(mask) if v == 1), len(mask))
            packed_samples.append((ids, mask, prompt_len, old_logps))

        # 构建 batch tensor（右侧 padding 对齐到最长序列）
        seq_lens = torch.tensor([len(ids) for ids, _, _, _ in packed_samples], device=args.device)
        max_len = seq_lens.max().item()
        input_ids = torch.tensor([ids + [tokenizer.pad_token_id] * (max_len - len(ids)) for ids, _, _, _ in packed_samples], device=args.device)
        prompt_lens = torch.tensor([prompt_len for _, _, prompt_len, _ in packed_samples], device=args.device)
        # full_response_masks: 标记哪些 token 是模型生成的（参与梯度）
        full_response_masks = torch.tensor([mask + [0] * (max_len - len(mask)) for _, mask, _, _ in packed_samples], device=args.device, dtype=torch.float32)
        # old_per_token_logps: rollout 时记录的旧策略 log prob
        old_per_token_logps = torch.tensor([old_logps + [0.0] * ((max_len - 1) - len(old_logps)) for _, _, _, old_logps in packed_samples], device=args.device, dtype=torch.float32)
        full_mask = (input_ids != tokenizer.pad_token_id).long()

        # =================================================================
        # Step 3: 计算 Reward
        # =================================================================
        rewards = calculate_rewards(prompts, completions, gt_batch, tools_batch, args.num_generations, reward_model, device=args.device, turn_outputs_batch=turn_outputs_batch, unfinished_batch=unfinished_batch)

        # =================================================================
        # Step 4: 前向推理 —— 计算当前策略和参考模型的 log prob
        # =================================================================
        model_unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
        with autocast_ctx:
            res = model_unwrapped(input_ids, attention_mask=full_mask)
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
            logits = res.logits[:, :-1, :]  # [B*num_gen, seq_len-1, vocab]
            # 计算当前策略的 per-token log probability
            per_token_logps = F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            # 参考模型的 log prob（用于 KL 惩罚）
            ref_per_token_logps = compute_per_token_logps(ref_model, input_ids, input_ids.size(1) - 1, attention_mask=full_mask)

        # =================================================================
        # Step 5: 构建 completion mask（截止到 EOS）
        # =================================================================
        # completion_mask 基于 full_response_masks，进一步截断到 EOS 位置
        completion_mask = full_response_masks[:, 1:]  # 对齐 logits 的偏移
        is_eos = (input_ids[:, 1:] == tokenizer.eos_token_id) & completion_mask.bool()
        eos_idx = torch.full((completion_mask.size(0),), completion_mask.size(1) - 1, device=args.device, dtype=torch.long)
        has_eos = is_eos.any(dim=1)
        eos_idx[has_eos] = is_eos.int().argmax(dim=1)[has_eos]
        pos = torch.arange(completion_mask.size(1), device=args.device).unsqueeze(0)
        completion_mask = completion_mask * (pos <= eos_idx.unsqueeze(1)).float()
        token_counts = completion_mask.sum(dim=1)
        valid_rows = token_counts > 0  # 排除完全无有效 token 的行

        # [可选] Debug 模式打印
        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(messages_batch)):
                Logger(f"[DEBUG] step={step}, gt[{i}]: {repr(gt_batch[i])}")
                Logger('-'*100)
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    plen, slen = prompt_lens[idx].item(), seq_lens[idx].item()
                    Logger(f"{'=' * 30} [DEBUG] gen[{i}][{j}] CONTEXT_BEGIN {'=' * 30}")
                    Logger(contexts[idx])
                    Logger(f"{'=' * 31} [DEBUG] gen[{i}][{j}] CONTEXT_END {'=' * 31}")
                    Logger(f"[DEBUG] gen[{i}][{j}] prompt_len={plen}, seq_len={slen}")
                    tokens = input_ids[idx, plen:slen].tolist()
                    text = tokenizer.decode(tokens, skip_special_tokens=False)
                    Logger(f"{'=' * 28} [DEBUG] gen[{i}][{j}] COMPLETION_BEGIN [{plen}:{slen}] {'=' * 28}")
                    Logger(text)
                    Logger(f"{'=' * 29} [DEBUG] gen[{i}][{j}] COMPLETION_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{i}][{j}] reward={rewards[idx].item():.4f}")
                    Logger('='*100)

        # =================================================================
        # Step 6: GRPO 组内标准化 Advantage
        # =================================================================
        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
        std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
        advantages = (rewards - mean_r) / (std_r + 1e-4)

        # =================================================================
        # Step 7: 计算策略损失（CISPO / GRPO）
        # =================================================================
        # KL 惩罚（k3 estimator）
        kl_div = ref_per_token_logps - per_token_logps
        per_token_kl = torch.exp(kl_div) - kl_div - 1

        # 重要性采样比率
        ratio = torch.exp(per_token_logps - old_per_token_logps)

        if args.loss_type == "cispo":
            # CISPO: clamp ratio 上界 + detach，梯度仅通过 logps
            clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
            per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
        else:
            # 标准 GRPO (PPO-clip)
            clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
            per_token_loss1 = ratio * advantages.unsqueeze(1)
            per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
            per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)

        # 只在有效行上计算损失（避免空序列产生 nan）
        policy_loss = (((per_token_loss * completion_mask).sum(dim=1)[valid_rows] / token_counts[valid_rows].clamp(min=1)).mean()
                       if valid_rows.any() else per_token_loss.sum() * 0.0)
        loss = (policy_loss + aux_loss) / args.accumulation_steps
        loss.backward()

        # =================================================================
        # Step 8: 参数更新
        # =================================================================
        if step % args.accumulation_steps == 0:
            if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        # =================================================================
        # Step 9: 日志记录
        # =================================================================
        if step % args.log_interval == 0 or step == iters:
            pl = loss.item() * args.accumulation_steps
            ar = rewards.mean().item()
            al = token_counts.float().mean().item()
            kl = ((ref_per_token_logps - per_token_logps) * completion_mask).sum().item() / max(token_counts.sum().item(), 1)
            gs = grouped_rewards.std(dim=1, unbiased=False).mean().item()
            am, ast = advantages.mean().item(), advantages.std().item()
            lr = optimizer.param_groups[0]['lr']
            Logger(f'Epoch:[{epoch+1}/{args.epochs}]({step}/{iters}), Reward:{ar:.4f}, KL:{kl:.4f}, GrpStd:{gs:.4f}, AdvStd:{ast:.4f}, Loss:{pl:.4f}, AvgLen:{al:.2f}, AdvMean:{am:.4f}, LR:{lr:.8f}')
            if wandb and is_main_process():
                wandb.log({"reward":ar,"kl_ref":kl,"group_reward_std":gs,"advantages_std":ast,"policy_loss":pl,"avg_response_len":al,"advantages_mean":am,"learning_rate":lr})

        # =================================================================
        # Step 10: 模型保存
        # =================================================================
        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer,
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # 同步策略权重到 rollout 引擎
        if step % args.save_interval == 0 or step == iters: rollout_engine.update_policy(model)

        # 清理显存
        del per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, mean_r, std_r, advantages, completion_mask

    # epoch 结束，消耗剩余累积梯度
    if last_step > start_step and last_step % args.accumulation_steps != 0:
        if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()


# =============================================================================
# 主程序入口
# =============================================================================
if __name__ == "__main__":
    # ========== 命令行参数 ==========
    parser = argparse.ArgumentParser(description="MiniMind Agent RL")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='agent', type=str, help="保存权重名称")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="批次大小")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型 bfloat16/float16")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="模型层数")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="最大序列长度")
    parser.add_argument("--max_gen_len", type=int, default=768, help="单次最大生成长度")
    parser.add_argument("--max_total_len", type=int, default=2500, help="训练侧最终总长度上界")
    parser.add_argument("--data_path", type=str, default="../dataset/agent_rl.jsonl", help="训练数据路径")
    parser.add_argument("--num_generations", type=int, default=4, help="每个prompt生成数量")
    parser.add_argument("--beta", type=float, default=0.1, help="KL散度惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo"], help="loss类型")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="epsilon上界")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="加载预训练权重名称")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否从checkpoint恢复")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb记录")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Agent-RL", help="wandb项目名称")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile")
    parser.add_argument("--debug_mode", action="store_true", help="调试模式")
    parser.add_argument("--debug_interval", type=int, default=20, help="调试日志间隔")
    parser.add_argument("--thinking_ratio", type=float, default=0.1, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_agent", help="SGLang共享存储路径")
    args = parser.parse_args()

    # ========== 1. 初始化分布式环境 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))

    # ========== 2. 模型配置 ==========
    os.makedirs(args.save_dir, exist_ok=True)
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume == 1 else None

    # ========== 3. 混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)

    # ========== 4. 实验跟踪 ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb.init(project=args.wandb_project, name=f"Agent-RL-E{args.epochs}-B{args.batch_size}-LR{args.learning_rate}", id=wandb_id, resume=resume)

    # ========== 5. 初始化模型 ==========
    # 策略模型（待训练）
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)

    # 参考模型（冻结，用于 KL 惩罚）
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)

    # 奖励模型（用于非工具调用场景的质量评分）
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    Logger(f'Loaded reward model from {args.reward_model_path}')

    # Rollout 引擎
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

    # ========== 6. 数据和优化器 ==========
    train_ds = AgentRLDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    # 自定义 collate_fn：将 batch 中的 messages/tools/gt 分别聚合为列表
    def collate_fn(batch): return {'messages': [b['messages'] for b in batch], 'tools': [b['tools'] for b in batch], 'gt': [b['gt'] for b in batch]}
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collate_fn)
    iters = len(loader_for_count)
    total_optimizer_steps = math.ceil(iters / args.accumulation_steps) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)

    # ========== 7. 断点续训 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)

    # ========== 8. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)

    # ========== 9. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(42 + epoch); indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
        if skip > 0:
            Logger(f'Epoch [{epoch+1}/{args.epochs}]: skip {start_step} steps')
            rl_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            rl_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))

    # ========== 10. 清理 ==========
    if dist.is_initialized(): dist.destroy_process_group()
