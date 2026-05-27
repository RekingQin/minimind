"""
训练工具函数集合
"""
import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM

def get_model_params(model, config):
    total = sum(p.numel() for p in model.parameters()) / 1e6
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))
    n_active = getattr(config, 'num_experts_per_tok', 0)
    n_shared = getattr(config, 'n_shared_experts', 0)
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6
    base = total - (expert * n_routed) - (shared_expert * n_shared)
    active = base + (expert * n_active) + (shared_expert * n_shared)
    if active < total: Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else: Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr*(0.1 + 0.45*(1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def lm_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, wandb=None, save_dir='../checkpoints', **kwargs):
    """
    模型检查点的保存与加载函数（双模式）。
    - model 不为 None 时：保存模型权重和完整训练状态（用于断点续训）
    - model 为 None 时：从磁盘加载训练状态以恢复训练

    Args:
        lm_config: 模型配置对象，包含 hidden_size、use_moe 等属性
        weight: 权重文件名前缀，如 'full_sft'、'pretrain' 等
        model: 模型实例（传入则保存，None 则加载）
        optimizer: 优化器实例
        epoch: 当前训练轮次
        step: 当前训练步数
        wandb: wandb 实例，用于记录 run id 以便续训时恢复
        save_dir: 检查点保存目录
        **kwargs: 其他需要保存的对象（如 scheduler、scaler 等）
    """
    os.makedirs(save_dir, exist_ok=True)
    # 根据是否使用 MoE 架构生成文件名后缀
    moe_path = '_moe' if lm_config.use_moe else ''
    # 纯模型权重路径（用于推理部署）
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    # 完整训练状态路径（包含 optimizer、epoch、step 等，用于断点续训）
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # ===== 保存模式 =====

        # 解包 DDP 包装，获取原始模型
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model
        # 解包 torch.compile 的包装（_orig_mod 是 compile 后模型的原始引用）
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        state_dict = raw_model.state_dict()
        # 将权重转为 FP16 并移到 CPU，减少磁盘占用
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 先写入临时文件再原子替换，防止保存中断导致文件损坏
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 获取 wandb run id，用于续训时恢复同一个 wandb run
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # 构建完整的断点续训数据
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,  # 记录保存时的 GPU 数量
            'wandb_id': wandb_id
        }

        # 处理额外传入的可保存对象（如 lr_scheduler、grad_scaler 等）
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    # 同样需要解包 DDP 和 torch.compile
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        # 同样使用原子写入保存 resume 文件
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)
        # 释放内存
        del state_dict, resume_data
        torch.cuda.empty_cache()
    else:
        # ===== 加载模式 =====
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            # 如果恢复训练时 GPU 数量发生变化，按比例换算 step
            # （因为每个 step 处理的数据量 = batch_size * world_size）
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None


def init_model(lm_config, from_weight='pretrain', tokenizer_path='../model', save_dir='../out', device='cuda'):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    if from_weight!= 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        weights = torch.load(weight_path, map_location=device)
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """
    奖励模型封装类，用于 RLHF 流程中对模型生成的回复进行质量评分。
    分数越高表示回复质量越好，可作为强化学习的奖励信号。
    """

    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        """
        Args:
            model_path: 预训练奖励模型的路径（需支持 get_score 方法）
            device: 推理设备，默认 "cuda"
            dtype: 模型精度，默认 FP16 以节省显存
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
        # 将模型移至指定设备并设为评估模式（关闭 dropout 等）
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()  # 禁用梯度计算，节省显存和加速推理
    def get_score(self, messages, response):
        """
        对给定对话历史下的一条回复进行打分。

        Args:
            messages: 对话历史列表，格式为 [{"role": "user/assistant", "content": "..."}]
            response: 待评分的模型回复文本

        Returns:
            float: 裁剪到 [-3.0, 3.0] 区间的奖励分数
        """
        # 手动拼接 prompt：因为大多数奖励模型的 get_score 接口只接受单轮对话（一个 user + 一个 assistant），
        # 但实际场景可能存在多轮对话，所以需要将多轮历史压缩拼接为单轮 user 消息。
        # 拼接效果示例：
        #   原始多轮: [user: "你好", assistant: "你好！", user: "解释量子力学"]
        #   拼接后 user: "user: 你好\nassistant: 你好！\n以上是对话历史。我的新问题是：\n解释量子力学"
        # 注意：此 prompt 模板可根据所使用的奖励模型调整，若模型本身支持多轮输入则可跳过拼接。
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])
        last_query = messages[-1]['content'] if messages else ""
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query

        # 构造奖励模型的输入格式：用户问题 + 待评分的助手回复
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]
        # 调用奖励模型的打分接口
        score = self.model.get_score(self.tokenizer, eval_messages)
        # 将分数裁剪到 [-3, 3] 区间，防止极端值影响训练稳定性
        return max(min(score, 3.0), -3.0)