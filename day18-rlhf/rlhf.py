#!/usr/bin/env python3
"""
Day 18: RLHF（基于人类反馈的强化学习）— 教模型"讨好"人类

本脚本演示：
1. Reward Model 训练：用人类偏好排序数据训练奖励模型
2. 偏好损失（Bradley-Terry）：让 RM 学会区分"好回答"和"差回答"
3. PPO 训练循环：SFT 模型生成 -> RM 打分 -> PPO 更新
4. KL 惩罚：防止模型偏离 SFT 太远（奖励作弊防御）
5. 对比实验：SFT 模型 vs RLHF 模型的不同表现
6. 奖励曲线：观察 RLHF 训练中奖励如何变化

运行方式：python3 rlhf.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import random

# ============================================================
# 0. 设备 & 随机种子
# ============================================================
device = "cpu"  # 保证兼容性
torch.manual_seed(42)
random.seed(42)

print("=" * 70)
print("Day 18: RLHF (Reinforcement Learning from Human Feedback)")
print("=" * 70)

# ============================================================
# 1. 模型定义（复用 Day 16/17 的 MiniGPT 架构）
# ============================================================
print("\n" + "=" * 70)
print("Part 1: Model Definition - MiniGPT + Reward Model")
print("=" * 70)


class Head(nn.Module):
    """单头自注意力"""

    def __init__(self, d_model, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        # 因果掩码：下三角矩阵，防止看到未来 token
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        # 缩放点积注意力
        scale = k.shape[-1] ** 0.5
        wei = q @ k.transpose(-2, -1) / scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """多头自注意力 = 多个 Head 拼接 + 投影"""

    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = d_model // n_heads
        self.heads = nn.ModuleList([
            Head(d_model, head_size, block_size, dropout)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """一个完整的 Transformer Block"""

    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, n_heads, block_size, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-Norm 残差连接
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """玩具级 GPT 模型"""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, loss_mask=None):
        """前向传播，返回 logits 和可选的 loss"""
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            if loss_mask is not None:
                mask_flat = loss_mask.view(B * T)
                targets_masked = targets_flat.clone()
                targets_masked[mask_flat == 0] = -100
                loss = F.cross_entropy(logits_flat, targets_masked)
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)
            return logits, loss
        return logits

    def get_logits(self, idx):
        """只获取 logits（不计算 loss）"""
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """自回归生成"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


class RewardModel(nn.Module):
    """
    奖励模型：输入 "指令+回答" 的文本，输出一个标量分数。
    架构：复用 Transformer encoder + Value Head (线性层映射到标量)。
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        # Value Head: 把 d_model 维向量映射到 1 维标量
        self.value_head = nn.Linear(d_model, 1)

    def forward(self, idx):
        """
        输入 idx: (B, T) token ID 序列
        输出 reward: (B,) 标量奖励分数
        """
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)  # (B, T, d_model)
        # 取最后一个 token 的隐藏状态作为整体表示
        last_hidden = x[:, -1, :]  # (B, d_model)
        reward = self.value_head(last_hidden).squeeze(-1)  # (B,)
        return reward


# 模型配置
VOCAB_SIZE = 200
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
BLOCK_SIZE = 128
DROPOUT = 0.1

print(f"Config: vocab_size={VOCAB_SIZE}, d_model={D_MODEL}, "
      f"n_heads={N_HEADS}, n_layers={N_LAYERS}, block_size={BLOCK_SIZE}")

# ============================================================
# 2. Tokenizer
# ============================================================
print("\n" + "=" * 70)
print("Part 2: Tokenizer")
print("=" * 70)

SPECIAL_TOKENS = {
    "<|im_start|>": 0,
    "<|im_end|>": 1,
    "<|pad|>": 2,
}

# 构建字符映射
all_chars = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 .,!?;:'\"\n-+"
    "=()"
    "\u00d7"  # x
    "\uff08\uff09\uff1a\uff0c\u3002\uff1f\u3001\uff01"  # CN punctuation
    "\u662f\u7684\u6211\u8981\u628a\u4e0b\u9762\u53e5\u5b50\u7ffb\u8bd1\u6210\u82f1\u6587"
    "\u5149\u5408\u4f5c\u7528\u690d\u7269\u5229\u7528\u9633\u5149\u548c\u4e2d\u6c34\u5236\u9020\u517b\u5206\u8fc7\u7a0b"
    "\u5177\u4f53\u6765\u8bf4\u53f6\u7eff\u7d20\u5438\u6536\u592a\u9633\u80fd\u91cf\u5c06\u4e8c\u6c27\u5316\u78b3\u8f6c\u5316\u8461\u8404\u7cd6\u6c27\u6c14\u91ca\u653e\u51fa\u6765\u4eca\u5929\u6c14\u5f88"
    "\u5317\u4eac\u6674\u6700\u9ad8\u6e29\u5ea6\u5ea6\u6570\u56de\u7b54\u95ee\u9898\u4ec0\u4e48\u4e00\u7bc7\u5173\u4e8e\u6625\u77ed\u8bd7\u6b4c\u98ce\u62c2\u9762\u67f3\u4e1d\u957f\u6843\u674f\u82b1\u5f00\u6ee1\u9662\u9999"
    "\u71d5\u5b50\u5f52\u6765\u5bfb\u65e7\u5de2\u4e00\u6c60\u78a7\u6c34\u6620\u659c\u9633\u9ed1\u6d1e\u5b87\u5b99\u5f15\u529b\u6781\u5f3a\u5929\u4f53\u8fde\u5149\u90fd\u9003\u4e0d\u51fa"
    "\u662f\u5927\u8d28\u91cf\u6052\u661f\u6b7b\u4ea1\u540e\u4ea7\u7269\u5bc6\u5ea6\u65e0\u9650\u5927\u5947\u70b9\u4f1a\u541e\u566c\u5207\u7b80\u7565\u89e3\u91ca"
    "\u6b63\u786e\u4e0d\u5b8c\u6574\u4f60\u597d\u4e16\u8fd9\u4e2a\u56de\u592a\u68d2\u4e86\u8fd8\u53ef\u4ee5\u5f88\u8be6\u7ec6\u4f46\u4e0d\u591f\u51c6\u786e"
    "\u8bf7\u4ecb\u7ecd\u7b80\u6d01\u660e\u91cf\u5b50\u7ea0\u7f20\u795e\u79d8\u8054\u7cfb\u6d4b\u91cf\u77ac\u95f4\u786e\u5b9a\u53e6\u4e2a\u72b6\u6001\u8ddd\u79bb\u591a\u8fdc\u4fe1\u606f\u4f20\u9012\u6001"
    "\u00d7"
)
# Actually let me just use a simpler approach - build the char set directly
char_set = set()
for ch in (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 .,!?;:'\"\n-+()=\u00d7"
    "\u662f\u7684\u6211\u8981\u628a\u4e0b\u9762\u53e5\u5b50\u7ffb\u8bd1\u6210\u82f1\u6587"
    "\u5149\u5408\u4f5c\u7528\u690d\u7269\u5229\u7528\u9633\u5149\u548c\u4e2d\u6c34\u5236\u9020\u517b\u5206\u8fc7\u7a0b"
    "\u5177\u4f53\u6765\u8bf4\u53f6\u7eff\u7d20\u5438\u6536\u592a\u9633\u80fd\u91cf\u5c06\u4e8c\u6c27\u5316\u78b3\u8f6c\u5316\u8461\u8404\u7cd6\u6c27\u6c14\u91ca\u653e\u51fa\u6765\u4eca\u5929\u6c14\u5f88"
    "\u5317\u4eac\u6674\u6700\u9ad8\u6e29\u5ea6\u5ea6\u6570\u56de\u7b54\u95ee\u9898\u4ec0\u4e48\u4e00\u7bc7\u5173\u4e8e\u6625\u77ed\u8bd7\u6b4c\u98ce\u62c2\u9762\u67f3\u4e1d\u957f\u6843\u674f\u82b1\u5f00\u6ee1\u9662\u9999"
    "\u71d5\u5b50\u5f52\u6765\u5bfb\u65e7\u5de2\u4e00\u6c60\u78a7\u6c34\u6620\u659c\u9633\u9ed1\u6d1e\u5b87\u5b99\u5f15\u529b\u6781\u5f3a\u5929\u4f53\u8fde\u5149\u90fd\u9003\u4e0d\u51fa"
    "\u662f\u5927\u8d28\u91cf\u6052\u661f\u6b7b\u4ea1\u540e\u4ea7\u7269\u5bc6\u5ea6\u65e0\u9650\u5927\u5947\u70b9\u4f1a\u541e\u566c\u5207\u7b80\u7565\u89e3\u91ca"
    "\u6b63\u786e\u4e0d\u5b8c\u6574\u4f60\u597d\u4e16\u8fd9\u4e2a\u56de\u592a\u68d2\u4e86\u8fd8\u53ef\u4ee5\u5f88\u8be6\u7ec6\u4f46\u4e0d\u591f\u51c6\u786e"
    "\u8bf7\u4ecb\u7ecd\u7b80\u6d01\u660e\u91cf\u5b50\u7ea0\u7f20\u795e\u79d8\u8054\u7cfb\u6d4b\u91cf\u77ac\u95f4\u786e\u5b9a\u53e6\u4e2a\u72b6\u6001\u8ddd\u79bb\u591a\u8fdc\u4fe1\u606f\u4f20\u9012\u6001"
    "\uff08\uff09\uff1a\uff0c\u3002\uff1f\u3001\uff01"
):
    char_set.add(ch)

char_list = sorted(char_set)
char_to_id = {}
id_to_char = {}
for i, ch in enumerate(char_list):
    token_id = len(SPECIAL_TOKENS) + i
    char_to_id[ch] = token_id
    id_to_char[token_id] = ch

VOCAB_SIZE = max(id_to_char.keys()) + 10
UNK_ID = VOCAB_SIZE - 1


def encode(text):
    """文本 -> token ID 列表"""
    ids = []
    i = 0
    while i < len(text):
        # 检查特殊 token
        matched = False
        for special in SPECIAL_TOKENS:
            if text[i:i+len(special)] == special:
                ids.append(SPECIAL_TOKENS[special])
                i += len(special)
                matched = True
                break
        if matched:
            continue
        # 普通字符
        ch = text[i]
        if ch in char_to_id:
            ids.append(char_to_id[ch])
        else:
            ids.append(UNK_ID)
        i += 1
    return ids


def decode(ids):
    """token ID 列表 -> 文本"""
    chars = []
    for tid in ids:
        if tid in id_to_char:
            chars.append(id_to_char[tid])
        elif tid == SPECIAL_TOKENS["<|im_start|>"]:
            chars.append("<|im_start|>")
        elif tid == SPECIAL_TOKENS["<|im_end|>"]:
            chars.append("<|im_end|>")
        elif tid == SPECIAL_TOKENS["<|pad|>"]:
            chars.append("")
        else:
            chars.append("?")
    return "".join(chars)


print(f"Vocab size: {VOCAB_SIZE}")
print(f"Char coverage: {len(char_to_id)} chars")

# ============================================================
# 3. SFT Data + Human Preference Data
# ============================================================
print("\n" + "=" * 70)
print("Part 3: SFT Data & Human Preference Data")
print("=" * 70)

# SFT instruction-response pairs
sft_data = [
    {"instruction": "\u4ec0\u4e48\u662f\u5149\u5408\u4f5c\u7528\uff1f",
     "response": "\u5149\u5408\u4f5c\u7528\u662f\u690d\u7269\u5229\u7528\u9633\u5149\u548c\u4e2d\u6c34\u5236\u9020\u517b\u5206\u7684\u8fc7\u7a0b\u3002"},
    {"instruction": "\u56de\u7b54\u95ee\u9898\uff1a1+1=?",
     "response": "1+1=2"},
    {"instruction": "\u4eca\u5929\u5317\u4eac\u7684\u5929\u6c14\u600e\u4e48\u6837\uff1f",
     "response": "\u5317\u4eac\u4eca\u5929\u6674\uff0c\u6700\u9ad8\u6e29\u5ea628\u5ea6\u3002"},
    {"instruction": "\u8bf7\u4ecb\u7ecd\u9ed1\u6d1e\u3002",
     "response": "\u9ed1\u6d1e\u662f\u5b87\u5b99\u4e2d\u5f15\u529b\u6781\u5f3a\u7684\u5929\u4f53\uff0c\u8fde\u5149\u90fd\u9003\u4e0d\u51fa\u6765\u3002"},
    {"instruction": "\u5199\u4e00\u9996\u5173\u4e8e\u6625\u7684\u77ed\u8bd7\u6b4c\u3002",
     "response": "\u6625\u98ce\u62c2\u9762\u67f3\u4e1d\u957f\uff0c\u6843\u674f\u82b1\u5f00\u6ee1\u9662\u9999\u3002\u71d5\u5b50\u5f52\u6765\u5bfb\u65e7\u5de2\uff0c\u4e00\u6c60\u78a7\u6c34\u6620\u659c\u9633\u3002"},
]


def build_sft_sample(instruction, response, max_len=BLOCK_SIZE):
    """构建 SFT 训练样本（带损失掩码）"""
    prefix_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    prefix_ids = encode(prefix_text)
    response_ids = encode(response)
    suffix_ids = encode("<|im_end|>")

    full_ids = prefix_ids + response_ids + suffix_ids
    target_ids = full_ids[1:] + [SPECIAL_TOKENS["<|pad|>"]]

    # 只在回答部分计算损失
    response_start = len(prefix_ids)
    response_end = len(prefix_ids) + len(response_ids) + 1
    loss_mask = [0] * len(full_ids)
    for i in range(response_start, min(response_end, len(loss_mask))):
        loss_mask[i] = 1

    return full_ids[:max_len], target_ids[:max_len], loss_mask[:max_len]


# Build SFT dataset
all_sft_inputs = []
all_sft_targets = []
all_sft_masks = []
for item in sft_data:
    inp, tgt, mask = build_sft_sample(item["instruction"], item["response"])
    all_sft_inputs.append(inp)
    all_sft_targets.append(tgt)
    all_sft_masks.append(mask)

print(f"SFT data: {len(all_sft_inputs)} samples")

# ---- Human Preference Data ----
# Each entry: instruction, chosen (good), rejected (bad)
preference_data = [
    {
        "instruction": "\u8bf7\u4ecb\u7ecd\u9ed1\u6d1e\u3002",
        "chosen": "\u9ed1\u6d1e\u662f\u5b87\u5b99\u4e2d\u5f15\u529b\u6781\u5f3a\u7684\u5929\u4f53\uff0c\u8fde\u5149\u90fd\u9003\u4e0d\u51fa\u6765\u3002\u662f\u5927\u8d28\u91cf\u6052\u661f\u6b7b\u4ea1\u540e\u7684\u4ea7\u7269\u3002",
        "rejected": "\u9ed1\u6d1e\u662f\u5929\u7a7a\u4e2d\u9ed1\u8272\u7684\u4e1c\u897f\u3002",
    },
    {
        "instruction": "\u8bf7\u4ecb\u7ecd\u9ed1\u6d1e\u3002",
        "chosen": "\u9ed1\u6d1e\u662f\u5b87\u5b99\u4e2d\u5f15\u529b\u6781\u5f3a\u7684\u5929\u4f53\uff0c\u8fde\u5149\u90fd\u9003\u4e0d\u51fa\u6765\u3002",
        "rejected": "\u9ed1\u6d1e\u662f\u9ed1\u6d1e\u3002",
    },
    {
        "instruction": "\u4ec0\u4e48\u662f\u5149\u5408\u4f5c\u7528\uff1f",
        "chosen": "\u5149\u5408\u4f5c\u7528\u662f\u690d\u7269\u5229\u7528\u9633\u5149\u548c\u4e2d\u6c34\u5236\u9020\u517b\u5206\u7684\u8fc7\u7a0b\u3002\u5177\u4f53\u6765\u8bf4\uff0c\u53f6\u7eff\u7d20\u5438\u6536\u592a\u9633\u80fd\u91cf\uff0c\u5c06\u4e8c\u6c27\u5316\u78b3\u8f6c\u5316\u8461\u8404\u7cd6\u548c\u6c27\u6c14\u91ca\u653e\u51fa\u6765\u3002",
        "rejected": "\u5149\u5408\u4f5c\u7528\u662f\u690d\u7269\u3002",
    },
    {
        "instruction": "\u4ec0\u4e48\u662f\u5149\u5408\u4f5c\u7528\uff1f",
        "chosen": "\u5149\u5408\u4f5c\u7528\u662f\u690d\u7269\u5229\u7528\u9633\u5149\u548c\u4e2d\u6c34\u5236\u9020\u517b\u5206\u7684\u8fc7\u7a0b\u3002",
        "rejected": "\u5149\u5408\u4f5c\u7528\u662f\u5149\u5408\u4f5c\u7528\u3002",
    },
    {
        "instruction": "\u56de\u7b54\u95ee\u9898\uff1a1+1=?",
        "chosen": "1+1=2",
        "rejected": "1+1=3",
    },
    {
        "instruction": "\u56de\u7b94\u95ee\u9898\uff1a1+1=?",
        "chosen": "1+1=2",
        "rejected": "1+1=1+1",
    },
    {
        "instruction": "\u4eca\u5929\u5317\u4eac\u7684\u5929\u6c14\u600e\u4e48\u6837\uff1f",
        "chosen": "\u5317\u4eac\u4eca\u5929\u6674\uff0c\u6700\u9ad8\u6e29\u5ea628\u5ea6\u3002",
        "rejected": "\u4eca\u5929\u5929\u6c14\u5f88\u597d\u3002",
    },
    {
        "instruction": "\u5199\u4e00\u9996\u5173\u4e8e\u6625\u7684\u77ed\u8bd7\u6b4c\u3002",
        "chosen": "\u6625\u98ce\u62c2\u9762\u67f3\u4e1d\u957f\uff0c\u6843\u674f\u82b1\u5f00\u6ee1\u9662\u9999\u3002\u71d5\u5b50\u5f52\u6765\u5bfb\u65e7\u5de2\uff0c\u4e00\u6c60\u78a7\u6c34\u6620\u659c\u9633\u3002",
        "rejected": "\u6625\u98ce\u5439\uff0c\u82b1\u5f00\u4e86\u3002",
    },
]


def build_reward_input(instruction, response, max_len=BLOCK_SIZE):
    """构建 Reward Model 的输入（拼接指令和回答，padding 到固定长度）"""
    text = (f"<|im_start|>user\n{instruction}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>")
    ids = encode(text)
    if len(ids) < max_len:
        ids = ids + [SPECIAL_TOKENS["<|pad|>"]] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids


print(f"\nHuman preference data: {len(preference_data)} pairs")
print("Examples:")
for i in range(min(2, len(preference_data))):
    d = preference_data[i]
    print(f"  [{i}] Instruction: {d['instruction']}")
    print(f"      Chosen:   {d['chosen']}")
    print(f"      Rejected: {d['rejected']}")
    print()

# ============================================================
# 4. Phase 1: SFT Training (fast version)
# ============================================================
print("=" * 70)
print("Part 4: Phase 1 - SFT Training (fast)")
print("=" * 70)

sft_model = MiniGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE, DROPOUT).to(device)
total_params = sum(p.numel() for p in sft_model.parameters())
print(f"SFT model params: {total_params:,}")

# Quick pretraining
print("\nQuick pretraining...")
pretrain_text = (
    "\u5149\u5408\u4f5c\u7528\u662f\u690d\u7269\u5229\u7528\u9633\u5149\u548c\u4e2d\u6c34\u5236\u9020\u517b\u5206\u7684\u8fc7\u7a0b\u3002"
    "\u5177\u4f53\u6765\u8bf4\uff0c\u53f6\u7eff\u7d20\u5438\u6536\u592a\u9633\u80fd\u91cf\uff0c\u5c06\u4e8c\u6c27\u5316\u78b3\u8f6c\u5316\u8461\u8404\u7cd6\u548c\u6c27\u6c14\u91ca\u653e\u51fa\u6765\u3002"
    "\u9ed1\u6d1e\u662f\u5b87\u5b99\u4e2d\u5f15\u529b\u6781\u5f3a\u7684\u5929\u4f53\uff0c\u8fde\u5149\u90fd\u9003\u4e0d\u51fa\u6765\u3002"
    "\u662f\u5927\u8d28\u91cf\u6052\u661f\u6b7b\u4ea1\u540e\u7684\u4ea7\u7269\u3002"
    "\u4eca\u5929\u5929\u6c14\u5f88\u597d\u3002\u5317\u4eac\u4eca\u5929\u6674\uff0c\u6700\u9ad8\u6e29\u5ea628\u5ea6\u3002"
    "\u6625\u98ce\u62c2\u9762\u67f3\u4e1d\u957f\uff0c\u6843\u674f\u82b1\u5f00\u6ee1\u9662\u9999\u3002"
    "\u71d5\u5b50\u5f52\u6765\u5bfb\u65e7\u5de2\uff0c\u4e00\u6c60\u78a7\u6c34\u6620\u659c\u9633\u3002"
    "1+1=2\u3002\u6211\u8981\u5b66\u4e60\u3002"
)
pretrain_ids = encode(pretrain_text)
pt_inputs = []
pt_targets = []
block_size = 64
for i in range(0, len(pretrain_ids) - block_size - 1, 8):
    pt_inputs.append(pretrain_ids[i:i + block_size])
    pt_targets.append(pretrain_ids[i + 1:i + block_size + 1])

pt_inputs_t = torch.tensor(pt_inputs, dtype=torch.long).to(device)
pt_targets_t = torch.tensor(pt_targets, dtype=torch.long).to(device)

pt_optimizer = torch.optim.AdamW(sft_model.parameters(), lr=3e-4)
sft_model.train()
for step in range(200):
    idx = torch.randint(0, len(pt_inputs), (4,))
    x = pt_inputs_t[idx]
    y = pt_targets_t[idx]
    logits, loss = sft_model(x, y)
    pt_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0)
    pt_optimizer.step()
print(f"  Pretrain Loss: {loss.item():.4f}")

# SFT finetuning
print("SFT finetuning...")
sft_optimizer = torch.optim.AdamW(sft_model.parameters(), lr=5e-5)
sft_model.train()
for epoch in range(30):
    indices = list(range(len(all_sft_inputs)))
    random.shuffle(indices)
    for i in range(0, len(indices), 4):
        batch_idx = indices[i:i + 4]
        max_len_b = max(len(all_sft_inputs[j]) for j in batch_idx)
        batch_in = []
        batch_tgt = []
        batch_mask = []
        for j in batch_idx:
            inp = all_sft_inputs[j]
            tgt = all_sft_targets[j]
            msk = all_sft_masks[j]
            pad_len = max_len_b - len(inp)
            batch_in.append(inp + [SPECIAL_TOKENS["<|pad|>"]] * pad_len)
            batch_tgt.append(tgt + [-100] * pad_len)
            batch_mask.append(msk + [0] * pad_len)
        x = torch.tensor(batch_in, dtype=torch.long).to(device)
        y = torch.tensor(batch_tgt, dtype=torch.long).to(device)
        m = torch.tensor(batch_mask,