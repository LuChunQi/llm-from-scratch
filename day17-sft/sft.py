#!/usr/bin/env python3
"""
Day 17: SFT（监督微调）— 让模型从"会说话"到"会听话"

本脚本演示：
1. SFT 数据构造：多种类型的指令-回答对
2. Chat Template 格式化 + 损失掩码
3. 从预训练模型加载 → SFT 微调
4. 对比实验：SFT 前后模型面对指令的不同表现
5. 灾难性遗忘演示：学习率过大时模型会"忘掉"知识

运行方式：python3 sft.py
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
device = "cpu"  # 保证兼容性，CPU 即可运行
torch.manual_seed(42)
random.seed(42)

print("=" * 70)
print("Day 17: SFT（监督微调）— 让模型从'会说话'到'会听话'")
print("=" * 70)

# ============================================================
# 1. 模型定义（和 Day 16 一样的 MiniGPT 架构）
# ============================================================
print("\n" + "=" * 70)
print("第一部分：MiniGPT 模型定义（复用 Day 16 架构）")
print("=" * 70)


class Head(nn.Module):
    """单头自注意力"""

    def __init__(self, d_model, head_size, block_size, dropout=0.1):
        super().__init__()
        # Q, K, V 投影矩阵
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        # 因果掩码（下三角矩阵），防止看到未来的 token
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        # 计算注意力分数（缩放点积）
        scale = k.shape[-1] ** 0.5
        wei = q @ k.transpose(-2, -1) / scale  # (B, T, T)
        # 因果掩码：把未来位置设为 -inf
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """多头自注意力 = 多个 Head 拼接 + 投影"""

    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = d_model // n_heads
        # 创建 n_heads 个注意力头
        self.heads = nn.ModuleList([
            Head(d_model, head_size, block_size, dropout)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 拼接所有头的输出
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """前馈网络（FFN）"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),  # 扩展 4 倍
            nn.GELU(),                          # 激活函数
            nn.Linear(4 * d_model, d_model),    # 压缩回原尺寸
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
        # 残差连接 + LayerNorm（Pre-Norm 变体）
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """玩具级 GPT 模型"""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        # Token Embedding：每个 token → d_model 维向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Position Embedding：每个位置 → d_model 维向量
        self.position_embedding = nn.Embedding(block_size, d_model)
        # N 层 Transformer Block
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)  # 最终 LayerNorm
        self.lm_head = nn.Linear(d_model, vocab_size)  # 输出投影到词表大小

    def forward(self, idx, targets=None, loss_mask=None):
        """
        前向传播
        idx: (B, T) 输入 token ID
        targets: (B, T) 目标 token ID
        loss_mask: (B, T) 损失掩码，1 = 计算损失，0 = 忽略
        """
        B, T = idx.shape

        # Token Embedding + Position Embedding
        tok_emb = self.token_embedding(idx)            # (B, T, d_model)
        pos_emb = self.position_embedding(
            torch.arange(T, device=idx.device)
        )  # (T, d_model)
        x = tok_emb + pos_emb                          # (B, T, d_model)

        # 通过 Transformer Blocks
        x = self.blocks(x)                             # (B, T, d_model)
        x = self.ln_f(x)                               # (B, T, d_model)

        # 输出 logits
        logits = self.lm_head(x)                       # (B, T, vocab_size)

        # 计算损失（如果提供了 targets）
        if targets is not None:
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)

            if loss_mask is not None:
                # ---- 核心：SFT 损失掩码机制 ----
                # 把 loss_mask 为 0 的位置的 target 设为 -100
                # PyTorch CrossEntropyLoss 的 ignore_index=−100 会跳过这些位置
                mask_flat = loss_mask.view(B * T)
                targets_masked = targets_flat.clone()
                targets_masked[mask_flat == 0] = -100  # 忽略非回答部分
                loss = F.cross_entropy(logits_flat, targets_masked)
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)
            return logits, loss

        return logits

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """自回归生成文本"""
        for _ in range(max_new_tokens):
            # 截断到 block_size 长度
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            # 取最后一个位置的 logits
            logits = logits[:, -1, :] / temperature
            # Top-K 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# 模型配置
VOCAB_SIZE = 200  # 简化词表（足够覆盖我们的示例）
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
BLOCK_SIZE = 128
DROPOUT = 0.1

print(f"模型配置: vocab_size={VOCAB_SIZE}, d_model={D_MODEL}, "
      f"n_heads={N_HEADS}, n_layers={N_LAYERS}, block_size={BLOCK_SIZE}")

# ============================================================
# 2. 简单 Tokenizer（字符 + 特殊 token）
# ============================================================
print("\n" + "=" * 70)
print("第二部分：Tokenizer + Chat Template")
print("=" * 70)

# 特殊 token
SPECIAL_TOKENS = {
    "<|im_start|>": 0,
    "<|im_end|>": 1,
    "<|pad|>": 2,
}
# 可打印 ASCII 字符从 ID=32（空格）开始
CHAR_OFFSET = 32  # 空格的 ASCII 码
# 构建 char ↔ id 映射
char_to_id = {}
id_to_char = {}
for i, ch in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"\n-+×=（）：，。？、！是的我要把下面句子翻译成英文光合作用是植物利用阳光和水中制造养分过程具体来说叶绿素吸收太阳光能量将二氧化碳水转化葡萄糖氧气释放出来今天气很北京晴最高温度度数回答问题什么一篇关于春短诗歌风拂面柳丝长桃杏花开满院香燕子归来寻旧巢一池碧水映斜阳"):
    token_id = len(SPECIAL_TOKENS) + i
    char_to_id[ch] = token_id
    id_to_char[token_id] = ch

# 更新词表大小为实际的 token 数量
VOCAB_SIZE = max(id_to_char.keys()) + 10  # 留一些余量

# 在特殊 token 中补充未知字符的映射
UNK_ID = VOCAB_SIZE - 1  # 未知字符的 ID


def encode(text):
    """文本 → token ID 列表"""
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
    """token ID 列表 → 文本"""
    chars = []
    for tid in ids:
        if tid in id_to_char:
            chars.append(id_to_char[tid])
        # 特殊 token 保留原文
        elif tid == SPECIAL_TOKENS["<|im_start|>"]:
            chars.append("<|im_start|>")
        elif tid == SPECIAL_TOKENS["<|im_end|>"]:
            chars.append("<|im_end|>")
        elif tid == SPECIAL_TOKENS["<|pad|>"]:
            chars.append("")
        # 未知 token 用 ? 表示
        else:
            chars.append("?")
    return "".join(chars)


# 演示编码/解码
demo_text = "什么是光合作用？"
demo_ids = encode(demo_text)
print(f"编码演示: '{demo_text}' → {demo_ids[:10]}...")
print(f"解码验证: {decode(demo_ids)}")

# ============================================================
# 3. SFT 数据构造
# ============================================================
print("\n" + "=" * 70)
print("第三部分：SFT 数据构造")
print("=" * 70)

# 定义多种类型的指令-回答对
sft_data = [
    # 知识问答
    {
        "instruction": "什么是光合作用？",
        "response": "光合作用是植物利用阳光和水中制造养分的过程。"
    },
    {
        "instruction": "什么是光合作用？",
        "response": "具体来说，叶绿素吸收太阳光能量，将二氧化碳和水转化葡萄糖和氧气释放出来。"
    },
    # 翻译
    {
        "instruction": "把下面的句子翻译成英文：今天天气很好",
        "response": "Today the weather is very nice."
    },
    {
        "instruction": "把下面的句子翻译成英文：我要学习",
        "response": "I want to study."
    },
    # 数学
    {
        "instruction": "回答问题：1+1=?",
        "response": "1+1=2"
    },
    {
        "instruction": "回答问题：8×7=?",
        "response": "8×7=56"
    },
    # 创意写作
    {
        "instruction": "写一首关于春的短诗歌。",
        "response": "春风拂面柳丝长，桃杏花开满院香。燕子归来寻旧巢，一池碧水映斜阳。"
    },
    # 对话
    {
        "instruction": "今天北京的天气怎么样？",
        "response": "北京今天晴，最高温度28度。"
    },
]


def build_sft_sample(instruction, response, max_len=BLOCK_SIZE):
    """
    构建 SFT 训练样本
    返回: (input_ids, target_ids, loss_mask)
    - input_ids: 模型输入的 token 序列
    - target_ids: 预测目标（左移一位）
    - loss_mask: 1 = 回答部分（计算损失），0 = 指令部分（忽略）
    """
    # 用 Chat Template 格式化
    text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    input_ids = encode(text)

    # 记录回答部分的起始和结束位置
    # 格式: <start>user\n{instruction}<end>\n<start>assistant\n{response}<end>
    # 我们需要找到 "assistant\n" 后面到 "<end>" 之间的位置

    # 找回答部分的起始位置
    assistant_prefix = "<|im_start|>assistant\n"
    response_suffix = "<|im_end|>"

    # 构建完整文本，标记回答区域
    prefix_text = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    prefix_ids = encode(prefix_text)
    response_ids = encode(response)
    suffix_ids = encode(response_suffix)

    full_ids = prefix_ids + response_ids + suffix_ids

    # 构建 target_ids（左移一位）
    target_ids = full_ids[1:] + [SPECIAL_TOKENS["<|pad|>"]]

    # 构建 loss_mask：只在回答部分 + 结尾 token 为 1
    # 回答部分 = prefix_ids 长度之后，到结尾
    response_start = len(prefix_ids)
    response_end = len(prefix_ids) + len(response_ids) + 1  # +1 包含结尾 token

    loss_mask = [0] * len(full_ids)
    for i in range(response_start, min(response_end, len(loss_mask))):
        loss_mask[i] = 1

    # 截断到 max_len
    input_ids = full_ids[:max_len]
    target_ids = target_ids[:max_len]
    loss_mask = loss_mask[:max_len]

    return input_ids, target_ids, loss_mask


# 构建所有训练样本
all_inputs = []
all_targets = []
all_masks = []

for item in sft_data:
    inp, tgt, mask = build_sft_sample(item["instruction"], item["response"])
    all_inputs.append(inp)
    all_targets.append(tgt)
    all_masks.append(mask)

# 展示一条数据的结构
print(f"\n📊 SFT 数据统计:")
print(f"  总样本数: {len(all_inputs)}")
print(f"  样本长度范围: {min(len(x) for x in all_inputs)} ~ {max(len(x) for x in all_inputs)} tokens")

# 详细展示第一条数据
sample_idx = 0
inp, tgt, mask = all_inputs[sample_idx], all_targets[sample_idx], all_masks[sample_idx]
print(f"\n📝 样本 {sample_idx} 详细结构:")
print(f"  原始数据: 指令='{sft_data[sample_idx]['instruction']}'")
print(f"           回答='{sft_data[sample_idx]['response']}'")
print(f"  编码后长度: {len(inp)} tokens")

# 标记回答部分
response_chars = []
for i in range(len(mask)):
    if mask[i] == 1:
        response_chars.append(decode([tgt[i]]))
print(f"  回答部分 token 数: {sum(mask)}")
print(f"  损失掩码分布: {''.join(['▓' if m else '░' for m in mask])}")
print(f"  (░ = 指令部分(忽略), ▓ = 回答部分(训练))")

# ============================================================
# 4. 数据加载器
# ============================================================
print("\n" + "=" * 70)
print("第四部分：构建 DataLoader & 开始 SFT 训练")
print("=" * 70)


def collate_fn(batch_indices):
    """将不同长度的样本 padding 到同一长度"""
    inputs = [all_inputs[i] for i in batch_indices]
    targets = [all_targets[i] for i in batch_indices]
    masks = [all_masks[i] for i in batch_indices]

    # 找到最大长度
    max_len = max(len(x) for x in inputs)

    # Padding
    padded_inputs = []
    padded_targets = []
    padded_masks = []
    for inp, tgt, msk in zip(inputs, targets, masks):
        pad_len = max_len - len(inp)
        padded_inputs.append(inp + [SPECIAL_TOKENS["<|pad|>"]] * pad_len)
        padded_targets.append(tgt + [-100] * pad_len)  # padding 部分也设为 -100
        padded_masks.append(msk + [0] * pad_len)

    return (
        torch.tensor(padded_inputs, dtype=torch.long),
        torch.tensor(padded_targets, dtype=torch.long),
        torch.tensor(padded_masks, dtype=torch.float),
    )


# ============================================================
# 5. SFT 训练（对比实验）
# ============================================================
print("\n" + "=" * 70)
print("第五部分：SFT 训练")
print("=" * 70)

# 重新初始化模型（更新 VOCAB_SIZE）
VOCAB_SIZE = max(id_to_char.keys()) + 10
model = MiniGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE, DROPOUT).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")

# ---- 第一步：模拟"预训练" ----
# 用少量文本让模型先学会基本的语言模式
print("\n📖 步骤 1：模拟预训练（让模型先学会基本语言模式）...")

pretrain_text = (
    "光合作用是植物利用阳光和水中制造养分的过程。"
    "具体来说，叶绿素吸收太阳光能量，将二氧化碳和水转化葡萄糖和氧气释放出来。"
    "今天天气很好。北京今天晴，最高温度28度。"
    "春风拂面柳丝长，桃杏花开满院香。"
    "燕子归来寻旧巢，一池碧水映斜阳。"
    "1+1=2。8×7=56。我要学习。"
)

pretrain_ids = encode(pretrain_text)
# 构造预训练样本：输入和目标差一位
pt_inputs = []
pt_targets = []
block_size = 64
for i in range(0, len(pretrain_ids) - block_size - 1, 8):
    pt_inputs.append(pretrain_ids[i:i+block_size])
    pt_targets.append(pretrain_ids[i+1:i+block_size+1])

pt_inputs_t = torch.tensor(pt_inputs, dtype=torch.long).to(device)
pt_targets_t = torch.tensor(pt_targets, dtype=torch.long).to(device)

# 预训练优化器
pt_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

print(f"  预训练数据: {len(pt_inputs)} 个样本")
model.train()
for step in range(200):
    # 随机选一批
    idx = torch.randint(0, len(pt_inputs), (4,))
    x = pt_inputs_t[idx]
    y = pt_targets_t[idx]
    logits, loss = model(x, y)
    pt_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    pt_optimizer.step()
    if step % 50 == 0:
        print(f"  Step {step:3d} | Loss: {loss.item():.4f}")

print(f"  预训练完成！最终 Loss: {loss.item():.4f}")

# 保存预训练模型的权重（用于对比）
pretrain_state = copy.deepcopy(model.state_dict())

# ---- 测试预训练模型面对指令的表现 ----
print("\n🧪 预训练模型面对指令的表现:")
test_instruction = "什么是光合作用？"
test_input = f"<|im_start|>user\n{test_instruction}<|im_end|>\n<|im_start|>assistant\n"
test_ids = encode(test_input)
test_tensor = torch.tensor([test_ids], dtype=torch.long).to(device)

model.eval()
with torch.no_grad():
    generated = model.generate(test_tensor, max_new_tokens=30, temperature=0.8, top_k=20)
generated_text = decode(generated[0].tolist())
print(f"  指令: {test_instruction}")
# 提取 assistant 之后的部分
if "<|im_start|>assistant\n" in generated_text:
    response = generated_text.split("<|im_start|>assistant\n")[-1]
    response = response.split("<|im_end|>")[0] if "<|im_end|>" in response else response
else:
    response = generated_text
print(f"  回答: {response}")
print(f"  → 预训练模型: 自顾自地续写，没有真正回答问题\n")

# ---- 第二步：SFT 微调 ----
print("📖 步骤 2：SFT 微调（教模型听指令）...")

# SFT 优化器（学习率小 10 倍！）
sft_optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
num_epochs = 30
for epoch in range(num_epochs):
    total_loss = 0.0
    n_batches = 0
    # 每轮遍历所有样本
    indices = list(range(len(all_inputs)))
    random.shuffle(indices)
    for i in range(0, len(indices), 4):
        batch_idx = indices[i:i+4]
        x, y, m = collate_fn(batch_idx)
        x, y, m = x.to(device), y.to(device), m.to(device)

        # 前向传播（带损失掩码）
        logits, loss = model(x, y, loss_mask=m)
        sft_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sft_optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    if epoch % 5 == 0 or epoch == num_epochs - 1:
        print(f"  Epoch {epoch:2d}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

print(f"  SFT 微调完成！")

# ---- 测试 SFT 后模型面对指令的表现 ----
print("\n🧪 SFT 模型面对指令的表现:")
test_cases = [
    "什么是光合作用？",
    "回答问题：1+1=?",
    "把下面的句子翻译成英文：今天天气很好",
]

for test_instruction in test_cases:
    test_input = f"<|im_start|>user\n{test_instruction}<|im_end|>\n<|im_start|>assistant\n"
    test_ids = encode(test_input)
    test_tensor = torch.tensor([test_ids], dtype=torch.long).to(device)

    model.eval()
    with torch.no_grad():
        generated = model.generate(test_tensor, max_new_tokens=25, temperature=0.5, top_k=15)
    generated_text = decode(generated[0].tolist())

    # 提取回答
    if "<|im_start|>assistant\n" in generated_text:
        response = generated_text.split("<|im_start|>assistant\n")[-1]
        response = response.split("<|im_end|>")[0] if "<|im_end|>" in response else response
    else:
        response = generated_text

    print(f"  指令: {test_instruction}")
    print(f"  回答: {response}")
    print()

# ============================================================
# 6. 灾难性遗忘演示
# ============================================================
print("=" * 70)
print("第六部分：灾难性遗忘演示")
print("=" * 70)

print("\n如果 SFT 学习率太大，模型会忘掉预训练学到的知识！")
print("我们来做一个对比实验...")

# 创建一个"大学习率"版本来演示灾难性遗忘
forgetting_model = MiniGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE, DROPOUT).to(device)
forgetting_model.load_state_dict(copy.deepcopy(pretrain_state))

# 用非常大的学习率做 SFT
forgetting_optimizer = torch.optim.AdamW(forgetting_model.parameters(), lr=5e-3)  # 学习率大 100 倍！

print("\n用超大学习率 (5e-3) 做 SFT...")
forgetting_model.train()
for epoch in range(20):
    total_loss = 0.0
    n_batches = 0
    indices = list(range(len(all_inputs)))
    random.shuffle(indices)
    for i in range(0, len(indices), 4):
        batch_idx = indices[i:i+4]
        x, y, m = collate_fn(batch_idx)
        x, y, m = x.to(device), y.to(device), m.to(device)
        logits, loss = forgetting_model(x, y, loss_mask=m)
        forgetting_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(forgetting_model.parameters(), 1.0)
        forgetting_optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    if epoch % 5 == 0 or epoch == 19:
        print(f"  Epoch {epoch:2d} | Loss: {avg_loss:.4f}")

# 测试灾难性遗忘的模型
print("\n🧪 灾难性遗忘模型面对指令:")
for test_instruction in test_cases[:2]:
    test_input = f"<|im_start|>user\n{test_instruction}<|im_end|>\n<|im_start|>assistant\n"
    test_ids = encode(test_input)
    test_tensor = torch.tensor([test_ids], dtype=torch.long).to(device)
    forgetting_model.eval()
    with torch.no_grad():
        generated = forgetting_model.generate(test_tensor, max_new_tokens=25, temperature=0.5, top_k=15)
    generated_text = decode(generated[0].tolist())
    if "<|im_start|>assistant\n" in generated_text:
        response = generated_text.split("<|im_start|>assistant\n")[-1]
        response = response.split("<|im_end|>")[0] if "<|im_end|>" in response else response
    else:
        response = generated_text
    print(f"  指令: {test_instruction}")
    print(f"  回答: {response}")
    print(f"  → 学习率太大，模型虽然学了 SFT 数据，但输出质量下降！")
    print()

# ============================================================
# 7. 损失掩码可视化
# ============================================================
print("=" * 70)
print("第七部分：损失掩码（Loss Masking）机制可视化")
print("=" * 70)

print("""
损失掩码是 SFT 最关键的技巧：

┌─────────────────────────────────────────────────────┐
│  输入序列:                                          │
│  <|im_start|> user                                  │
│  什么是光合作用？        ← 指令部分                 │
│  <|im_end|>                                         │
│  <|im_start|> assistant                            │
│  光合作用是...          ← 回答部分（计算损失）      │
│  <|im_end|>                                         │
├─────────────────────────────────────────────────────┤
│  Loss Mask:                                         │
│  ░░░░░░░░░░░░░░░░░░▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓            │
│  ░ = 指令部分（不计算损失，target 设为 -100）        │
│  ▓ = 回答部分（计算损失，模型学着生成回答）          │
├─────────────────────────────────────────────────────┤
│  效果：                                             │
│  模型不会学着"预测用户的问题"                       │
│  模型只会学着"在指令之后生成合适的回答"             │
└─────────────────────────────────────────────────────┘
""")

# 实际展示一个样本
inp, tgt, mask = all_inputs[0], all_targets[0], all_masks[0]
mask_visual = "".join(["▓" if m else "░" for m in mask])
print(f"  样本 0 的损失掩码: {mask_visual}")
print(f"  回答部分 token 数: {sum(mask)} / {len(mask)} 总 token")
print(f"  只训练 {sum(mask)/len(mask)*100:.0f}% 的 token → 效率更高、效果更好")

# ============================================================
# 8. 总结
# ============================================================
print("\n" + "=" * 70)
print("第八部分：总结")
print("=" * 70)

print("""
🔑 SFT 的核心要点:

1. SFT 数据 = (指令, 回答) 配对
   → 教模型"听到这种问题，就那样回答"

2. Chat Template 区分角色
   → <|im_start|>user ... <|im_end|> <|im_start|>assistant ...

3. 损失掩码（Loss Masking）
   → 只在助手回答部分计算损失
   → 指令部分 target 设为 -100（PyTorch 自动忽略）

4. 小学习率防遗忘
   → SFT 学习率 ≈ 预训练学习率 / 10
   → 太大会导致灾难性遗忘

5. SFT 学的是模仿，不是优化
   → 学会了"像人类标注者那样回答"
   → 但不一定是最好的回答 → 引出 RLHF/DPO

⚙️ 训练配置对比:
┌──────────┬──────────────┬──────────────┐
│ 参数      │ 预训练        │ SFT          │
├──────────┼──────────────┼──────────────┤
│ 数据量    │ 1TB+         │ 10K~100K 条  │
│ 学习率    │ 1e-4~6e-4    │ 1e-5~5e-5    │
│ 训练轮数  │ 1 epoch      │ 3~5 epochs   │
│ 训练成本  │ 百万美元      │ 几十~几百美元 │
└──────────┴──────────────┴──────────────┘

→ 明天预告: RLHF — 让模型不只模仿，而是优化"人类觉得好"的目标
""")

print("✅ Day 17 运行完毕！")
