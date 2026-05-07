#!/usr/bin/env python3
"""
Day 5 - Transformer 骨架代码
============================
用 PyTorch 从零搭建一个完整的 Transformer Encoder 模型。

包含：
  - Scaled Dot-Product Attention（缩放点积注意力）
  - Multi-Head Attention（多头注意力）
  - Position-wise Feed-Forward Network（前馈网络）
  - Positional Encoding（位置编码）
  - Transformer Encoder Layer（编码器层）
  - Transformer Encoder（多层堆叠）
  - 完整前向传播演示
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 缩放点积注意力 (Scaled Dot-Product Attention)
# ============================================================
# 公式: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
# 核心: 用 Q 和 K 的点积衡量"相关性"，除以 sqrt(d_k) 稳定梯度，
#       softmax 归一化后对 V 加权求和。

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    参数:
        Q: Query 矩阵, shape [batch, heads, seq_len, d_k]
        K: Key   矩阵, shape [batch, heads, seq_len, d_k]
        V: Value 矩阵, shape [batch, heads, seq_len, d_v]
        mask: 可选掩码, shape 可广播到 [batch, heads, seq_len, seq_len]
              被掩码的位置填 -1e9（softmax 后趋近于 0）

    返回:
        output: 注意力输出, shape [batch, heads, seq_len, d_v]
        attn_weights: 注意力权重, shape [batch, heads, seq_len, seq_len]
    """
    # 获取 key 的维度，用于缩放
    d_k = K.size(-1)

    # ---- 步骤 1: 计算 Q 和 K 的点积（相似度矩阵）----
    # Q: [batch, heads, seq_len, d_k]
    # K^T: [batch, heads, d_k, seq_len]
    # scores: [batch, heads, seq_len, seq_len]
    # scores[b,h,i,j] = 第 b 个样本、第 h 个头中，位置 i 对位置 j 的"原始关注度"
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # ---- 步骤 2: 应用掩码（如果有的话）----
    # Decoder 中使用：把"未来"位置的 score 设为极小值
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # ---- 步骤 3: Softmax 归一化 → 注意力权重 ----
    # 每一行（对应一个 query 位置）对所有 key 位置做 softmax
    # attn_weights[b,h,i,:] 表示位置 i 对所有位置的注意力分布
    attn_weights = F.softmax(scores, dim=-1)

    # ---- 步骤 4: 用注意力权重对 V 加权求和 ----
    # attn_weights: [batch, heads, seq_len, seq_len]
    # V: [batch, heads, seq_len, d_v]
    # output: [batch, heads, seq_len, d_v]
    output = torch.matmul(attn_weights, V)

    return output, attn_weights


# ============================================================
# 2. 多头注意力 (Multi-Head Attention)
# ============================================================
# 为什么需要多头？单头注意力只能学一种"关注模式"，
# 多头让模型同时从多个视角关注不同的信息。

class MultiHeadAttention(nn.Module):
    """
    多头注意力层

    参数:
        d_model: 模型总维度（所有头的维度之和）
        n_heads: 注意力头数
    """

    def __init__(self, d_model, n_heads):
        super().__init__()

        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model        # 模型总维度
        self.n_heads = n_heads        # 注意力头数
        self.d_k = d_model // n_heads  # 每个头的维度

        # Q、K、V 的线性投影层：把 d_model 维映射到 d_model 维
        # 实际上每个头只用到 d_k 维，但这里把所有头打包在一起算
        self.W_q = nn.Linear(d_model, d_model)  # Query 投影
        self.W_k = nn.Linear(d_model, d_model)  # Key 投影
        self.W_v = nn.Linear(d_model, d_model)  # Value 投影

        # 输出投影层：把多头拼接后的结果映射回 d_model 维
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        """
        前向传播

        参数:
            query: [batch, seq_len, d_model]
            key:   [batch, seq_len, d_model]
            value: [batch, seq_len, d_model]
            mask:  可选掩码

        返回:
            output: [batch, seq_len, d_model]
            attn_weights: [batch, n_heads, seq_len, seq_len]
        """
        batch_size = query.size(0)

        # ---- 线性投影 + 拆分成多头 ----
        # 先投影到 d_model 维，然后 reshape 成 [batch, seq_len, n_heads, d_k]
        # 再转置成 [batch, n_heads, seq_len, d_k]，方便并行计算
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # ---- 计算注意力 ----
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # ---- 合并多头 ----
        # [batch, n_heads, seq_len, d_k] → [batch, seq_len, n_heads, d_k] → [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )

        # ---- 输出投影 ----
        output = self.W_o(attn_output)

        return output, attn_weights


# ============================================================
# 3. 位置编码 (Positional Encoding)
# ============================================================
# Transformer 没有循环结构，不知道词的顺序。
# 位置编码给每个位置一个唯一的"条形码"，
# 加到词嵌入上，让模型感知位置信息。

class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码（原论文方案）

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        参数:
            d_model: 模型维度
            max_len: 最大序列长度
            dropout: Dropout 比率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # ---- 预计算位置编码矩阵 ----
        # pe: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # position: [0, 1, 2, ..., max_len-1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term: 10000^(2i/d_model) 的倒数，用于频率调节
        # 这个让低维度用低频（变化慢），高维度用高频（变化快）
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数列
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数列

        # 加一个 batch 维度: [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        # 注册为 buffer（不参与梯度更新，但会随模型保存/加载）
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数:
            x: 输入嵌入, shape [batch, seq_len, d_model]

        返回:
            加入位置编码后的输出, shape [batch, seq_len, d_model]
        """
        # 取前 seq_len 个位置的编码，加到输入上
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# 4. 前馈网络 (Position-wise Feed-Forward Network)
# ============================================================
# 对每个位置独立应用相同的两层全连接网络
# FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
# 中间维度通常是 d_model 的 4 倍（先扩展再压缩）

class PositionWiseFFN(nn.Module):
    """
    位置级前馈网络

    参数:
        d_model: 输入/输出维度
        d_ff: 中间隐藏层维度（通常 = 4 * d_model）
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)   # 扩展: d_model → d_ff
        self.fc2 = nn.Linear(d_ff, d_model)    # 压缩: d_ff → d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        参数:
            x: [batch, seq_len, d_model]
        返回:
            [batch, seq_len, d_model]
        """
        # 先扩展 → 激活 → Dropout → 压缩
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


# ============================================================
# 5. Transformer Encoder Layer（一个编码器层）
# ============================================================
# 每层包含：
#   1. Multi-Head Self-Attention + Add & Norm
#   2. Feed-Forward Network + Add & Norm
# "Add" = 残差连接，"Norm" = Layer Normalization

class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层

    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        d_ff: FFN 中间维度
        dropout: Dropout 比率
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # 子层 1: 多头自注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads)

        # 子层 2: 前馈网络
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)

        # Layer Normalization（每个子层后都有一层）
        self.norm1 = nn.LayerNorm(d_model)  # 注意力之后的归一化
        self.norm2 = nn.LayerNorm(d_model)  # FFN 之后的归一化

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        参数:
            x: [batch, seq_len, d_model]
            mask: 可选掩码
        返回:
            [batch, seq_len, d_model]
        """
        # ---- 子层 1: Multi-Head Self-Attention + Add & Norm ----
        # 自注意力: Q=K=V=x（所以叫"自"注意力）
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        # 残差连接 + Dropout + LayerNorm
        x = self.norm1(x + self.dropout1(attn_output))

        # ---- 子层 2: FFN + Add & Norm ----
        ffn_output = self.ffn(x)
        # 残差连接 + Dropout + LayerNorm
        x = self.norm2(x + self.dropout2(ffn_output))

        return x, attn_weights


# ============================================================
# 6. Transformer Encoder（多层堆叠）
# ============================================================
# 把 N 个 Encoder Layer 叠在一起

class TransformerEncoder(nn.Module):
    """
    Transformer 编码器

    参数:
        vocab_size: 词汇表大小
        d_model: 模型维度
        n_heads: 注意力头数
        d_ff: FFN 中间维度
        n_layers: 编码器层数
        max_len: 最大序列长度
        dropout: Dropout 比率
    """

    def __init__(self, vocab_size, d_model=512, n_heads=8, d_ff=2048,
                 n_layers=6, max_len=5000, dropout=0.1):
        super().__init__()

        self.d_model = d_model

        # Token Embedding: 把 token ID 转成 d_model 维向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 位置编码
        self.position_encoding = PositionalEncoding(d_model, max_len, dropout)

        # N 个 Encoder Layer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 最终的 LayerNorm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        参数:
            x: 输入 token IDs, shape [batch, seq_len]
            mask: 可选掩码
        返回:
            encoder_output: [batch, seq_len, d_model]
            all_attn_weights: 每层的注意力权重列表
        """
        # ---- Token Embedding + 缩放 ----
        # 原论文中把 embedding 乘以 sqrt(d_model)，让embedding 的量级
        # 和位置编码匹配（位置编码的值大约在 [-1, 1]，embedding 可能很小）
        x = self.token_embedding(x) * math.sqrt(self.d_model)

        # ---- 加上位置编码 ----
        x = self.position_encoding(x)

        # ---- 逐层通过 Encoder Layer ----
        all_attn_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, mask)
            all_attn_weights.append(attn_weights)

        # ---- 最终 LayerNorm ----
        x = self.norm(x)

        return x, all_attn_weights


# ============================================================
# 7. 完整的前向传播演示
# ============================================================

def demo_transformer():
    """演示 Transformer Encoder 的完整前向传播"""

    print("=" * 60)
    print("  Day 5 - Transformer 骨架代码演示")
    print("=" * 60)
    print()

    # ---- 超参数 ----
    vocab_size = 1000     # 词汇表大小
    d_model = 128         # 模型维度（为了演示用小一点的值）
    n_heads = 4           # 注意力头数（128 / 4 = 32 per head）
    d_ff = 512            # FFN 中间维度（= 4 * d_model）
    n_layers = 3          # 编码器层数
    batch_size = 2        # 批大小
    seq_len = 10          # 序列长度

    print(f"📋 模型配置:")
    print(f"   词汇表大小: {vocab_size}")
    print(f"   模型维度 (d_model): {d_model}")
    print(f"   注意力头数: {n_heads}")
    print(f"   每头维度 (d_k): {d_model // n_heads}")
    print(f"   FFN 中间维度: {d_ff}")
    print(f"   编码器层数: {n_layers}")
    print(f"   输入批大小: {batch_size}")
    print(f"   序列长度: {seq_len}")
    print()

    # ---- 创建模型 ----
    model = TransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
    )

    # ---- 统计参数量 ----
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 参数统计:")
    print(f"   总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   约 {total_params / 1e6:.2f}M 参数")
    print()

    # ---- 构造随机输入 ----
    # 模拟两个句子，每个 10 个 token
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"📥 输入数据:")
    print(f"   shape: {input_ids.shape}")
    print(f"   样本 1: {input_ids[0].tolist()}")
    print(f"   样本 2: {input_ids[1].tolist()}")
    print()

    # ---- 前向传播 ----
    model.eval()  # 切换到评估模式（关闭 Dropout）
    with torch.no_grad():
        output, all_attn_weights = model(input_ids)

    # ---- 输出结果 ----
    print(f"📤 输出结果:")
    print(f"   输出 shape: {output.shape}")
    print(f"   (期望: [{batch_size}, {seq_len}, {d_model}])")
    print(f"   ✅ 形状正确!" if output.shape == torch.Size([batch_size, seq_len, d_model]) else "   ❌ 形状不匹配!")
    print()

    # ---- 注意力权重分析 ----
    print(f"🔍 注意力权重分析:")
    for i, attn in enumerate(all_attn_weights):
        print(f"   第 {i+1} 层:")
        print(f"     shape: {attn.shape} (batch, heads, seq_len, seq_len)")

        # 看第一个样本、第一个头的注意力分布
        head_0_attn = attn[0, 0]  # [seq_len, seq_len]
        print(f"     第1个样本-第1个头 注意力矩阵 (第1行, 即位置0对各位置的关注度):")
        for j in range(seq_len):
            print(f"       位置0 → 位置{j}: {head_0_attn[0, j].item():.4f}")
        print()

    # ---- 展示位置编码 ----
    print(f"📏 位置编码可视化 (前4个位置, 前8个维度):")
    pe = model.position_encoding.pe[0]  # [max_len, d_model]
    for pos in range(4):
        vals = [f"{pe[pos, d].item():+.4f}" for d in range(8)]
        print(f"   位置 {pos}: [{', '.join(vals)}, ...]")
    print()

    # ---- 展示 Embedding + 位置编码的效果 ----
    print(f"🔤 Token Embedding + 位置编码:")
    with torch.no_grad():
        raw_emb = model.token_embedding(input_ids) * math.sqrt(model.d_model)
        emb_with_pe = model.position_encoding(raw_emb)
    print(f"   原始 embedding 范围: [{raw_emb.min().item():.4f}, {raw_emb.max().item():.4f}]")
    print(f"   加位置编码后范围:    [{emb_with_pe.min().item():.4f}, {emb_with_pe.max().item():.4f}]")
    print(f"   位置编码引入的变化量级: ~{(emb_with_pe - raw_emb).abs().mean().item():.4f}")
    print()

    # ---- 模型结构概览 ----
    print(f"🏗️ 模型结构:")
    print(f"   TransformerEncoder")
    print(f"   ├── token_embedding: Embedding({vocab_size}, {d_model})")
    print(f"   ├── position_encoding: PositionalEncoding(d_model={d_model})")
    for i in range(n_layers):
        print(f"   ├── layers[{i}]: TransformerEncoderLayer")
        print(f"   │   ├── self_attn: MultiHeadAttention(heads={n_heads}, d_k={d_model//n_heads})")
        print(f"   │   └── ffn: FFN({d_model} → {d_ff} → {d_model})")
    print(f"   └── norm: LayerNorm({d_model})")
    print()

    print("=" * 60)
    print("  ✅ Transformer Encoder 演示完成！")
    print("  接下来 Day 6-15 会逐个深入每个组件的细节")
    print("=" * 60)


if __name__ == "__main__":
    demo_transformer()
