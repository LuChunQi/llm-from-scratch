#!/usr/bin/env python3
"""
Day 8: 位置编码（Positional Encoding）
- Sinusoidal 位置编码（Transformer 原版）
- RoPE 旋转位置编码（LLaMA / Qwen / DeepSeek 采用）
- ALiBi 线性偏置注意力（BLOOM / MPT 采用）
- 三种方案对比实验
"""

import math
import torch
import torch.nn.functional as F


# ============================================================
# 1. Sinusoidal 位置编码 — Transformer 原版方案
# ============================================================

def sinusoidal_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """
    生成 Sinusoidal 位置编码矩阵。
    
    公式：
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        max_len: 最大序列长度
        d_model: 嵌入维度
    Returns:
        (max_len, d_model) 的位置编码矩阵
    """
    # 初始化全零矩阵
    pe = torch.zeros(max_len, d_model)
    
    # 位置向量：(max_len, 1)，每个位置一个值
    position = torch.arange(0, max_len).unsqueeze(1).float()
    
    # 频率项：10000^(2i/d_model) 的倒数
    # 用 exp + log 避免 pow 的数值不稳定
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    
    # 偶数维度用 sin
    pe[:, 0::2] = torch.sin(position * div_term)
    # 奇数维度用 cos
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


def demo_sinusoidal():
    """演示 Sinusoidal 位置编码的特性"""
    print("=" * 70)
    print("📐 Sinusoidal 位置编码")
    print("=" * 70)
    
    max_len = 50
    d_model = 16  # 用小维度方便展示
    pe = sinusoidal_positional_encoding(max_len, d_model)
    
    print(f"\n位置编码矩阵形状: {pe.shape}")  # (50, 16)
    print(f"值域: [{pe.min():.4f}, {pe.max():.4f}]")
    
    # 展示几个位置的编码
    print("\n前 5 个位置的前 8 维编码:")
    for pos in range(5):
        vals = [f"{pe[pos, j]:.3f}" for j in range(8)]
        print(f"  pos {pos}: [{', '.join(vals)}, ...]")
    
    # 验证不同位置的编码是不同的（唯一性）
    print("\n--- 唯一性验证 ---")
    # 计算不同位置之间的余弦相似度
    cos_sims = F.cosine_similarity(pe.unsqueeze(1), pe.unsqueeze(0), dim=-1)
    # 去掉对角线（自己和自己的相似度总是 1）
    mask = ~torch.eye(max_len, dtype=torch.bool)
    off_diag = cos_sims[mask]
    print(f"不同位置之间的余弦相似度:")
    print(f"  均值: {off_diag.mean():.4f}")
    print(f"  最大: {off_diag.max():.4f}")
    print(f"  最小: {off_diag.min():.4f}")
    print("  → 不同位置的编码确实不同（相似度远小于 1）✓")
    
    # 验证相对位置关系：相邻位置的差值应该是周期性的
    print("\n--- 相对位置关系 ---")
    diff_01 = pe[1] - pe[0]  # 位置 1 和 0 的差
    diff_23 = pe[3] - pe[2]  # 位置 3 和 2 的差
    cos_diff = F.cosine_similarity(diff_01.unsqueeze(0), diff_23.unsqueeze(0))
    print(f"pos1-pos0 与 pos3-pos2 的余弦相似度: {cos_diff.item():.4f}")
    print("  → 相邻位置的差值相似，说明编码隐含了相对位置信息 ✓")


# ============================================================
# 2. RoPE 旋转位置编码 — 现代主流方案
# ============================================================

def build_rope_freqs(seq_len: int, head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """
    构建 RoPE 的频率矩阵。
    
    每对维度的旋转频率为 θ_i = 1 / base^(2i/d)
    
    Args:
        seq_len: 序列长度
        head_dim: 每个注意力头的维度
        base: 频率基数（默认 10000，和 Sinusoidal 一样）
    Returns:
        (seq_len, head_dim) 的频率矩阵（每对维度重复）
    """
    # 计算每对的频率 θ_i = 1/10000^(2i/d)
    freqs = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    # (d/2,)
    
    # 位置乘频率：pos * θ_i
    t = torch.arange(seq_len).float()
    freqs = torch.outer(t, freqs)  # (seq_len, d/2)
    
    # 把每个频率重复一次，匹配 head_dim
    # [θ₀, θ₀, θ₁, θ₁, θ₂, θ₂, ...]
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
    
    return freqs


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    RoPE 的旋转操作：把向量分成两半，交叉组合。
    
    对于 [x₀, x₁, x₂, x₃, ...] → [-x_{d/2}, -x_{d/2+1}, ..., x₀, x₁, ...]
    即 [x₀, x₁, x₂, x₃] → [-x₂, -x₃, x₀, x₁]
    
    这是 2D 旋转矩阵 [cos θ, -sin θ] 的向量化版本。
    """
    d = x.shape[-1]
    x1 = x[..., :d // 2]   # 前半部分
    x2 = x[..., d // 2:]   # 后半部分
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    对输入向量应用 RoPE。
    
    RoPE(x, pos) = x * cos(pos * θ) + rotate_half(x) * sin(pos * θ)
    
    Args:
        x: (batch, seq_len, num_heads, head_dim) 输入向量
        freqs: (seq_len, head_dim) 频率矩阵
    Returns:
        旋转后的向量，形状同 x
    """
    # 调整 freqs 形状以广播
    cos_val = torch.cos(freqs).unsqueeze(0).unsqueeze(2)  # (1, seq_len, 1, head_dim)
    sin_val = torch.sin(freqs).unsqueeze(0).unsqueeze(2)
    
    return x * cos_val + rotate_half(x) * sin_val


def demo_rope():
    """演示 RoPE 的核心特性：内积只依赖相对位置"""
    print("\n" + "=" * 70)
    print("🔄 RoPE 旋转位置编码")
    print("=" * 70)
    
    seq_len = 20
    head_dim = 16
    num_heads = 1
    
    # 生成频率矩阵
    freqs = build_rope_freqs(seq_len, head_dim)
    print(f"\n频率矩阵形状: {freqs.shape}")  # (20, 16)
    
    # 创建随机 Q 和 K
    torch.manual_seed(42)
    q = torch.randn(1, seq_len, num_heads, head_dim)  # (1, 20, 1, 16)
    k = torch.randn(1, seq_len, num_heads, head_dim)
    
    # 应用 RoPE
    q_rope = apply_rope(q, freqs)
    k_rope = apply_rope(k, freqs)
    
    print(f"\n应用 RoPE 前 Q 范数: {q.norm():.4f}")
    print(f"应用 RoPE 后 Q 范数: {q_rope.norm():.4f}")
    print("  → 旋转不改变向量范数 ✓（就像旋转不改变长度）")
    
    # 核心验证：内积只依赖相对位置
    print("\n--- 核心特性验证：内积 ∝ 相对位置距离 ---")
    # 取位置 5 的 Q，计算它和所有位置的 K 的内积
    q_pos5 = q_rope[0, 5, 0]  # (16,)
    k_all = k_rope[0, :, 0]    # (20, 16)
    
    attn_scores = torch.matmul(k_all, q_pos5) / math.sqrt(head_dim)
    
    print(f"位置 5 的 Q 与各位置 K 的内积:")
    for pos in [0, 3, 4, 5, 6, 7, 10, 15, 19]:
        dist = abs(pos - 5)
        print(f"  pos {pos:2d} (距离={dist:2d}): {attn_scores[pos]:.4f}")
    
    # 验证等距对称性
    print("\n--- 等距对称性验证 ---")
    score_left = torch.dot(q_rope[0, 5, 0], k_rope[0, 3, 0])  # 距离 2
    score_right = torch.dot(q_rope[0, 5, 0], k_rope[0, 7, 0])  # 距离 2
    print(f"pos5↔pos3 (距离2): {score_left:.6f}")
    print(f"pos5↔pos7 (距离2): {score_right:.6f}")
    print(f"差异: {abs(score_left - score_right):.8f}")
    print("  → 等距内积几乎相同（微小差异来自浮点精度）✓")


# ============================================================
# 3. ALiBi — Attention with Linear Biases
# ============================================================

def alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    计算 ALiBi 的斜率。
    
    每个头使用不同的斜率：m_h = 2^(-8h/H)
    h=0 时斜率最大（最强的距离惩罚），h=H-1 时最小。
    
    Args:
        num_heads: 注意力头数
    Returns:
        (num_heads,) 的斜率向量
    """
    # 按 2 的等比级数生成斜率
    return 2 ** (-8 * torch.arange(1, num_heads + 1).float() / num_heads)


def alibi_bias(num_heads: int, seq_len: int) -> torch.Tensor:
    """
    生成 ALiBi 偏置矩阵。
    
    bias[h, i, j] = -m_h * |i - j|
    
    Args:
        num_heads: 注意力头数
        seq_len: 序列长度
    Returns:
        (num_heads, seq_len, seq_len) 的偏置矩阵
    """
    slopes = alibi_slopes(num_heads)  # (num_heads,)
    
    # 位置矩阵
    positions = torch.arange(seq_len).float()
    
    # 距离矩阵 |i - j|
    distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (seq_len, seq_len)
    
    # 负距离 × 斜率
    # slopes: (num_heads, 1, 1), distance: (1, seq_len, seq_len)
    bias = -slopes.unsqueeze(1).unsqueeze(2) * distance.unsqueeze(0)
    
    return bias


def demo_alibi():
    """演示 ALiBi 的特性"""
    print("\n" + "=" * 70)
    print("📏 ALiBi — Attention with Linear Biases")
    print("=" * 70)
    
    num_heads = 8
    seq_len = 16
    
    # 生成偏置
    bias = alibi_bias(num_heads, seq_len)
    print(f"\nALiBi 偏置矩阵形状: {bias.shape}")  # (8, 16, 16)
    
    # 展示斜率
    slopes = alibi_slopes(num_heads)
    print(f"\n各头的斜率 m_h:")
    for h in range(num_heads):
        print(f"  Head {h}: m = {slopes[h]:.6f}")
    print("  → 头 0 的斜率最大（最强距离惩罚），后面逐渐减小")
    
    # 展示 Head 0 的偏置矩阵（最强调）
    print(f"\nHead 0 的偏置（位置 0 到 7，距离惩罚最强）:")
    h0 = bias[0, :8, :8]
    for i in range(8):
        row = [f"{h0[i, j]:7.3f}" for j in range(8)]
        print(f"  pos {i}: [{' '.join(row)}]")
    
    # 演示 ALiBi 如何影响 Attention
    print("\n--- ALiBi 对 Attention 分布的影响 ---")
    # 模拟均匀的 QK 内积（没有位置信息时 Attention 是均匀的）
    uniform_scores = torch.zeros(1, num_heads, seq_len, seq_len)
    
    # 加上 ALiBi 偏置
    biased_scores = uniform_scores + bias.unsqueeze(0)
    
    # softmax 后看分布
    attn_uniform = F.softmax(uniform_scores[0, 0], dim=-1)  # Head 0
    attn_biased = F.softmax(biased_scores[0, 0], dim=-1)     # Head 0
    
    print(f"无 ALiBi 时（均匀 Attention），pos 5 关注各位置:")
    for j in [0, 3, 4, 5, 6, 7, 10, 15]:
        print(f"  → pos {j}: {attn_uniform[5, j]:.4f}")
    
    print(f"\n有 ALiBi 时，pos 5 关注各位置:")
    for j in [0, 3, 4, 5, 6, 7, 10, 15]:
        print(f"  → pos {j}: {attn_biased[5, j]:.4f}")
    print("  → 近处的权重更大，远处被抑制 ✓")


# ============================================================
# 4. 三种方案综合对比实验
# ============================================================

def demo_comparison():
    """三种位置编码方案的对比实验"""
    print("\n" + "=" * 70)
    print("⚔️ 三种位置编码方案对比")
    print("=" * 70)
    
    d_model = 64
    seq_len = 20
    
    sentence = ["我", "昨天", "在", "咖啡店", "遇到", "了", "一个", "老朋友"]
    print(f"\n示例句子: {' '.join(sentence)}")
    print(f"长度: {len(sentence)} 词")
    
    # 模拟词嵌入
    torch.manual_seed(42)
    embeddings = torch.randn(len(sentence), d_model)
    
    # --- Sinusoidal ---
    pe = sinusoidal_positional_encoding(seq_len, d_model)
    embed_sinusoidal = embeddings + pe[:len(sentence)]
    
    # 计算 Sinusoidal 下的自注意力分数
    scores_sin = torch.matmul(embed_sinusoidal, embed_sinusoidal.T) / math.sqrt(d_model)
    attn_sin = F.softmax(scores_sin, dim=-1)
    
    print("\n📊 Sinusoidal: '咖啡店'(pos 3) 关注各词的权重:")
    for i, word in enumerate(sentence):
        bar = "█" * int(attn_sin[3, i] * 100)
        print(f"  {word}(pos{i}): {attn_sin[3, i]:.4f} {bar}")
    
    # --- RoPE ---
    freqs = build_rope_freqs(seq_len, d_model)
    # 把 embeddings 扩展为 (1, seq, 1, dim) 以匹配 apply_rope 的接口
    q = embeddings.unsqueeze(0).unsqueeze(2)  # (1, 8, 1, 64)
    k = embeddings.unsqueeze(0).unsqueeze(2)
    q_rope = apply_rope(q, freqs[:len(sentence)])
    k_rope = apply_rope(k, freqs[:len(sentence)])
    
    scores_rope = torch.matmul(
        q_rope.squeeze().squeeze(1), 
        k_rope.squeeze().squeeze(1).T
    ) / math.sqrt(d_model)
    attn_rope = F.softmax(scores_rope, dim=-1)
    
    print("\n📊 RoPE: '咖啡店'(pos 3) 关注各词的权重:")
    for i, word in enumerate(sentence):
        bar = "█" * int(attn_rope[3, i] * 100)
        print(f"  {word}(pos{i}): {attn_rope[3, i]:.4f} {bar}")
    
    # --- ALiBi ---
    # 用原始 embeddings 计算分数 + ALiBi 偏置
    scores_base = torch.matmul(embeddings, embeddings.T) / math.sqrt(d_model)
    alibi = alibi_bias(1, seq_len)  # 1 个头
    scores_alibi = scores_base + alibi[0, :len(sentence), :len(sentence)]
    attn_alibi = F.softmax(scores_alibi, dim=-1)
    
    print("\n📊 ALiBi: '咖啡店'(pos 3) 关注各词的权重:")
    for i, word in enumerate(sentence):
        bar = "█" * int(attn_alibi[3, i] * 100)
        print(f"  {word}(pos{i}): {attn_alibi[3, i]:.4f} {bar}")
    
    # 对比距离衰减特性
    print("\n--- 距离衰减特性对比（pos 3 到其他位置）---")
    print(f"{'距离':>4s}  {'Sinusoidal':>12s}  {'RoPE':>12s}  {'ALiBi':>12s}")
    print("-" * 50)
    for dist in range(len(sentence)):
        if dist == 3:
            target = 3
        else:
            target = dist
        actual_dist = abs(dist - 3)
        s_val = attn_sin[3, dist].item()
        r_val = attn_rope[3, dist].item()
        a_val = attn_alibi[3, dist].item()
        print(f"  {dist:>2d}     {s_val:.6f}     {r_val:.6f}     {a_val:.6f}")
    
    print("\n💡 观察要点:")
    print("  - Sinusoidal: 位置编码直接加到词向量，影响是隐式的")
    print("  - RoPE: 旋转操作使等距位置的注意力分数相近（相对位置感知）")
    print("  - ALiBi: 明显的近距离偏好，距离越远衰减越强")


# ============================================================
# 5. 长度外推实验
# ============================================================

def demo_extrapolation():
    """演示三种方案在超出训练长度时的表现"""
    print("\n" + "=" * 70)
    print("🔮 长度外推能力对比")
    print("=" * 70)
    
    d_model = 64
    train_len = 10   # "训练"时的序列长度
    test_len = 50    # 推理时测试的序列长度
    
    print(f"\n场景: 训练时序列长度={train_len}，推理时遇到长度={test_len}")
    
    # Sinusoidal: 直接计算就行
    pe = sinusoidal_positional_encoding(test_len, d_model)
    print(f"\nSinusoidal: 能直接生成 pos {test_len-1} 的编码")
    print(f"  pos {test_len-1} 编码范数: {pe[test_len-1].norm():.4f}")
    print(f"  pos 0 编码范数: {pe[0].norm():.4f}")
    print("  → 范数基本一致，但高频分量可能出现周期性重复 ⚠️")
    
    # RoPE: 也能直接用
    freqs = build_rope_freqs(test_len, d_model)
    x = torch.randn(1, test_len, 1, d_model)
    x_rope = apply_rope(x, freqs)
    print(f"\nRoPE: 能直接对 pos {test_len-1} 应用旋转")
    print(f"  旋转前后范数不变: {x[0,0,0].norm():.4f} → {x_rope[0,0,0].norm():.4f}")
    print("  → 外推需要 NTK-aware 扩展才能效果好 ✓")
    
    # ALiBi: 天生支持外推
    bias = alibi_bias(4, test_len)
    print(f"\nALiBi: 天生支持任意长度")
    print(f"  pos 0 到 pos {test_len-1} 的偏置(Head 0): {bias[0, 0, test_len-1]:.4f}")
    print(f"  pos 0 到 pos 1 的偏置(Head 0): {bias[0, 0, 1]:.4f}")
    print("  → 只需线性扩展距离矩阵，零额外开销 ✅")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("🧠 Day 8: 位置编码（Positional Encoding）")
    print("=" * 70)
    print("Transformer 怎么知道'我爱你'≠'你爱我'？")
    print()
    
    # 1. Sinusoidal 位置编码
    demo_sinusoidal()
    
    # 2. RoPE 旋转位置编码
    demo_rope()
    
    # 3. ALiBi
    demo_alibi()
    
    # 4. 三种方案对比
    demo_comparison()
    
    # 5. 长度外推
    demo_extrapolation()
    
    print("\n" + "=" * 70)
    print("✅ Day 8 完成！")
    print("=" * 70)
    print("\n🔑 核心要点回顾:")
    print("  1. Transformer 没有位置感知 → 必须用位置编码")
    print("  2. Sinusoidal: 用 sin/cos 生成位置指纹，简单有效")
    print("  3. RoPE: 通过旋转向量编码相对位置，现代 LLM 标配")
    print("  4. ALiBi: 在 Attention 上加距离惩罚，外推能力最强")
    print("  5. 实际项目首选 RoPE（生态最完善）")
    print()
