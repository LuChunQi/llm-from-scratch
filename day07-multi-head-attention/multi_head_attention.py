#!/usr/bin/env python3
"""
Day 7: Multi-Head Attention — 多头注意力机制
==============================================

本代码包含：
1. 手写 Multi-Head Attention（纯 PyTorch）
2. 可视化不同头的注意力热力图
3. 对比单头 vs 多头的效果差异

运行方式：python3 multi_head_attention.py
依赖：pip install torch matplotlib numpy
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# 第一部分：Multi-Head Attention 实现
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    核心思想：
    - 把 Q、K、V 分别投影到 h 个低维子空间
    - 每个子空间独立做 Scaled Dot-Product Attention
    - 拼接所有头的结果，再做一次线性投影
    
    类比：就像编辑部里有多个记者，每人负责调查一个角度，
    最后主编把所有记者的报道汇总成一篇完整的文章。
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        参数：
            d_model:   模型维度（如 512）
            num_heads: 注意力头的数量（如 8）
            dropout:   dropout 概率
        """
        super().__init__()
        
        # 确保 d_model 能被 num_heads 整除
        assert d_model % num_heads == 0, \
            f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        # 每个头的维度
        self.d_k = d_model // num_heads
        
        # Q、K、V 的投影矩阵（合并所有头）
        # 用一个 [d_model, d_model] 的大矩阵一次投影出所有头
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        
        # 输出投影矩阵
        self.W_O = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None, return_attn=False):
        """
        前向传播
        
        参数：
            query:       [batch, seq_len_q, d_model]
            key:         [batch, seq_len_k, d_model]
            value:       [batch, seq_len_k, d_model]
            mask:        可选的 mask 矩阵
            return_attn: 是否返回注意力权重（用于可视化）
        
        返回：
            output:      [batch, seq_len_q, d_model]
            attn_weights: [batch, num_heads, seq_len_q, seq_len_k]（可选）
        """
        batch_size = query.size(0)
        
        # --- 第1步：线性投影 + reshape 成多头 ---
        # [batch, seq_len, d_model] -> [batch, seq_len, num_heads, d_k]
        # -> [batch, num_heads, seq_len, d_k]
        Q = self.W_Q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_K(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_V(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # --- 第2步：计算 Scaled Dot-Product Attention ---
        # Q @ K^T: [batch, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用 mask（如果有）
        if mask is not None:
            # mask 为 0 的位置设为 -inf，softmax 后变成 0
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # softmax 得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 注意力权重 @ V: [batch, num_heads, seq_len_q, d_k]
        attn_output = torch.matmul(attn_weights, V)
        
        # --- 第3步：拼接所有头 ---
        # [batch, num_heads, seq_len_q, d_k] -> [batch, seq_len_q, num_heads, d_k]
        # -> [batch, seq_len_q, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # --- 第4步：输出投影 ---
        output = self.W_O(attn_output)
        
        if return_attn:
            return output, attn_weights
        return output


class SingleHeadAttention(nn.Module):
    """
    单头注意力（用于对比）
    
    和 Multi-Head Attention 的区别：
    - 只有一个头，d_k = d_model
    - 没有子空间分解
    """
    
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None, return_attn=False):
        batch_size = query.size(0)
        
        # 投影（不做多头拆分）
        Q = self.W_Q(query)   # [batch, seq_len, d_model]
        K = self.W_K(key)
        V = self.W_V(value)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        output = self.W_O(attn_output)
        
        if return_attn:
            return output, attn_weights.unsqueeze(1)  # 加一个 head 维度方便对比
        return output


# ============================================================
# 第二部分：可视化函数
# ============================================================

def print_attention_heatmap(attn_weights, tokens, head_idx=None):
    """
    用字符画打印注意力热力图
    
    参数：
        attn_weights: [num_heads, seq_len, seq_len] 或 [seq_len, seq_len]
        tokens:       token 列表
        head_idx:     如果是多头，指定看哪个头
    """
    if attn_weights.dim() == 3:
        # 多头情况
        if head_idx is not None:
            attn = attn_weights[head_idx].detach().numpy()
            print(f"\n  === 注意力头 {head_idx} 的注意力矩阵 ===\n")
        else:
            # 打印所有头
            for h in range(attn_weights.size(0)):
                attn = attn_weights[h].detach().numpy()
                _print_single_heatmap(attn, tokens, f"头 {h}")
            return
    else:
        attn = attn_weights.detach().numpy()
        print(f"\n  === 单头注意力矩阵 ===\n")
    
    _print_single_heatmap(attn, tokens, "")


def _print_single_heatmap(attn, tokens, label):
    """打印单个注意力热力图"""
    if label:
        print(f"  --- {label} ---")
    
    # 缩短 token 显示
    short_tokens = [t[:6] for t in tokens]
    
    # 打印表头
    print(f"  {'':>8}", end="")
    for t in short_tokens:
        print(f"{t:>7}", end="")
    print()
    
    # 打印每行
    for i, token in enumerate(short_tokens):
        print(f"  {token:>8}", end="")
        for j in range(len(tokens)):
            val = attn[i, j]
            # 用符号表示大小
            if val > 0.3:
                symbol = "███"
            elif val > 0.15:
                symbol = "▓▓▓"
            elif val > 0.08:
                symbol = "▒▒▒"
            elif val > 0.03:
                symbol = "░░░"
            else:
                symbol = " · "
            print(f" {symbol}", end="")
        print()
    print()


# ============================================================
# 第三部分：实验与演示
# ============================================================

def demo_multi_head_attention():
    """演示：多头注意力的基本使用"""
    print("=" * 70)
    print("  实验1：Multi-Head Attention 基本演示")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 模型参数
    d_model = 64     # 用小一点的维度方便演示
    num_heads = 4    # 4 个头，每个头 16 维
    seq_len = 6
    batch_size = 1
    
    # 创建模型
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 模拟输入：6 个 token 的嵌入
    # 假设句子："The cat sat on the mat"
    tokens = ["The", "cat", "sat", "on", "the", "mat"]
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n  输入维度: {x.shape}")
    print(f"  d_model={d_model}, num_heads={num_heads}, d_k={d_model//num_heads}")
    print(f"  句子: {' '.join(tokens)}")
    
    # 前向传播
    output, attn_weights = mha(x, x, x, return_attn=True)
    
    print(f"\n  输出维度: {output.shape}")
    print(f"  注意力权重维度: {attn_weights.shape}")
    print(f"  (= [batch={attn_weights.size(0)}, heads={attn_weights.size(1)}, "
          f"seq={attn_weights.size(2)}, seq={attn_weights.size(3)}])")
    
    # 打印每个头的注意力模式
    print(f"\n  注意力矩阵图例：███ >0.3 | ▓▓▓ >0.15 | ▒▒▒ >0.08 | ░░░ >0.03 | · ≈0")
    
    # [batch, heads, seq, seq] -> [heads, seq, seq]
    attn_np = attn_weights[0]
    print_attention_heatmap(attn_np, tokens)
    
    # 分析每个头的"特性"——哪个 token 获得了最多注意力
    print("  每个头最关注的 token（按行平均）:")
    for h in range(num_heads):
        col_sums = attn_np[h].mean(dim=0)  # 每个 token 被关注的平均权重
        top_idx = col_sums.argmax().item()
        print(f"    头 {h}: 最关注 '{tokens[top_idx]}' (平均权重={col_sums[top_idx]:.4f})")


def demo_head_diversity():
    """演示：不同头学到了不同的关注模式"""
    print("\n" + "=" * 70)
    print("  实验2：不同头的注意力多样性")
    print("=" * 70)
    
    torch.manual_seed(123)
    
    # 构造一个有指代消解问题的句子
    # "The dog chased the cat but it ran away" -> "it" 指代谁？
    tokens = ["The", "dog", "chased", "the", "cat", "but", "it", "ran", "away"]
    seq_len = len(tokens)
    d_model = 64
    num_heads = 4
    
    # 创建模型，用特殊初始化让不同的头有不同的行为倾向
    mha = MultiHeadAttention(d_model, num_heads)
    
    # 构造输入嵌入，让 "dog"、"cat"、"it" 有特殊关系
    x = torch.randn(1, seq_len, d_model)
    # 让 "it" (index 6) 的嵌入更接近 "dog" (index 1) 而不是 "cat" (index 4)
    x[0, 6] = x[0, 1] * 0.7 + torch.randn(d_model) * 0.3  # it 偏向 dog
    
    output, attn_weights = mha(x, x, x, return_attn=True)
    
    print(f"\n  句子: '{' '.join(tokens)}'")
    print(f"  问题: 'it' 指代 'dog' 还是 'cat'？")
    
    # 重点看 "it" (index 6) 对其他词的注意力
    attn_np = attn_weights[0]  # [heads, seq, seq]
    
    print(f"\n  'it' 在各头的注意力分配:")
    print(f"  {'词':>10}", end="")
    for h in range(num_heads):
        print(f"  {'头'+str(h):>8}", end="")
    print()
    print(f"  {'-'*50}")
    
    for j, token in enumerate(tokens):
        print(f"  {token:>10}", end="")
        for h in range(num_heads):
            w = attn_np[h, 6, j].item()  # "it" 对 token j 的注意力
            bar = "█" * int(w * 30)
            print(f"  {w:>5.3f} {bar:<3}", end="")
        print()
    
    # 统计每个头认为 "it" 最可能是谁
    print(f"\n  各头对 'it' 的指代判断:")
    for h in range(num_heads):
        it_attn = attn_np[h, 6, :]  # "it" 对所有词的注意力
        # 排除自己（index 6）和标点
        top_vals, top_idxs = it_attn.topk(3)
        top_tokens = [(tokens[idx], val.item()) for idx, val in zip(top_idxs, top_vals) if idx != 6]
        if top_tokens:
            print(f"    头 {h}: 最可能指代 '{top_tokens[0][0]}' (权重={top_tokens[0][1]:.4f})")


def demo_single_vs_multi_head():
    """演示：单头 vs 多头的表达能力对比"""
    print("\n" + "=" * 70)
    print("  实验3：单头 vs 多头 — 表达能力对比")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    d_model = 64
    seq_len = 8
    num_heads = 4
    
    # 创建模型
    single_head = SingleHeadAttention(d_model)
    multi_head = MultiHeadAttention(d_model, num_heads)
    
    # 随机输入
    x = torch.randn(1, seq_len, d_model)
    
    # 前向传播
    out_single, attn_single = single_head(x, x, x, return_attn=True)
    out_multi, attn_multi = multi_head(x, x, x, return_attn=True)
    
    print(f"\n  单头输出方差: {out_single.var().item():.6f}")
    print(f"  多头输出方差: {out_multi.var().item():.6f}")
    
    # 计算注意力权重的"多样性"
    # 单头只有一个模式
    single_attn = attn_single[0, 0]  # [seq, seq]
    
    # 多头有多个模式，计算头间的余弦距离来衡量多样性
    multi_attn = attn_multi[0]  # [heads, seq, seq]
    
    # 把每个头的注意力展平成向量
    head_vectors = multi_attn.reshape(num_heads, -1)  # [heads, seq*seq]
    
    print(f"\n  --- 多头注意力模式的余弦相似度矩阵 ---")
    print(f"  (数值越低 = 两个头越不同 = 多样性越好)")
    print(f"  {'':>6}", end="")
    for h in range(num_heads):
        print(f"  {'头'+str(h):>6}", end="")
    print()
    
    avg_diversity = 0
    count = 0
    for i in range(num_heads):
        print(f"  {'头'+str(i):>6}", end="")
        for j in range(num_heads):
            cos_sim = F.cosine_similarity(
                head_vectors[i].unsqueeze(0), 
                head_vectors[j].unsqueeze(0)
            ).item()
            print(f"  {cos_sim:>+6.3f}", end="")
            if i != j:
                avg_diversity += cos_sim
                count += 1
        print()
    
    avg_diversity /= count
    print(f"\n  平均头间余弦相似度: {avg_diversity:.4f}")
    print(f"  (1.0 = 完全相同，0.0 = 完全正交/不同)")
    if avg_diversity < 0.9:
        print(f"  ✓ 多头学到了不同的注意力模式！多样性良好。")
    else:
        print(f"  ⚠ 注意力头之间比较相似，可能需要更多训练。")
    
    # 计算单头注意力的"信息熵"
    single_entropy = -(single_attn * (single_attn + 1e-10).log()).sum(dim=-1).mean()
    
    # 多头各头的平均信息熵
    multi_entropies = -(multi_attn * (multi_attn + 1e-10).log()).sum(dim=-1).mean(dim=-1)
    
    print(f"\n  单头注意力平均熵: {single_entropy.item():.4f}")
    print(f"  多头各头注意力熵:")
    for h in range(num_heads):
        print(f"    头 {h}: {multi_entropies[h].item():.4f}")
    print(f"  (熵越高 = 注意力越分散/均匀；熵越低 = 注意力越集中)")


def demo_parameter_count():
    """演示：多头 vs 单头的参数量对比"""
    print("\n" + "=" * 70)
    print("  实验4：参数量与计算量对比")
    print("=" * 70)
    
    configs = [
        (64, 1, "单头 (d=64)"),
        (64, 4, "4头 (d=64, d_k=16)"),
        (64, 8, "8头 (d=64, d_k=8)"),
        (512, 1, "单头 (d=512)"),
        (512, 8, "8头 (d=512, d_k=64)"),
        (768, 12, "12头 (d=768, d_k=64) — BERT-Base"),
    ]
    
    print(f"\n  {'配置':>30} | {'参数量':>12} | {'每头维度':>8}")
    print(f"  {'-'*60}")
    
    for d_model, num_heads, name in configs:
        model = MultiHeadAttention(d_model, num_heads)
        total_params = sum(p.numel() for p in model.parameters())
        d_k = d_model // num_heads
        print(f"  {name:>30} | {total_params:>12,d} | d_k={d_k}")
    
    print(f"\n  💡 关键发现：")
    print(f"  - d_model 相同时，不管用多少个头，参数量一样！")
    print(f"  - 因为 W_Q/W_K/W_V/W_O 都是 [d_model, d_model]")
    print(f"  - 多头只是改变了 attention 的计算方式，不增加参数")


def demo_masked_attention():
    """演示：带 Causal Mask 的多头注意力"""
    print("\n" + "=" * 70)
    print("  实验5：Causal Mask — 自回归生成中的多头注意力")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    d_model = 32
    num_heads = 4
    seq_len = 5
    tokens = ["我", "爱", "大", "语言", "模型"]
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(1, seq_len, d_model)
    
    # 创建 Causal Mask（下三角矩阵）
    # 1 = 可以看，0 = 不能看
    mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
    
    print(f"\n  句子: {' '.join(tokens)}")
    print(f"\n  Causal Mask (1=可见, 0=遮蔽):")
    print(f"  {'':>6}", end="")
    for t in tokens:
        print(f"  {t:>4}", end="")
    print()
    for i, t in enumerate(tokens):
        print(f"  {t:>6}", end="")
        for j in range(seq_len):
            val = int(mask[0, 0, i, j].item())
            print(f"    {val}", end="")
        print()
    
    output, attn_weights = mha(x, x, x, mask=mask, return_attn=True)
    
    print(f"\n  应用 Causal Mask 后的注意力（头 0）:")
    attn_np = attn_weights[0, 0].detach().numpy()
    
    print(f"  {'':>6}", end="")
    for t in tokens:
        print(f"  {t:>6}", end="")
    print()
    for i, t in enumerate(tokens):
        print(f"  {t:>6}", end="")
        for j in range(seq_len):
            val = attn_np[i, j]
            if val < 1e-6:
                symbol = "  -   "
            elif val > 0.3:
                symbol = f" {val:.3f}*"
            else:
                symbol = f" {val:.3f} "
            print(symbol, end="")
        print()
    
    print(f"\n  ✓ 注意：下三角（未来位置）的注意力权重为 0（或接近 0）")
    print(f"  这确保了模型生成时只能看到前面的词，不能偷看后面的！")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          🧠 Day 7: Multi-Head Attention 多头注意力            ║")
    print("║          从零理解 LLM 课程 — 手写代码系列                      ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()
    
    # 实验1：基本演示
    demo_multi_head_attention()
    
    # 实验2：头的多样性
    demo_head_diversity()
    
    # 实验3：单头 vs 多头对比
    demo_single_vs_multi_head()
    
    # 实验4：参数量对比
    demo_parameter_count()
    
    # 实验5：Causal Mask
    demo_masked_attention()
    
    print("\n" + "=" * 70)
    print("  🎉 Day 7 完成！")
    print("=" * 70)
    print()
    print("  今天我们学到了：")
    print("  1. Multi-Head Attention 把注意力空间切成 h 个子空间")
    print("  2. 每个头独立计算 Attention，最后拼接 + 线性投影")
    print("  3. 不同头自动学到不同的关注模式（语法、语义、位置...）")
    print("  4. 多头不增加计算量（参数量也相同），但大大提升表达能力")
    print("  5. Causal Mask 确保自回归生成时只能看到前面的词")
    print()
    print("  明天预告：Day 8 — 位置编码")
    print("  Transformer 的注意力本身没有位置概念，那它怎么知道词的顺序？")
    print()
