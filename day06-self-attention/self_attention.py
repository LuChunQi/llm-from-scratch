#!/usr/bin/env python3
"""
Day 6: Self-Attention — 让每个词"看见"所有其他词

包含两个实现：
1. 纯 NumPy 版：手动实现每一步，看清数学细节
2. PyTorch 版：实际工程中推荐使用的写法

还会可视化注意力矩阵，直观理解"谁关注谁"
"""

import numpy as np

print("=" * 70)
print("🧠 Day 6: Self-Attention — 自注意力机制")
print("=" * 70)

# ============================================================
# 第一部分：纯 NumPy 实现（看清楚每一步）
# ============================================================
print("\n" + "─" * 70)
print("📌 第一部分：纯 NumPy 实现")
print("─" * 70)

def softmax_np(x, axis=-1):
    """手动实现 softmax（数值稳定版）
    
    数值稳定技巧：减去最大值，防止 exp 溢出
    就像考试排名——不用知道绝对分数，只要知道相对差距
    """
    # 减去每行的最大值，防止 e^x 溢出
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def self_attention_numpy(X, W_Q, W_K, W_V, verbose=False):
    """纯 NumPy 实现的 Self-Attention
    
    参数：
        X: 输入序列，形状 [seq_len, d_model]
        W_Q: Query 权重矩阵，形状 [d_model, d_k]
        W_K: Key 权重矩阵，形状 [d_model, d_k]
        W_V: Value 权重矩阵，形状 [d_model, d_v]
        verbose: 是否打印详细过程（默认 False）
    
    返回：
        output: 注意力输出，形状 [seq_len, d_v]
        attn_weights: 注意力权重矩阵，形状 [seq_len, seq_len]
    """
    seq_len, d_model = X.shape
    d_k = W_Q.shape[1]  # Key/Query 的维度
    
    # ──────────────────────────────────────────────
    # 步骤 1: 线性变换，生成 Q、K、V
    # 每个词向量分别乘以三个权重矩阵，得到"提问向量"、"标签向量"、"内容向量"
    # ──────────────────────────────────────────────
    Q = X @ W_Q  # [seq_len, d_k]  — "我想知道什么"
    K = X @ W_K  # [seq_len, d_k]  — "我能提供什么线索"
    V = X @ W_V  # [seq_len, d_v]  — "我的实际内容"
    
    if verbose:
        print(f"\n  Q 形状: {Q.shape}  (每个词的'查询'向量)")
        print(f"  K 形状: {K.shape}  (每个词的'键'向量)")
        print(f"  V 形状: {V.shape}  (每个词的'值'向量)")
    
    # ──────────────────────────────────────────────
    # 步骤 2: 计算注意力分数（点积）
    # Q · K^T = 每个词的"查询"和所有词的"键"的点积
    # 点积越大 → 两个向量越"相似" → 关注度越高
    # ──────────────────────────────────────────────
    scores = Q @ K.T  # [seq_len, seq_len]
    if verbose:
        print(f"\n  原始分数矩阵形状: {scores.shape}")
        print(f"  scores[i][j] = 第 i 个词对第 j 个词的'关注度原始分数'")
    
    # ──────────────────────────────────────────────
    # 步骤 3: 缩放（Scale）
    # 除以 sqrt(d_k)，防止分数太大导致 softmax 饱和
    # 类比：1000 个人投票 vs 10 个人投票，总分差距很大，需要归一化
    # ──────────────────────────────────────────────
    scaled_scores = scores / np.sqrt(d_k)
    if verbose:
        print(f"\n  缩放因子 √d_k = √{d_k} = {np.sqrt(d_k):.4f}")
        print(f"  缩放后分数范围: [{scaled_scores.min():.4f}, {scaled_scores.max():.4f}]")
    
    # ──────────────────────────────────────────────
    # 步骤 4: Softmax 归一化
    # 把分数变成概率分布（每行之和 = 1）
    # 每一行表示一个词对所有词的"注意力分配比例"
    # ──────────────────────────────────────────────
    attn_weights = softmax_np(scaled_scores, axis=-1)  # [seq_len, seq_len]
    
    # ──────────────────────────────────────────────
    # 步骤 5: 加权求和
    # 用注意力权重对 V 做加权平均
    # 高关注度的词贡献大，低关注度的词贡献小
    # ──────────────────────────────────────────────
    output = attn_weights @ V  # [seq_len, d_v]
    
    return output, attn_weights


# ──────────────────────────────────────────────────
# 演示：用一个小句子来测试
# ──────────────────────────────────────────────────
print("\n" + "🎬 " + "=" * 66)
print("🎬 示例：用 Self-Attention 处理一个中文句子")
print("🎬 " + "=" * 66)

# 假设句子是 "小明 给 小红 一本书"
# 每个"词"用一个 4 维向量表示（实际中通常是 512 维或 768 维）
np.random.seed(42)  # 固定随机种子，确保可复现

words = ["小明", "给", "小红", "一本书"]
seq_len = len(words)
d_model = 4  # 嵌入维度（实际中 512/768）
d_k = 4      # Query/Key 维度
d_v = 4      # Value 维度

# 模拟词嵌入（实际中由 Embedding 层学习得到）
X = np.random.randn(seq_len, d_model).astype(np.float32)

print(f"\n  句子: {' '.join(words)}")
print(f"  序列长度: {seq_len}, 嵌入维度: {d_model}")
print(f"\n  输入矩阵 X (每个词的向量表示):")
for i, word in enumerate(words):
    print(f"    {word}: {X[i]}")

# 随机初始化权重（实际中通过训练学习）
W_Q = np.random.randn(d_model, d_k).astype(np.float32) * 0.5
W_K = np.random.randn(d_model, d_k).astype(np.float32) * 0.5
W_V = np.random.randn(d_model, d_v).astype(np.float32) * 0.5

# 运行 Self-Attention（开启详细输出）
output, attn_weights = self_attention_numpy(X, W_Q, W_K, W_V, verbose=True)

print(f"\n  输出矩阵形状: {output.shape}")
print(f"\n  📊 注意力权重矩阵（谁关注谁）:")
print(f"  {'':>8s}", end="")
for w in words:
    print(f"  {w:>6s}", end="")
print()
for i, word in enumerate(words):
    print(f"  {word:>6s}", end="")
    for j in range(seq_len):
        print(f"  {attn_weights[i][j]:6.3f}", end="")
    print()

# 验证：每一行的注意力权重之和 = 1（softmax 保证）
print(f"\n  ✅ 验证 softmax：每行之和 = {[f'{s:.4f}' for s in attn_weights.sum(axis=1)]}")

print(f"\n  输出向量（经过注意力加权后的新表示）:")
for i, word in enumerate(words):
    print(f"    {word}: {[f'{v:.4f}' for v in output[i]]}")


# ============================================================
# 第二部分：为什么需要缩放？（缩放的直观演示）
# ============================================================
print("\n\n" + "─" * 70)
print("📌 第二部分：为什么需要缩放因子 1/√d_k？")
print("─" * 70)

# 当 d_k 很大时，点积的方差也会增大
# 导致 softmax 输出接近 one-hot → 梯度消失
for d_k_test in [4, 16, 64, 256, 1024]:
    np.random.seed(42)
    q = np.random.randn(d_k_test).astype(np.float32)
    k = np.random.randn(d_k_test).astype(np.float32)
    
    dot_product = np.dot(q, k)
    scaled = dot_product / np.sqrt(d_k_test)
    
    # 模拟多个 query-key 对的分数
    scores = np.random.randn(10).astype(np.float32) * np.sqrt(d_k_test)
    probs = softmax_np(scores)
    max_prob = probs.max()
    
    print(f"  d_k = {d_k_test:>4d}: 点积 = {dot_product:>8.2f}, "
          f"缩放后 = {scaled:>8.2f}, "
          f"softmax 最大概率 = {max_prob:.4f} "
          f"{'⚠️ 接近 one-hot!' if max_prob > 0.9 else '✅'}")

print(f"\n  💡 结论：d_k 越大，缩放越重要！")
print(f"     没有缩放 → softmax 饱和 → 梯度消失 → 训练困难")


# ============================================================
# 第三部分：PyTorch 实现（实际工程写法）
# ============================================================
print("\n\n" + "─" * 70)
print("📌 第三部分：PyTorch 实现（工程推荐写法）")
print("─" * 70)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("  ⚠️ PyTorch 未安装，跳过 PyTorch 演示")
    print("  安装方法：pip install torch")

if TORCH_AVAILABLE:
    
    class SelfAttention(nn.Module):
        """PyTorch 实现的 Self-Attention 层
        
        这是实际工程中推荐使用的写法：
        - 用 nn.Linear 代替手动矩阵乘法（自动管理参数）
        - 用 F.softmax 代替手写 softmax（数值稳定）
        - 支持 batch 维度
        
        参数：
            d_model: 模型维度（输入和输出的向量维度）
            d_k: Query/Key 的维度（默认等于 d_model）
        """
        
        def __init__(self, d_model, d_k=None):
            super().__init__()
            self.d_k = d_k or d_model
            
            # 三个线性变换层，分别生成 Q、K、V
            # nn.Linear 包含可训练的权重矩阵 W 和偏置 b
            self.W_Q = nn.Linear(d_model, self.d_k, bias=False)
            self.W_K = nn.Linear(d_model, self.d_k, bias=False)
            self.W_V = nn.Linear(d_model, self.d_k, bias=False)
        
        def forward(self, x):
            """
            参数：
                x: 输入张量，形状 [batch_size, seq_len, d_model]
                   （支持 batch 维度，这是 NumPy 版没有的）
            
            返回：
                output: 注意力输出 [batch_size, seq_len, d_k]
                attn_weights: 注意力权重 [batch_size, seq_len, seq_len]
            """
            batch_size, seq_len, d_model = x.shape
            
            # 步骤 1: 生成 Q、K、V
            Q = self.W_Q(x)  # [batch, seq_len, d_k]
            K = self.W_K(x)  # [batch, seq_len, d_k]
            V = self.W_V(x)  # [batch, seq_len, d_k]
            
            # 步骤 2 & 3: 计算缩放点积注意力分数
            # transpose(-2, -1) 交换最后两个维度，相当于转置 K
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
            
            # 步骤 4: Softmax 归一化
            # dim=-1 表示在最后一个维度（seq_len）上做 softmax
            attn_weights = F.softmax(scores, dim=-1)
            
            # 步骤 5: 加权求和
            output = torch.matmul(attn_weights, V)
            
            return output, attn_weights
    
    
    # ──────────────────────────────────────────────
    # 演示 PyTorch 版本
    # ──────────────────────────────────────────────
    print("\n  🎬 PyTorch Self-Attention 演示:")
    
    # 创建 Self-Attention 层
    d_model = 8
    attn_layer = SelfAttention(d_model=d_model)
    
    # 模拟输入：batch_size=1, seq_len=4, d_model=8
    x_torch = torch.randn(1, 4, d_model)
    
    output_torch, attn_torch = attn_layer(x_torch)
    
    print(f"\n  输入形状: {x_torch.shape}")
    print(f"  输出形状: {output_torch.shape}")
    print(f"  注意力权重形状: {attn_torch.shape}")
    
    print(f"\n  📊 PyTorch 注意力权重:")
    attn_np = attn_torch[0].detach().numpy()
    print(f"  {'':>8s}", end="")
    for w in words:
        print(f"  {'词'+str(words.index(w)+1):>6s}", end="")
    print()
    for i in range(4):
        print(f"  {'词'+str(i+1):>6s}", end="")
        for j in range(4):
            print(f"  {attn_np[i][j]:6.3f}", end="")
        print()
    
    # 参数量统计
    total_params = sum(p.numel() for p in attn_layer.parameters())
    print(f"\n  📊 模型参数量: {total_params} (3 个 {d_model}×{d_model} 的权重矩阵)")
    
    
    # ============================================================
    # 第四部分：PyTorch vs NumPy 对比验证
    # ============================================================
    print("\n\n" + "─" * 70)
    print("📌 第四部分：NumPy vs PyTorch 结果一致性验证")
    print("─" * 70)
    
    # 用相同的权重和输入
    np.random.seed(42)
    torch.manual_seed(42)
    
    d_test = 4
    seq_test = 3
    
    # NumPy 版
    X_np = np.random.randn(seq_test, d_test).astype(np.float32)
    WQ_np = np.random.randn(d_test, d_test).astype(np.float32) * 0.5
    WK_np = np.random.randn(d_test, d_test).astype(np.float32) * 0.5
    WV_np = np.random.randn(d_test, d_test).astype(np.float32) * 0.5
    
    out_np, attn_np = self_attention_numpy(X_np, WQ_np, WK_np, WV_np, verbose=False)
    
    # PyTorch 版（用相同的权重）
    attn_t = SelfAttention(d_test)
    with torch.no_grad():
        attn_t.W_Q.weight = nn.Parameter(torch.tensor(WQ_np.T))
        attn_t.W_K.weight = nn.Parameter(torch.tensor(WK_np.T))
        attn_t.W_V.weight = nn.Parameter(torch.tensor(WV_np.T))
        
        X_t = torch.tensor(X_np).unsqueeze(0)  # 添加 batch 维度
        out_t, attn_t_w = attn_t(X_t)
    
    # 比较
    diff_output = np.abs(out_np - out_t[0].numpy()).max()
    diff_attn = np.abs(attn_np - attn_t_w[0].numpy()).max()
    
    print(f"\n  输出最大差异: {diff_output:.2e}")
    print(f"  注意力权重最大差异: {diff_attn:.2e}")
    
    if diff_output < 1e-5 and diff_attn < 1e-5:
        print(f"  ✅ NumPy 和 PyTorch 结果一致！")
    else:
        print(f"  ⚠️ 存在微小浮点差异（正常现象）")


# ============================================================
# 第五部分：Self-Attention 的直觉演示
# ============================================================
print("\n\n" + "─" * 70)
print("📌 第五部分：自注意力的直觉演示 — 指代消解")
print("─" * 70)

# 模拟一个有趣的场景：
# "猫 追 老鼠 它 跑 了"  →  "它"应该关注"猫"还是"老鼠"？
print("""
  句子: "猫 追 老鼠 它 跑 了"
  
  如果"它"指"猫"（猫跑了）：
    → "它"应该高关注"猫"
  
  如果"它"指"老鼠"（老鼠跑了）：
    → "它"应该高关注"老鼠"
  
  Self-Attention 让模型能根据上下文自动学习这种关联！
""")

# 用精心设计的向量来演示
# 让"跑"的向量更接近"猫"的向量（暗示"它"="猫"）
np.random.seed(123)
referral_words = ["猫", "追", "老鼠", "它", "跑", "了"]
ref_seq_len = len(referral_words)
ref_d = 8

# 基础嵌入
X_ref = np.random.randn(ref_seq_len, ref_d).astype(np.float32)

# 让"它"（index 3）的向量更接近"猫"（index 0）的向量
# 这样注意力分数中 "它"→"猫" 会更高
X_ref[3] = X_ref[0] * 0.6 + X_ref[2] * 0.2 + np.random.randn(ref_d).astype(np.float32) * 0.3

WQ_ref = np.random.randn(ref_d, ref_d).astype(np.float32) * 0.3
WK_ref = np.random.randn(ref_d, ref_d).astype(np.float32) * 0.3
WV_ref = np.random.randn(ref_d, ref_d).astype(np.float32) * 0.3

out_ref, attn_ref = self_attention_numpy(X_ref, WQ_ref, WK_ref, WV_ref, verbose=True)

print(f"  📊 '它' 对各词的注意力权重:")
print(f"  {'词':>6s}  {'注意力':>8s}  {'可视化':>30s}")
print(f"  {'─'*6}  {'─'*8}  {'─'*30}")
for j, word in enumerate(referral_words):
    weight = attn_ref[3][j]  # "它"（第3个词）对所有词的关注度
    bar = "█" * int(weight * 50)
    marker = " 👈" if word == "猫" else "" 
    print(f"  {word:>6s}  {weight:>8.4f}  {bar}{marker}")

# 找到"它"最关注的词
most_attended_idx = np.argmax(attn_ref[3])
print(f"\n  🎯 '它'最关注的词: '{referral_words[most_attended_idx]}' "
      f"(注意力 = {attn_ref[3][most_attended_idx]:.4f})")


# ============================================================
# 第六部分：复杂度分析
# ============================================================
print("\n\n" + "─" * 70)
print("📌 第六部分：计算复杂度分析")
print("─" * 70)

import time

print(f"\n  Self-Attention 的时间复杂度: O(n² · d)")
print(f"  其中 n = 序列长度, d = 向量维度\n")
print(f"  {'序列长度':>10s}  {'矩阵大小':>12s}  {'计算时间':>12s}")
print(f"  {'─'*10}  {'─'*12}  {'─'*12}")

for n in [32, 64, 128, 256, 512, 1024]:
    d = 64
    X_bench = np.random.randn(n, d).astype(np.float32)
    WQ_bench = np.random.randn(d, d).astype(np.float32) * 0.3
    WK_bench = np.random.randn(d, d).astype(np.float32) * 0.3
    WV_bench = np.random.randn(d, d).astype(np.float32) * 0.3
    
    # 计时（取多次平均）
    times = []
    for _ in range(5):
        start = time.time()
        _ = self_attention_numpy(X_bench, WQ_bench, WK_bench, WV_bench, verbose=False)
        times.append(time.time() - start)
    
    avg_time = np.mean(times) * 1000  # 转毫秒
    mat_size = n * n
    
    print(f"  {n:>10d}  {mat_size:>12,d}  {avg_time:>10.2f}ms")

print(f"\n  💡 注意：序列长度翻倍 → 计算量约 4 倍（n² 增长）")
print(f"     这就是为什么长文本处理需要 Flash Attention 等优化！")


# ============================================================
# 总结
# ============================================================
print("\n\n" + "=" * 70)
print("🎓 总结")
print("=" * 70)
print("""
  Self-Attention 的核心公式：
  
    Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
  
  五步流程：
    1️⃣  线性变换：X → Q, K, V（三个不同的"视角"）
    2️⃣  计算分数：Q · Kᵀ（匹配程度）
    3️⃣  缩放：÷ √dₖ（防止 softmax 饱和）
    4️⃣  归一化：softmax（变成概率分布）
    5️⃣  加权求和：权重 × V（融合信息）
  
  关键特性：
    ✅ 全局感受野 — 每个位置看到所有位置
    ✅ 动态权重 — 根据输入内容自适应
    ✅ 可并行 — 矩阵运算，不需要串行
    ⚠️ O(n²) 复杂度 — 长序列是瓶颈
  
  下一步：Day 7 — Multi-Head Attention（多头注意力）
    为什么一个头不够？多个头各看什么？
""")
