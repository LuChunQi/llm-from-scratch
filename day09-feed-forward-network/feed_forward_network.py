#!/usr/bin/env python3
"""
Day 9: Feed-Forward Network — Transformer 的"消化系统"
========================================================

本代码包含四个实验：
1. 基础 FFN 实现 — 对比有无激活函数的差异
2. 激活函数全家福 — 可视化 ReLU / GELU / Swish / SwiGLU
3. "键值记忆"实验 — 验证 FFN 的键值记忆特性
4. FFN 参数量分析 — FFN 在 Transformer 中占多少参数

运行方式：python3 feed_forward_network.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ============================================================
# 实验 1：基础 FFN 实现
# ============================================================
print("=" * 60)
print("实验 1：基础 FFN — 有无激活函数的对比")
print("=" * 60)


class SimpleFFN(nn.Module):
    """标准 FFN：两层线性变换 + 激活函数"""

    def __init__(self, d_model, d_ff, activation="relu"):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)    # 升维：d_model → d_ff
        self.w2 = nn.Linear(d_ff, d_model)    # 降维：d_ff → d_model
        self.activation = activation

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        h = self.w1(x)                        # 升维到 d_ff
        if self.activation == "relu":
            h = F.relu(h)                     # ReLU 激活
        elif self.activation == "gelu":
            h = F.gelu(h)                     # GELU 激活
        elif self.activation == "none":
            pass                              # 不加激活函数
        out = self.w2(h)                      # 降维回 d_model
        return out


# 创建一个小 FFN
d_model = 8    # 嵌入维度
d_ff = 32      # FFN 中间维度（4倍）

torch.manual_seed(42)
ffn_with_relu = SimpleFFN(d_model, d_ff, activation="relu")
ffn_with_gelu = SimpleFFN(d_model, d_ff, activation="gelu")
ffn_no_activation = SimpleFFN(d_model, d_ff, activation="none")

# 让三个 FFN 使用相同的权重，只对比激活函数的效果
ffn_no_activation.load_state_dict(ffn_with_relu.state_dict())
ffn_with_gelu.load_state_dict(ffn_with_relu.state_dict())

# 创建输入（模拟一个词的向量表示）
x = torch.randn(1, 1, d_model)
print(f"输入向量: {x[0, 0, :4].detach().numpy()} ... (前4维)")

# 分别通过三个 FFN
out_relu = ffn_with_relu(x)
out_gelu = ffn_with_gelu(x)
out_none = ffn_no_activation(x)

print(f"\n带 ReLU 的输出:  {out_relu[0, 0, :4].detach().numpy()} ...")
print(f"带 GELU 的输出:  {out_gelu[0, 0, :4].detach().numpy()} ...")
print(f"无激活函数输出:  {out_none[0, 0, :4].detach().numpy()} ...")

# 演示：没有激活函数的两层线性变换等价于一层线性变换
# 数学证明：W2 · (W1 · x) = (W2 · W1) · x = W' · x
W1 = ffn_no_activation.w1.weight.data  # (d_ff, d_model)
W2 = ffn_no_activation.w2.weight.data  # (d_model, d_ff)
b1 = ffn_no_activation.w1.bias.data    # (d_ff,)
b2 = ffn_no_activation.w2.bias.data    # (d_model,)

# 合并两层为一个矩阵
W_combined = W2 @ W1                    # (d_model, d_model)
b_combined = W2 @ b1 + b2              # (d_model,)

# 用合并后的矩阵做前向传播
x_flat = x[0, 0]                       # (d_model,)
out_combined = W_combined @ x_flat + b_combined

print(f"\n两层线性变换结果:      {out_none[0, 0, :4].detach().numpy()} ...")
print(f"合并为一层的结果:      {out_combined[:4].numpy()} ...")
print(f"两者是否相同?         {torch.allclose(out_none[0, 0], out_combined, atol=1e-5)}")
print("\n💡 结论：没有激活函数，两层线性变换等价于一层！多出来的层毫无意义。")


# ============================================================
# 实验 2：激活函数全家福
# ============================================================
print("\n" + "=" * 60)
print("实验 2：激活函数全家福 — ReLU / GELU / Swish / SwiGLU")
print("=" * 60)

# 生成输入范围
x_range = np.linspace(-4, 4, 200)
x_tensor = torch.tensor(x_range, dtype=torch.float32)

# --- ReLU ---
relu_out = F.relu(x_tensor).numpy()

# --- GELU (精确版) ---
def gelu_exact(x):
    """精确 GELU: x * Φ(x)，其中 Φ 是标准正态分布的 CDF"""
    return x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))

gelu_out = gelu_exact(x_tensor).numpy()

# --- GELU (近似版，用 sigmoid) ---
def gelu_approx(x):
    """近似 GELU: x * sigmoid(1.702 * x)"""
    return x * torch.sigmoid(1.702 * x)

gelu_approx_out = gelu_approx(x_tensor).numpy()

# --- Swish ---
def swish(x, beta=1.0):
    """Swish 激活函数: x * sigmoid(β * x)"""
    return x * torch.sigmoid(torch.tensor(beta) * x)

swish_out = swish(x_tensor).numpy()

# --- SwiGLU ---
def swiglu(x):
    """SwiGLU: (x * V + c) ⊙ swish(x * W + b)
    简化演示：用两个不同的线性变换 + 门控
    """
    # 模拟两个分支
    gate_branch = swish(x_tensor)     # 门控分支：决定"保留多少"
    value_branch = x_tensor           # 值分支：要保留的内容
    return gate_branch * value_branch

swiglu_out = swiglu(x_tensor).numpy()

# 打印关键位置的值，做直观对比
sample_points = [-3, -1, 0, 1, 3]
print(f"\n{'x':>6} | {'ReLU':>8} | {'GELU':>8} | {'GELU*':>8} | {'Swish':>8} | {'SwiGLU':>8}")
print("-" * 70)
for xp in sample_points:
    xt = torch.tensor(xp, dtype=torch.float32)
    r = F.relu(xt).item()
    g = gelu_exact(xt).item()
    ga = gelu_approx(xt).item()
    s = swish(xt).item()
    sg = (swish(xt) * xt).item()
    print(f"{xp:>6.1f} | {r:>8.4f} | {g:>8.4f} | {ga:>8.4f} | {s:>8.4f} | {sg:>8.4f}")

print("\n💡 观察要点：")
print("   - ReLU: 负数直接归零，在 x=0 处有硬拐点")
print("   - GELU: 负数平滑趋近于零（不是硬切），曲线更圆滑")
print('   - Swish: 在负数区有小幅"鼓包"，不完全为零')
print('   - SwiGLU: 门控效果，正负区间都有自适应调节')


# ============================================================
# 实验 3："键值记忆"实验
# ============================================================
print("\n" + "=" * 60)
print("实验 3：FFN 的键值记忆特性")
print("=" * 60)

print("""
理论：FFN 的 W₁ 充当"键"（Key），W₂ 充当"值"（Value）
输入 x 和 W₁ 的某一行越相似 → 该行的"值"（W₂ 对应的列）越被激活
""")

# 构造一个简单的例子
# 假设词汇表只有4个概念：法国、中国、苹果、香蕉
# 用 one-hot 编码
torch.manual_seed(42)

d_model_kv = 4
d_ff_kv = 8

# 手动构造 W1 和 W2，模拟"键值记忆"
# W1 的每一行是一个"键"——代表一种模式
W1 = torch.zeros(d_ff_kv, d_model_kv)
# 键 0: 匹配"法国" [1, 0, 0, 0]
W1[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
# 键 1: 匹配"中国" [0, 1, 0, 0]
W1[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
# 键 2: 匹配"苹果" [0, 0, 1, 0]
W1[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
# 键 3: 匹配"香蕉" [0, 0, 0, 1]
W1[3] = torch.tensor([0.0, 0.0, 0.0, 1.0])
# 键 4-7: 其他模式（留空/随机）
W1[4:] = torch.randn(4, d_model_kv) * 0.1  # 弱随机模式

# W2 的每一列是"值"——当对应键被激活时输出的内容
W2 = torch.zeros(d_model_kv, d_ff_kv)
# 当"法国"键被激活 → 输出"巴黎"的概念
W2[0, 0] = 5.0   # "巴黎"维度，由"法国"键触发
W2[1, 0] = 0.0
# 当"中国"键被激活 → 输出"北京"
W2[0, 1] = 0.0
W2[1, 1] = 5.0   # "北京"维度
# 当"苹果"键被激活 → 输出"水果"
W2[2, 2] = 4.0   # "水果"维度
# 当"香蕉"键被激活 → 输出"水果"
W2[2, 3] = 4.0   # "水果"维度（和苹果共享！）

# 模拟 FFN 前向传播
def kv_ffn(x, W1, W2):
    """手动模拟 FFN，展示键值匹配过程"""
    # Step 1: 计算匹配分数（输入 x 和每个键的内积）
    scores = x @ W1.T  # (d_model,) @ (d_model, d_ff) → (d_ff,)
    
    # Step 2: ReLU 激活（过滤负分数）
    activated = F.relu(scores)
    
    # Step 3: 加权求和"值"
    output = W2 @ activated  # (d_model, d_ff) @ (d_ff,) → (d_model,)
    
    return scores, activated, output


# 测试：输入"法国"
x_france = torch.tensor([1.0, 0.0, 0.0, 0.0])
scores, activated, output = kv_ffn(x_france, W1, W2)
print(f"输入: '法国' [1, 0, 0, 0]")
print(f"  键匹配分数: {scores.numpy()}")
print(f"  ReLU 后:    {activated.numpy()}")
print(f"  输出:       {output.numpy()}")
print(f"  → 输出最强维度: 维度 0 (值={output[0]:.1f})，代表'巴黎' ✅")

# 测试：输入"中国"
x_china = torch.tensor([0.0, 1.0, 0.0, 0.0])
scores, activated, output = kv_ffn(x_china, W1, W2)
print(f"\n输入: '中国' [0, 1, 0, 0]")
print(f"  键匹配分数: {scores.numpy()}")
print(f"  ReLU 后:    {activated.numpy()}")
print(f"  输出:       {output.numpy()}")
print(f"  → 输出最强维度: 维度 1 (值={output[1]:.1f})，代表'北京' ✅")

# 测试：输入"苹果"
x_apple = torch.tensor([0.0, 0.0, 1.0, 0.0])
scores, activated, output = kv_ffn(x_apple, W1, W2)
print(f"\n输入: '苹果' [0, 0, 1, 0]")
print(f"  键匹配分数: {scores.numpy()}")
print(f"  ReLU 后:    {activated.numpy()}")
print(f"  输出:       {output.numpy()}")
print(f"  → 输出最强维度: 维度 2 (值={output[2]:.1f})，代表'水果' ✅")

# 测试：输入"香蕉"
x_banana = torch.tensor([0.0, 0.0, 0.0, 1.0])
scores, activated, output = kv_ffn(x_banana, W1, W2)
print(f"\n输入: '香蕉' [0, 0, 0, 1]")
print(f"  键匹配分数: {scores.numpy()}")
print(f"  ReLU 后:    {activated.numpy()}")
print(f"  输出:       {output.numpy()}")
print(f"  → 输出最强维度: 维度 2 (值={output[2]:.1f})，和'苹果'一样是'水果' ✅")

print("\n💡 苹果和香蕉都激活了'水果'这个维度——FFN 学到了'水果'这个概念！")
print("   这就是 FFN 作为'键值记忆'的核心：输入匹配键 → 输出对应的值（知识）")


# ============================================================
# 实验 4：FFN 参数量分析
# ============================================================
print("\n" + "=" * 60)
print("实验 4：FFN 在 Transformer 中的参数占比")
print("=" * 60)


class TransformerBlock(nn.Module):
    """一个简化的 Transformer Block，用于分析参数量"""

    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        # --- Attention 部分 ---
        self.q_proj = nn.Linear(d_model, d_model)  # Q 投影
        self.k_proj = nn.Linear(d_model, d_model)  # K 投影
        self.v_proj = nn.Linear(d_model, d_model)  # V 投影
        self.out_proj = nn.Linear(d_model, d_model) # 输出投影
        # 注意：实际实现中，QKV 通常合并为一个矩阵

        # --- FFN 部分 ---
        self.ffn_w1 = nn.Linear(d_model, d_ff)     # 升维
        self.ffn_w2 = nn.Linear(d_ff, d_model)     # 降维

    def count_parameters(self, module_name):
        """计算某部分的参数量"""
        total = 0
        for name, param in self.named_parameters():
            if module_name in name:
                total += param.numel()
        return total


def count_params(model, prefix=None):
    """统计模块的参数量"""
    total = 0
    for name, param in model.named_parameters():
        if prefix is None or prefix in name:
            total += param.numel()
    return total


# 模拟不同规模的 Transformer Block
configs = [
    ("Tiny (d=128)",  128,  4,   512),
    ("Small (d=512)", 512,  8,  2048),
    ("Base (d=768)",  768, 12,  3072),
    ("Large (d=1024)", 1024, 16, 4096),
]

print(f"\n{'配置':<20} | {'Attention参数':>12} | {'FFN参数':>10} | {'总参数':>10} | {'FFN占比':>8}")
print("-" * 75)

for name, d_model, n_heads, d_ff in configs:
    block = TransformerBlock(d_model, n_heads, d_ff)

    # Attention 参数：Q, K, V, Out 四个投影矩阵（各有权重 + 偏置）
    attn_params = count_params(block, "proj")
    # Q/K/V/Out 都有 "proj" 在名字里
    attn_params = (count_params(block, "q_proj") +
                   count_params(block, "k_proj") +
                   count_params(block, "v_proj") +
                   count_params(block, "out_proj"))

    # FFN 参数：W1 和 W2（各有权重 + 偏置）
    ffn_params = count_params(block, "ffn")

    total = attn_params + ffn_params
    ffn_ratio = ffn_params / total * 100

    print(f"{name:<20} | {attn_params:>12,} | {ffn_params:>10,} | {total:>10,} | {ffn_ratio:>6.1f}%")

print("\n💡 关键发现：")
print("   - FFN 的参数量始终约为 Attention 的 2 倍")
print("   - FFN 占了 Transformer Block 约 2/3 的参数！")
print("   - 所以说'FFN 是 Transformer 的知识库'——大部分参数都在这里")


# ============================================================
# 实验 5：实际 SwiGLU FFN 实现（现代 LLM 风格）
# ============================================================
print("\n" + "=" * 60)
print("实验 5：SwiGLU FFN — 现代 LLM 的标配")
print("=" * 60)


class SwiGLUFFN(nn.Module):
    """SwiGLU FFN，LLaMA/Qwen/DeepSeek 等现代模型使用"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)   # 门控分支（gate）
        self.w2 = nn.Linear(d_ff, d_model)   # 输出投影
        self.v = nn.Linear(d_model, d_ff)    # 值分支（value）
        # 注意：比标准 FFN 多了一个 V 矩阵，参数量多了 50%

    def forward(self, x):
        # swish(x @ W1) ⊙ (x @ V) → 再 @ W2
        gate = F.silu(self.w1(x))            # silu = swish
        value = self.v(x)
        return self.w2(gate * value)


class StandardFFN(nn.Module):
    """标准 FFN（GELU 版），GPT/BERT 使用"""

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


# 对比两种 FFN
d_model_demo = 64
d_ff_demo = 256

torch.manual_seed(42)
std_ffn = StandardFFN(d_model_demo, d_ff_demo)
swiglu_ffn = SwiGLUFFN(d_model_demo, d_ff_demo)

# 统计参数量
std_params = sum(p.numel() for p in std_ffn.parameters())
swiglu_params = sum(p.numel() for p in swiglu_ffn.parameters())

print(f"标准 FFN (GELU)  参数量: {std_params:,}")
print(f"SwiGLU FFN       参数量: {swiglu_params:,}")
print(f"SwiGLU 多出:       {(swiglu_params - std_params) / std_params * 100:.0f}%")
print("  （多了一个 V 矩阵，所以多了 50%）")

# 对比前向传播结果
x_demo = torch.randn(2, 10, d_model_demo)  # (batch=2, seq_len=10, d_model=64)
out_std = std_ffn(x_demo)
out_swiglu = swiglu_ffn(x_demo)

print(f"\n标准 FFN 输出形状:  {out_std.shape}")
print(f"SwiGLU FFN 输出形状: {out_swiglu.shape}")
print(f"标准 FFN 输出范数:  {out_std.norm():.4f}")
print(f"SwiGLU FFN 输出范数: {out_swiglu.norm():.4f}")

print("\n💡 SwiGLU 虽然多了 50% 参数，但因为门控机制更高效，")
print("   实际上同等参数预算下效果更好。")
print("   LLaMA 论文的发现：SwiGLU > PaLM (Swish) > GELU > ReLU")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 60)
print("🎯 Day 9 总结")
print("=" * 60)
print("""
1. FFN = 两层线性变换 + 激活函数：升维 → 激活 → 降维
2. 激活函数是灵魂：没有它，深层网络退化成浅层
3. 演进路线：ReLU → GELU → SwiGLU（从粗暴到精细）
4. FFN 的本质是"键值记忆"：W₁ 是键，W₂ 是值
5. FFN 占了 Transformer 约 2/3 的参数——是模型的"知识库"

下一节：Layer Normalization + 残差连接 — 保证深层网络的信息健康流动！
""")
