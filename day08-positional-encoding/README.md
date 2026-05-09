# Day 8: 位置编码 — Transformer 怎么知道"我爱你"≠"你爱我"？

## 🤔 一个致命的问题

上两节课我们搞定了 Self-Attention 和 Multi-Head Attention。Attention 机制很强大——它能让你关注到句子中重要的词。

但是，Attention 有一个**天生的盲点**：

> **它完全不知道词的顺序！**

你信不信？把"我爱你"打乱成"你爱我"、"爱你我"、"我爱死你"，在 Attention 眼里它们**一模一样**。

为什么？因为 Self-Attention 的核心操作是：

```
Attention(Q, K, V) = softmax(QK^T / √d) · V
```

这个操作本质上是：每个词和其他所有词做点积，算相似度，然后加权求和。**没有任何地方用到了位置信息**。它就像一个失忆症患者，能看到每个人，但完全分不清谁在前谁在后。

而语言是高度依赖顺序的：
- "狗咬人" vs "人咬狗" — 完全不同的意思
- "我喜欢你" vs "你喜欢我" — 视角不同
- "不，行" vs "行，不" — 一个同意，一个拒绝

所以 Transformer 必须有办法**告诉每个词"你在句子的哪个位置"**。这就是**位置编码（Positional Encoding）**要解决的问题。

---

## 🎯 位置编码的核心思想

一句话概括：**给每个位置生成一个独特的"身份证"，然后把它加到词向量上。**

就像给学生编学号：
- 张三 → 学号 01
- 李四 → 学号 02
- 王五 → 学号 03

这样即使张三和李四换了座位，你也能通过学号知道谁是谁。

在 Transformer 里，这个"学号"不是简单的数字 1, 2, 3... 而是一个**向量**，维度和词向量一样。位置向量 + 词向量 = 带位置信息的最终表示。

你可能会问：为什么不直接用 1, 2, 3？好问题，后面会解释。

---

## 📐 方法一：Sinusoidal 位置编码（Transformer 原版）

这是 2017 年 "Attention Is All You Need" 论文中的原始方案。

### 公式

对于位置 `pos`，维度 `i`：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

其中 `d` 是嵌入维度（比如 512）。

### 直觉理解

想象一个**精密的时钟系统**：
- 低维度（i 小）的频率高 → 变化快，像秒针
- 高维度（i 大）的频率低 → 变化慢，像时针

每个位置 pos 在这 d/2 个"时钟"上的读数组合起来，就形成了一个**唯一的位置指纹**。

### 为什么不用简单的 1, 2, 3...？

简单数字有几个问题：
1. **无法泛化**：训练时见过长度 100 的句子，推理时遇到长度 150 就不知道怎么编码了
2. **值域不匹配**：位置 1000 的值太大，会和词向量"打架"
3. **没有相对关系**：位置 1 和位置 2 的关系，应该和位置 100 与 101 的关系类似

Sinusoidal 编码解决了所有这些问题：
- 值域在 [-1, 1]，和词向量匹配
- 可以外推到任意长度
- 隐含了相对位置关系（通过三角函数的相位差）

### 代码实现

```python
import torch
import math

def sinusoidal_pe(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
    
    # 计算分母 10000^(2i/d)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    
    # 偶数维度用 sin，奇数维度用 cos
    pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, d_model/2)
    
    return pe
```

---

## 🚀 方法二：RoPE（旋转位置编码）

这是目前**最主流**的位置编码方案，被 LLaMA、Qwen、DeepSeek 等几乎所有现代 LLM 采用。

### 为什么需要更好的方案？

Sinusoidal 编码有个局限：它是**绝对位置**编码。它告诉模型"你在第几个位置"，但不直接帮助模型理解"你和某个词隔了多远"。

而语言中更重要的是**相对位置**：
- "我喜欢吃苹果"中，"我"和"吃"隔了 1 个词
- "我昨天特别喜欢吃了一个大苹果"中，"我"和"吃"隔了 4 个词

RoPE（Rotary Position Embedding）的设计目标是：**让两个位置的内积自然地反映它们的相对距离**。

### 核心思想：旋转！

想象一个 2D 平面上的点。如果我们**旋转**这个点，它离原点的距离不变，但方向变了。

RoPE 的想法：**用位置 pos 决定旋转的角度**。位置 0 不旋转，位置 1 旋转 θ，位置 2 旋转 2θ...

对于高维向量，把 d 维分成 d/2 对，每对独立旋转，角度不同（类似 Sinusoidal 的思路）。

### 数学表达

对于位置 `pos` 的向量 `x = [x₀, x₁, x₂, x₃, ...]`：

```
RoPE(x, pos) = [x₀cos(mθ₀) - x₁sin(mθ₀),
                x₀sin(mθ₀) + x₁cos(mθ₀),
                x₂cos(mθ₁) - x₃sin(mθ₁),
                x₂sin(mθ₁) + x₃cos(mθ₁),
                ...]
```

其中 `m` 是位置 `pos`，`θᵢ = 1/10000^(2i/d)`。

### 为什么 RoPE 这么强？

当你对位置 `m` 的向量 q 和位置 `n` 的向量 k 做内积（也就是 Attention 的核心操作）时：

```
<RoPE(q, m), RoPE(k, n)> = f(q, k, m - n)
```

**内积只依赖于相对位置 `m - n`！** 这正是我们想要的。

### 代码实现

```python
def rotate_half(x):
    """把向量的后半部分取负，前半后后交换"""
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, freqs):
    """应用 RoPE 到输入 x
    x: (batch, seq_len, num_heads, head_dim)
    freqs: (seq_len, head_dim)
    """
    cos_val = torch.cos(freqs).unsqueeze(0).unsqueeze(2)  # 广播
    sin_val = torch.sin(freqs).unsqueeze(0).unsqueeze(2)
    return x * cos_val + rotate_half(x) * sin_val
```

---

## 🆚 方法三：ALiBi（Attention with Linear Biases）

这是一个**不用位置编码**的方案。你没听错——它不往词向量上加任何东西，而是直接在 Attention 计算时动手脚。

### 核心思想

在算 Attention 分数（softmax 之前）时，直接减去一个和距离成正比的偏置：

```
attention_score(i, j) = q_i · k_j - m * |i - j|
```

其中 `m` 是一个固定的斜率，不同的注意力头使用不同的斜率。

就像在说：**你离我越远，我对你的注意力就打折越多。**

### 特点

- **极简**：不需要额外的位置编码，只需要在 Attention 加一个偏置
- **长度外推超强**：训练时用短序列，推理时可以直接用长序列
- 被 BLOOM、MPT 等模型采用

### 代码实现

```python
def alibi_bias(num_heads, seq_len):
    """生成 ALiBi 偏置矩阵"""
    # 每个头的斜率，按 2 的幂次衰减
    slopes = 2 ** (-8 * torch.arange(1, num_heads + 1) / num_heads)
    
    # 距离矩阵
    positions = torch.arange(seq_len)
    distance = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq_len, seq_len)
    distance = distance.abs()
    
    # (num_heads, seq_len, seq_len)
    bias = -slopes.unsqueeze(1).unsqueeze(1) * distance.unsqueeze(0)
    return bias
```

---

## 🔬 三种方案对比

| 特性 | Sinusoidal | RoPE | ALiBi |
|------|-----------|------|-------|
| **类型** | 绝对位置 | 相对位置（隐式） | 相对位置（显式） |
| **应用方式** | 加到词向量 | 旋转 Q 和 K | 加到 Attention 分数 |
| **长度外推** | 一般 | 好（配合 NTK-aware） | 极好 |
| **主流采用** | 原版 Transformer | LLaMA/Qwen/DeepSeek | BLOOM/MPT |
| **实现复杂度** | 简单 | 中等 | 简单 |

### 选择建议
- **学习理解** → Sinusoidal（最直观）
- **实际项目** → RoPE（当前主流，生态最完善）
- **需要超长上下文** → ALiBi 或 RoPE + NTK-aware 扩展

---

## 🧪 动手实验

今天的代码 `positional_encoding.py` 包含：

1. **Sinusoidal 位置编码可视化** — 看看不同位置的"指纹"长什么样
2. **RoPE 实现与验证** — 验证内积确实只依赖相对位置
3. **ALiBi 偏置可视化** — 看看不同头的距离惩罚长什么样
4. **三种方案对比实验** — 用同一句话看三种编码的差异

运行方式：
```bash
python3 positional_encoding.py
```

---

## 🔑 关键洞察

1. **Transformer 本身没有位置感知能力**，必须通过位置编码注入
2. **位置编码是加在输入端的**（Sinusoidal），还是融入 Attention 计算的（RoPE/ALiBi），这是核心区别
3. **RoPE 是当前最佳实践**，通过旋转实现相对位置感知，被几乎所有现代 LLM 采用
4. **长度外推**是位置编码的关键挑战——训练时没见过的位置，推理时能不能处理

---

## 📝 一句话总结

> 位置编码就是给 Transformer 装上"顺序感"——Sinusoidal 用正弦波当身份证，RoPE 用旋转编码相对距离，ALiBi 用距离惩罚直接告诉 Attention"别看太远"。

## 📖 关键术语速查

| 术语 | 含义 |
|------|------|
| **Positional Encoding** | 位置编码，为序列中每个位置生成唯一表示 |
| **Sinusoidal PE** | 用 sin/cos 函数生成的位置编码（Transformer 原版） |
| **RoPE** | 旋转位置编码，通过旋转向量编码相对位置 |
| **ALiBi** | 线性偏置注意力，在 Attention 分数上加距离惩罚 |
| **绝对位置编码** | 编码"你在第几个位置" |
| **相对位置编码** | 编码"你和某个词隔了多远" |
| **长度外推** | 训练时用短序列，推理时能否处理更长序列 |
| **频率矩阵** | RoPE 中的 θᵢ = 1/10000^(2i/d)，控制每对维度的旋转速度 |

---

> 🤔 **今天留下的悬念**：到今天为止，我们已经集齐了 Transformer 的所有零件——Self-Attention、Multi-Head Attention、位置编码、FFN、残差连接。接下来，就是把这些零件组装成一个完整的模型了。

**下节预告**：Day 9 — 动手搭建完整 Transformer！把所有零件拼装起来，跑通第一次前向传播。

---

*本课程代码开源于 [GitHub](https://github.com/nianyeye/llm-course)，欢迎 Star ⭐*
