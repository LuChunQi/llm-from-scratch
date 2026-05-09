# Day 6: Self-Attention — 让每个词"看见"所有其他词

> **一句话总结**：Self-Attention 是 Transformer 的心脏——它让序列中的每个位置都能直接"关注"所有其他位置，从而捕捉任意距离的依赖关系。

---

## 🤔 为什么需要 Self-Attention？

### 回忆一下 RNN 的痛点

在 Day 4 我们聊过 RNN 的致命缺陷：

- **信息瓶颈**：不管句子多长，所有信息都得压缩进一个隐藏状态向量，像一根细细的吸管传递信息
- **梯度消失/爆炸**：反向传播要穿过整个时间步链，长距离依赖很难学到
- **串行计算**：必须先处理第1个词，才能处理第2个，无法并行

Self-Attention 彻底换了个思路——

> **与其让信息沿着链条一步步传，不如让每个词直接和所有其他词"对话"。**

就像开会：
- **RNN 方式**：大家排成一排，第一个人把话传给第二个，第二个传给第三个……传到最后一个人耳朵里时，消息早就走样了
- **Self-Attention 方式**：所有人坐在一起，每个人都能直接听到其他所有人的发言，然后自己判断谁说的最重要

---

## 🔍 Self-Attention 的核心直觉

想象你在读这句话：

> "小明给小红一本书，**她**很开心。"

你要理解"她"指谁，必须"回头看"前面的词。Self-Attention 就是让模型在处理"她"这个词的时候，自动学会去"关注""小红"这个上下文。

更准确地说，对于序列中的**每一个位置**，Self-Attention 都会计算它和**所有其他位置**的"关联程度"，然后用这些关联程度做加权求和。

### 一个生活类比

想象一个**投资组合**：
- 你有 10 只股票（序列中的 10 个词）
- 对于第 1 只股票，你要判断它和其他 9 只的"关联度"
- 关联度高的股票，你给更多权重
- 最终你对第 1 只股票的"理解"，是所有股票的加权平均

这就是 Self-Attention 干的事。

---

## 📐 数学原理：一步步拆解

### 步骤 1：准备 Q、K、V

Self-Attention 的核心是三个矩阵变换：**Query (Q)**、**Key (K)**、**Value (V)**。

用图书馆做类比：

| 概念 | 类比 | 含义 |
|------|------|------|
| **Query (Q)** | 你想搜什么 | "我在找和 XX 相关的书" |
| **Key (K)** | 书的标签/索引 | "这本书是关于 XX 的" |
| **Value (V)** | 书的内容 | 书的实际内容 |

具体来说，对于输入序列 $X$（形状 `[seq_len, d_model]`）：

$$Q = X \cdot W_Q, \quad K = X \cdot W_K, \quad V = X \cdot W_V$$

其中 $W_Q, W_K, W_V$ 是可学习的参数矩阵，形状都是 `[d_model, d_k]`。

> **为什么需要三个矩阵？** 不能直接用 $X$ 吗？
> - 可以，但三个独立的变换给了模型更大的**表达能力**
> - Q 负责"提问"，K 负责"回答"，V 负责"提供信息"
> - 它们各自学习不同的视角

### 步骤 2：计算注意力分数

用 Q 和 K 的点积衡量"匹配程度"：

$$\text{scores} = Q \cdot K^T$$

形状变化：`[seq_len, d_k] × [d_k, seq_len] = [seq_len, seq_len]`

这个矩阵的每个元素 `scores[i][j]` 就表示：**第 i 个词对第 j 个词的"关注度"原始分数**。

### 步骤 3：缩放（Scale）

$$\text{scaled\_scores} = \frac{\text{scores}}{\sqrt{d_k}}$$

> **为什么要除以 $\sqrt{d_k}$？**
>
> 这是个关键细节。当 $d_k$ 很大时，点积的结果会变得很大（因为更多维度相加），导致 softmax 输出接近 one-hot（梯度几乎为零）。
>
> 直觉理解：如果你问 1000 个人同一个问题，每个人的回答汇总后数值会很大。除以 $\sqrt{d_k}$ 就是"归一化"一下，让数值回到合理范围。

### 步骤 4：Softmax 归一化

$$\text{attention\_weights} = \text{softmax}(\text{scaled\_scores})$$

Softmax 把每个位置的分数变成概率分布（和为 1），表示"我应该把多少注意力分配给每个词"。

### 步骤 5：加权求和

$$\text{output} = \text{attention\_weights} \cdot V$$

用注意力权重对 V 做加权平均，得到最终输出。

### 完整公式

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

这就是 Transformer 论文中那个著名的公式。现在你知道它的每一步在干什么了！

---

## 💻 代码实战

我们提供两个版本：
1. **纯 NumPy 版**：看清每一步的数学操作
2. **PyTorch 版**：实际工程中的写法

详见 `self_attention.py`。

### 核心代码片段

```python
# 最精简的 Self-Attention（5 行核心代码）
Q = X @ W_Q  # [seq_len, d_k]
K = X @ W_K  # [seq_len, d_k]
V = X @ W_V  # [seq_len, d_v]

scores = Q @ K.T / np.sqrt(d_k)       # [seq_len, seq_len]
attn_weights = softmax(scores, axis=1) # [seq_len, seq_len]
output = attn_weights @ V              # [seq_len, d_v]
```

---

## 🎯 Self-Attention 的关键特性

### 1. 全局感受野
每个位置都能直接看到所有其他位置。不像 CNN 需要堆叠很多层才能看到远处的信息。

### 2. 动态权重
注意力权重是**根据输入内容动态计算的**，不是固定的。同样的模型，不同的输入会产生不同的注意力模式。

### 3. 可并行
不像 RNN 必须串行，Self-Attention 可以用矩阵乘法一次性算完整个序列。

### 4. 计算复杂度
- 时间复杂度：$O(n^2 \cdot d)$，其中 $n$ 是序列长度
- 这是 Self-Attention 的主要瓶颈——长序列时计算量急剧增大
- 这也是为什么后续有 Linear Attention、Flash Attention 等优化方案（我们 Day 15 会讲）

---

## 🔬 注意力矩阵的可视化

理解 Self-Attention 最直观的方式是**可视化注意力矩阵**。

想象一个 4×4 的注意力矩阵（4 个词的句子）：

```
        词1  词2  词3  词4
词1    [0.7  0.1  0.1  0.1]   ← 词1 主要关注自己
词2    [0.1  0.5  0.3  0.1]   ← 词2 关注自己和词3
词3    [0.2  0.1  0.6  0.1]   ← 词3 关注词1和自己
词4    [0.1  0.1  0.1  0.7]   ← 词4 主要关注自己
```

每一行表示一个词对所有词的注意力分配。和为 1（softmax 保证）。

在实际的 Transformer 中，注意力模式往往非常有意义：
- 代词会关注它指代的名词
- 形容词会关注它修饰的名词
- 句末的标点关注整个句子

---

## 🆚 Self-Attention vs 其他注意力

| 类型 | 说明 |
|------|------|
| **Self-Attention** | Q、K、V 都来自同一个序列 |
| **Cross-Attention** | Q 来自一个序列，K、V 来自另一个序列（翻译任务中常见） |
| **Masked Self-Attention** | 未来位置被 mask 掉（Decoder 中使用） |

今天只讲 Self-Attention，后续 Day 7（Multi-Head）和 Day 12（Decoder-Only）会展开其他变体。

---

## ❓ 常见疑问

### Q: Self-Attention 和普通的 Attention 有什么区别？
**A**: 普通 Attention 通常指 Cross-Attention（Q 和 K/V 来自不同来源），而 Self-Attention 中 Q/K/V 都来自**同一个序列**。可以理解为"自己看自己"。

### Q: 为什么用点积而不是其他相似度度量？
**A**: 点积计算最高效（矩阵乘法），且在实践中效果很好。也有加性注意力（Additive Attention）的方案，但点积更快。

### Q: d_k 和 d_model 什么关系？
**A**: 在原始 Transformer 中，`d_k = d_model / num_heads`。但在 Single-Head Self-Attention 中，通常 `d_k = d_model`。Multi-Head 时会把 d_model 拆分成多个小空间。

### Q: Self-Attention 能处理变长序列吗？
**A**: 能。只需要把变长序列 padding 到相同长度，然后在 attention mask 中忽略 padding 位置即可。

---

## 📚 关键术语速查

| 术语 | 英文 | 含义 |
|------|------|------|
| 自注意力 | Self-Attention | 序列对自身的注意力机制 |
| 查询 | Query (Q) | "我在找什么"的向量表示 |
| 键 | Key (K) | "我能提供什么信息"的向量表示 |
| 值 | Value (V) | 实际的信息内容 |
| 缩放点积注意力 | Scaled Dot-Product Attention | 除以 √d_k 的点积注意力 |
| 注意力权重 | Attention Weights | softmax 归一化后的关注度 |
| 全局感受野 | Global Receptive Field | 每个位置能看到所有位置 |

---

## 🔗 扩展阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始论文
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar 的经典可视化教程
- [build-nanogpt](https://github.com/karpathy/build-nanogpt) — Karpathy 的最小 GPT 实现

---

**下节预告**：Day 6 我们拆解了单头 Self-Attention，但一个问题始终悬在头顶——**一个注意力头只能学到一种关注模式，语言的复杂性远不止于此。** 明天 Day 7 我们将升级到 Multi-Head Attention：为什么一个头不够用？多个"头"各自关注什么？

---

*本课程代码开源于 [GitHub](https://github.com/nianyeye/llm-course)，欢迎 Star ⭐*
