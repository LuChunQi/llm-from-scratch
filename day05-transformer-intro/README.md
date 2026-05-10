# Day 5 - Transformer 登场：Attention Is All You Need

---

## 🔗 上节回顾

昨天我们走完了一段精彩的演进之路：RNN 能"记住"前面的词，但记忆会衰减；LSTM 加了门控缓解衰减，但还是串行计算，又慢又记不住超长的序列；Seq2Seq 把长句子压成一个向量，信息装不下；最后 Attention 说出那句经典——"别压缩了，按需取用"。

结尾我们提到了一个疯狂的想法：**既然 Attention 根本不需要 RNN 的链式结构，那我们能不能干脆把 RNN 扔掉？**

2017 年，Google 的 8 位研究员用一篇论文回答了这个问题。标题霸气侧漏——**"Attention Is All You Need"**（注意力就是你需要的一切）。他们提出了 **Transformer**，一个纯 Attention 架构，彻底抛弃了 RNN。

今天你听过的所有大模型——GPT、BERT、LLaMA、Claude——全部基于它。

---

## 🎯 今天学什么？

我们要把 Transformer 从里到外拆解一遍：它由哪些组件构成？每个组件解决了什么问题？信息在里面是怎么流动的？

---

## 🧠 从一个比喻开始：团队开会

想象你在一个 10 人团队里开会，讨论一个项目方案：

- **RNN 的方式**：大家排成一排，每个人只能听前一个人说完，再把自己的想法传给下一个人。第 10 个人想了解第 1 个人说的内容？信息已经衰减得差不多了。
- **Transformer 的方式**：所有人围坐在圆桌旁，每个人可以**直接看向**任何其他人，决定自己要**关注谁**。而且每个人是同时看所有人的——**完全并行**。

这就是 Transformer 的核心思想：**每个位置都可以直接关注序列中的任何其他位置，不需要信息一步步传递。**

---

## 🏗️ Transformer 的整体架构

原版 Transformer 是一个 **Encoder-Decoder** 结构（用于翻译任务）：

```
输入 (源语言) → [Encoder × N 层] → 编码表示 → [Decoder × N 层] → 输出 (目标语言)
```

我们先不纠结每一个细节，先看大图：

```
┌──────────────────────────────────────────────────┐
│                   Transformer                     │
│                                                   │
│  输入嵌入 + 位置编码                               │
│       ↓                                           │
│  ┌─────────────────┐                              │
│  │  Multi-Head      │  ← 多头注意力               │
│  │  Self-Attention  │                              │
│  └────────┬────────┘                              │
│           ↓  (Add & Norm)                         │
│  ┌─────────────────┐                              │
│  │  Feed-Forward    │  ← 前馈网络                  │
│  │  Network (FFN)   │                              │
│  └────────┬────────┘                              │
│           ↓  (Add & Norm)                         │
│       ... (重复 N 次) ...                          │
│           ↓                                       │
│      输出表示                                      │
└──────────────────────────────────────────────────┘
```

每个 Transformer 层（也叫 Transformer Block）包含两个核心子层：

1. **Multi-Head Self-Attention**（多头自注意力）
2. **Feed-Forward Network**（前馈神经网络）

每个子层后面都有 **残差连接 + Layer Normalization**（Add & Norm）。

---

## 🔑 核心概念逐一拆解

### 1. 自注意力（Self-Attention）：每个人都在看所有人

> ❓ **问题**：RNN 串行传信息会衰减。那怎么让每个词都能直接"看到"所有其他词，不管它们隔多远？

#### 一个直觉理解

假设有一句话："小明去了商店，他买了一本书。"

当你读到"他"这个字时，你的大脑会自动把"他"和"小明"关联起来。**自注意力就是在做这件事——计算每个词和所有其他词的关联程度。**

#### Q、K、V：注意力三剑客

自注意力的计算用到三个矩阵：**Query（Q）、Key（K）、Value（V）**。

类比一下：
- **Query（查询）**：你在图书馆找书时输入的搜索关键词
- **Key（标签）**：每本书上贴的分类标签
- **Value（内容）**：书的实际内容

计算过程：
1. 每个词生成自己的 Q、K、V（通过乘以三个权重矩阵）
2. 用 Q 和所有 K 算"相似度"（点积），得到注意力分数
3. 用 softmax 归一化，得到注意力权重
4. 用权重对 V 加权求和，得到输出

数学公式：

```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
```

**为什么要除以 √d_k？** 因为当维度很大时，点积的值会变得很大，softmax 的梯度会消失（进入饱和区）。除以 √d_k 相当于做了一个温度调节，让梯度保持健康。

### 2. 多头注意力（Multi-Head Attention）：多个视角看世界

> ❓ **问题**：一组 Q/K/V 只能学到一种关注模式（比如只看语法关系）。但语言是复杂的——词与词之间有语法关系、语义关系、指代关系……一组注意力根本不够用。怎么办？

**答案：用多组 Q/K/V，让模型从多个不同的"视角"理解输入。**

类比：看一部电影，一个人关注剧情，一个人关注画面，一个人关注配乐——最后综合所有人的感受，得到更全面的理解。

```
输入 X
├── Head 1: Q₁=XW₁ᵠ, K₁=XW₁ᵏ, V₁=XW₁ᵛ → Attention₁
├── Head 2: Q₂=XW₂ᵠ, K₂=XW₂ᵏ, V₂=XW₂ᵛ → Attention₂
├── ...
└── Head h: Qₕ=XWₕᵠ, Kₕ=XWₕᵏ, Vₕ=XWₕᵛ → Attentionₕ
         ↓
   Concat(Attention₁, ..., Attentionₕ) × Wᴼ
         ↓
       输出
```

### 3. 位置编码（Positional Encoding）：注入顺序信息

> ❓ **问题**：Attention 机制有一个天生的盲点——它完全不知道词的顺序！"我爱你"和"你爱我"在 Attention 眼里一模一样。怎么弥补？

**答案：给每个位置生成一个独特的"身份证"，加到词向量上。**

原论文用的是正弦/余弦编码：

```math
PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})
```

```math
PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})
```

这就像给每个位置一个**独一无二的条形码**，让模型知道谁在前面、谁在后面。

### 4. 前馈网络（FFN）：每个位置的"思考"

> ❓ **问题**：Attention 负责收集全局信息，但收集来的信息还没被"消化"。谁来做这个思考加工？

每个 Transformer 层除了注意力，还有一个 FFN：

```math
\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2
```

这是一个两层全连接网络，对每个位置**独立**做非线性变换。如果说注意力是"收集信息"，那 FFN 就是"处理信息"。

### 5. 残差连接 & Layer Norm

> ❓ **问题**：Transformer 层叠得很深（GPT-3 有 96 层），深层网络容易出现梯度消失。怎么让梯度在深网中畅行无阻？

每个子层的输出都是：

```math
\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))
```

- **残差连接**（x + Sublayer(x)）：直接把输入"短路"到输出，缓解深层网络的梯度消失问题。就像高速路上的"快速通道"，信息可以直接流过。
- **Layer Normalization**：对每个样本的特征维度做归一化，稳定训练。

### 6. Encoder vs Decoder

> ❓ **问题**：Transformer 诞生时是为了翻译任务（ Encoder 读源语言，Decoder 生成目标语言）。但现在 GPT 只用 Decoder，BERT 只用 Encoder——为什么？

| 特性 | Encoder | Decoder |
|------|---------|---------|
| 注意力类型 | 双向自注意力 | 带掩码的自注意力 + 交叉注意力 |
| 能看到什么 | 整个输入序列 | 只能看到当前位置之前的内容 |
| 典型代表 | BERT | GPT |
| 用途 | 理解（分类、标注） | 生成（翻译、对话） |

**Decoder 的掩码（Mask）**：生成第 t 个词时，不能偷看第 t+1 个词，否则就是"作弊"。掩码把未来位置的注意力分数设为负无穷，softmax 后就变成 0。

---

## 🌊 信息流：一个完整的 Transformer 层长什么样？

以 Encoder 层为例，数据流如下：

```
输入 x (shape: [batch, seq_len, d_model])
  │
  ├──→ Multi-Head Self-Attention
  │         │
  │         ↓
  │    attention_output (shape: [batch, seq_len, d_model])
  │         │
  │         ↓ + x (残差连接)
  │         │
  │    LayerNorm
  │         │
  │         ↓
  │    norm_output (shape: [batch, seq_len, d_model])
  │         │
  ├──→ Feed-Forward Network
  │         │
  │         ↓
  │    ffn_output (shape: [batch, seq_len, d_model])
  │         │
  │         ↓ + norm_output (残差连接)
  │         │
  │    LayerNorm
  │         │
  ↓         ↓
输出 (shape: [batch, seq_len, d_model])
```

注意：**形状始终不变！** 这是 Transformer 的一个优雅之处——输入输出形状相同，所以可以像乐高积木一样堆叠任意多层。

---

## 📊 参数量速算

一个 Transformer 层的参数量（近似）：

| 组件 | 参数量 |
|------|--------|
| 组件 | 参数量 |
|------|--------|
| Q/K/V 投影 | 4 × d²_model（含输出投影） |
| FFN | 8 × d²_model（中间层通常 4 倍） |
| LayerNorm | 4 × d_model |
| **每层总计** | ≈ 12 × d²_model |

例如 d_model = 768（BERT-Base），每层约 12 × 768² ≈ 7M 参数，12 层总共约 84M（再加上 embedding 约 110M）。

---

## 🔬 原始论文的关键数字

| 超参数 | Base 模型 | Big 模型 |
|--------|----------|---------|
| 层数 (N) | 6 | 6 |
| d_model | 512 | 1024 |
| FFN 中间维度 | 2048 | 4096 |
| 注意力头数 (h) | 8 | 16 |
| 参数量 | 65M | 213M |

这些数字在今天看来很小（GPT-4 估计上万亿参数），但在 2017 年这是 SOTA。

---

## 🚀 为什么 Transformer 这么强？

1. **并行计算**：不像 RNN 需要逐步处理，Transformer 所有位置同时计算 → GPU 友好
2. **长距离依赖**：任何两个位置之间的"距离"都是 O(1)，不像 RNN 是 O(n)
3. **可扩展性**：更大的模型 + 更多数据 = 更强的性能（Scaling Law）
4. **灵活性**：Encoder-Decoder、Decoder-Only、Encoder-Only，三种变体各有所长

---

## 🔗 今天的代码

`transformer_skeleton.py` — 用 PyTorch 搭建一个完整的 Transformer 模型骨架，包含：
- Scaled Dot-Product Attention
- Multi-Head Attention
- Position-wise FFN
- Positional Encoding
- 完整的 Encoder Layer
- 堆叠多层 Encoder
- 用随机数据做一次前向传播，验证模型能跑通

---

## 📝 一句话总结

> Transformer 用"注意力"取代了"循环"，让每个位置都能直接看到全局信息，实现了并行计算和长距离依赖，开启了大规模预训练模型的时代。

---

> 🤔 **今天留下的悬念**：我们拆解了 Transformer 的骨架，但 Self-Attention 的内部计算细节还没展开——Q、K、V 到底怎么算？注意力权重是什么形状？缩放因子为什么是 √d_k？明天 Day 6 我们将深入 Self-Attention 的数学细节，手写每一行代码。

**下节预告**：Day 6 — Self-Attention 深度拆解，让每个词"看见"所有其他词。

---

*本课程代码开源于 [GitHub](https://github.com/nianyeye/llm-course)，欢迎 Star ⭐*

---

## 📖 关键术语速查

| 术语 | 解释 |
|------|------|
| **Self-Attention** | 自注意力，每个位置对序列中所有位置计算相关性 |
| **Q/K/V** | Query、Key、Value，注意力的三个核心矩阵 |
| **Multi-Head Attention** | 多头注意力，多组 QKV 并行计算，捕获不同模式的关联 |
| **Positional Encoding** | 位置编码，给无序的 Transformer 注入位置信息 |
| **FFN** | 前馈网络，对每个位置独立做非线性变换 |
| **Residual Connection** | 残差连接，输入直接加到输出，缓解梯度消失 |
| **Layer Normalization** | 层归一化，稳定训练过程 |
| **Mask** | 掩码，在 Decoder 中遮挡未来信息 |
| **Encoder** | 编码器，双向注意力，用于理解任务 |
| **Decoder** | 解码器，单向注意力，用于生成任务 |
| **d_model** | 模型的隐藏维度 |
| **Scaled Dot-Product** | 缩放点积注意力，除以 √d_k 防止梯度消失 |
