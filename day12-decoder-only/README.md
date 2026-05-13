# Day 12: Decoder-Only 架构 — 给 Transformer 戴上"眼罩"

## 🔗 上节回顾

昨天我们深入拆解了残差连接的数学本质：一个零参数的加法 `F(x) + x`，创造了无数条梯度直通路径，让信息在任意深的网络中无损流动。我们还从"残差流"的新视角理解了 Transformer——每层向河流中注入或提取信息，原始 embedding 始终完好无损。

至此，我们已经集齐了 Transformer 的所有核心零件：Self-Attention（看）、Multi-Head（多角度看）、位置编码（知道位置）、FFN（消化存储）、LayerNorm（稳定数值）、残差连接（信息保底）。

但我们留下了一个关键悬念：**这些零件目前是"全局视野"——每个 token 都能看到所有其他 token。** 在语言生成任务中，模型预测下一个词时不应该偷看后面的"答案"。怎么做到"只看过去，不看未来"？

今天我们解决这个问题，并由此引出 GPT 使用的 **Decoder-Only 架构**。

---

## 👁️ 从一个问题开始：为什么不能偷看未来？

> ❓ **问题**：模型看到 "The cat sat on the"，要预测下一个词。如果模型同时看到了 "mat"，那预测 "mat" 还有什么意义？

这就像考试时答案已经印在卷子上——模型会"抄答案"而不是"学推理"。训练时看着答案抄，推理时没答案了，当然就傻了。

用数学语言说：

- **训练时**：模型输入一整句话 `["The", "cat", "sat", "on", "the", "mat"]`
- **目标**：对每个位置 i，用位置 i 及之前的词预测位置 i+1 的词
- **约束**：位置 i 的预测**不能**用到位置 i+1, i+2, ... 的信息

这就是**因果性约束（Causality）**：原因必须在前，结果必须在后。

> ❓ **问题**：这不就是语言模型的老传统吗？N-gram 不也只看前面的词？

没错！从 N-gram 到 RNN，所有语言模型都遵守因果性——只看过去，不看未来。问题是，**Self-Attention 天生是"全员可见"的**。

回忆 Day 6：Self-Attention 中，每个 token 都和所有 token 计算注意力分数。这意味着 "The" 可以看到 "mat"，"cat" 也可以看到 "mat"。这就打破了因果约束。

所以我们需要一个机制：**在保留 Self-Attention 强大能力的同时，阻止信息从未来流向过去。**

---

## 🎭 Causal Mask：最优雅的解决方案

> ❓ **问题**：怎样让 Self-Attention 只看过去？一个朴素的想法是——让每个 token 只和前面的 token 算注意力。但怎么高效实现？

答案是 **Causal Mask（因果掩码）**，也叫 **Attention Mask** 或 **下三角掩码**。

### 核心思想

在计算注意力权重之前，用一个掩码矩阵把"不该看到"的位置设为负无穷：

```
Attention Scores (QK^T):
        pos0  pos1  pos2  pos3
pos0  [  2.1   0.5  -0.3   1.8 ]
pos1  [  0.9   3.2   0.1  -0.5 ]
pos2  [ -0.4   1.1   2.7   0.3 ]
pos3  [  1.5  -0.8   0.6   3.0 ]

Causal Mask (上三角 = -inf):
        pos0  pos1  pos2  pos3
pos0  [  2.1  -inf  -inf  -inf ]
pos1  [  0.9   3.2  -inf  -inf ]
pos2  [ -0.4   1.1   2.7  -inf ]
pos3  [  1.5  -0.8   0.6   3.0 ]

经过 Softmax:
        pos0  pos1  pos2  pos3
pos0  [ 1.00  0.00  0.00  0.00 ]
pos1  [ 0.19  0.81  0.00  0.00 ]
pos2  [ 0.07  0.38  0.55  0.00 ]
pos3  [ 0.25  0.06  0.11  0.58 ]
```

看懂了吗？下三角矩阵天然实现了因果性：

- **pos0**（第 1 个词）：只能看到自己 → 注意力权重 [1.0, 0, 0, 0]
- **pos1**（第 2 个词）：能看到 pos0 和自己 → 注意力权重 [0.19, 0.81, 0, 0]
- **pos2**（第 3 个词）：能看到 pos0, pos1 和自己 → 注意力权重 [0.07, 0.38, 0.55, 0]
- **pos3**（第 4 个词）：能看到所有 → 注意力权重 [0.25, 0.06, 0.11, 0.58]

**打比方**：Causal Mask 就像一副"马眼罩"——不是把眼睛蒙上，而是只挡住两侧和后方，让马只能看到前方。Transformer 里的每个 token 戴上这副眼罩后，只能看到自己和之前的 token。

### 为什么是负无穷？

因为 Softmax 的性质：

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

当 `x_i = -∞` 时，`exp(-∞) = 0`。所以被掩码遮住的位置在 Softmax 后权重精确为 0。这就意味着**零信息泄漏**——未来 token 的 Value 不会参与加权平均。

### 数学表达

标准的 Scaled Dot-Product Attention：

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

加入 Causal Mask：

```
Attention(Q, K, V) = softmax((QK^T / √d_k) + M) · V

其中 M 是掩码矩阵：
M[i][j] = 0     如果 j ≤ i（可以看到）
M[i][j] = -∞    如果 j > i（不能看到）
```

掩码 M 就是一个**上三角全为负无穷、下三角全为 0** 的矩阵。

### 用 PyTorch 一行生成

```python
# 生成下三角掩码
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
# mask[i][j] = True 表示位置 i 不应该看到位置 j

# 应用掩码
scores = scores.masked_fill(mask, float('-inf'))
```

`torch.triu` 取上三角，`diagonal=1` 表示不包含对角线。所以上三角（不含对角线）为 True（遮住），下三角（含对角线）为 False（可见）。简洁优雅。

---

## 🏗️ Decoder-Only Transformer Block

> ❓ **问题**：有了 Causal Mask，我们怎么把前面学的所有零件组装成一个完整的 Decoder-Only Block？

回顾我们之前搭的 Transformer Block（Day 10/11）：

```
x → LayerNorm → Self-Attention → (+x) → LayerNorm → FFN → (+x)
```

唯一的变化：**Self-Attention 加入 Causal Mask**。

```
x → LayerNorm → Causal Self-Attention → (+x) → LayerNorm → FFN → (+x)
                            ↑
                      加了 Causal Mask!
```

就这么简单——一行掩码，把"全员可见"的 Attention 变成了"只看过去"的 Causal Attention。

### Decoder-Only 的命名由来

Transformer 原始论文（2017）有两种架构：
- **Encoder**：双向注意力，用于理解输入（如翻译的源语言）
- **Decoder**：因果注意力，用于生成输出（如翻译的目标语言）

GPT 只用了 Decoder 部分（去掉了交叉注意力），所以叫 **Decoder-Only**。

今天的三大 LLM 阵营：
| 架构 | 代表模型 | 注意力方式 |
|------|----------|-----------|
| **Decoder-Only** | GPT、LLaMA、Mistral | 因果注意力（只看过去） |
| **Encoder-Only** | BERT | 双向注意力（看全部） |
| **Encoder-Decoder** | T5、BART | 编码看全部，解码只看过去 |

Decoder-Only 为什么赢了？因为**生成能力**——语言模型的核心任务就是"预测下一个词"，天然适合因果注意力。而 BERT 式的双向注意力虽然理解能力强，但不能自然地做生成。

---

## 📊 Causal Mask 的信息论分析

> ❓ **问题**：加了 Causal Mask 后，每个位置"看到的"信息量不一样。这会带来什么影响？

### 位置越靠前，信息越少

- **pos0**：只看到 1 个 token（自己）
- **pos1**：看到 2 个 token
- **pos3**：看到 4 个 token
- **pos(n)**：看到 n+1 个 token

这意味着**句子开头的 token 做预测最难**——它几乎没什么上下文信息。而**句子末尾的 token 拥有最多的上下文**。

### 实际影响

1. **训练损失不均匀**：句子前方的预测损失通常比后方高
2. **Prompt 工程的本质**：给模型足够多的前置信息（prompt），让预测位置有充足的上下文
3. **为什么 GPT 擅长续写**：续写时模型看到了整个 prompt，拥有最多的上下文信息

**打比方**：你在一间黑暗的房间里，面前有一条走廊。刚走进去时，你只能看到脚下（pos0）。每走一步，你多看到一步的距离。走得越远，你"回头看"的视野越开阔。

---

## 🔄 自回归生成：一个词一个词地吐出来

> ❓ **问题**：有了 Causal Mask，训练时可以一次性处理整个序列（并行）。但推理时呢？模型怎么一个词一个词生成？

这就是**自回归生成（Autoregressive Generation）** 的过程：

```
Step 1: 输入 [BOS]
        → 模型预测 "The"

Step 2: 输入 [BOS, "The"]
        → 模型预测 "cat"

Step 3: 输入 [BOS, "The", "cat"]
        → 模型预测 "sat"

Step 4: 输入 [BOS, "The", "cat", "sat"]
        → 模型预测 "on"

...直到预测出 [EOS] 或达到最大长度
```

每次生成一个新词后，把它**追加到输入末尾**，重新送入模型预测下一个词。循环往复，直到模型输出结束标记。

### 训练 vs 推理的关键区别

| | 训练 | 推理 |
|---|---|---|
| **输入** | 完整序列 | 逐步增长 |
| **并行** | ✅ 一次 forward | ❌ 每步一次 forward |
| **速度** | 快 | 慢（随长度线性增长） |
| **目标** | 预测每个位置的下一个词 | 只预测最后一个位置的下一个词 |

训练时并行效率高的原因：Causal Mask 让我们可以在**一次 forward pass** 中同时计算所有位置的预测。虽然 pos0 只能看到自己，但 pos3 可以看到全部——这些计算是同时完成的。

推理时就不行了，因为下一个词还没生成出来，没法提前算。

> 🤔 这个效率问题后来催生了 **KV Cache**（Day 13）——用空间换时间，让推理也能快起来。我们明天详细拆解。

---

## 🧩 Decoder-Only 的训练目标：Next Token Prediction

> ❓ **问题**：Decoder-Only 模型的训练目标到底是什么？

### 交叉熵损失

对于输入序列 `[x_0, x_1, ..., x_n]`，模型在每个位置 i 预测 x_{i+1}：

```
Loss = -Σ_{i=0}^{n-1} log P(x_{i+1} | x_0, x_1, ..., x_i)
```

翻译成人话：**在每个位置，模型输出的概率分布中，正确答案的概率越大越好。**

举个例子：
- 输入：`["我", "爱", "北京", "天安门"]`
- 位置 0 的目标：预测"爱"（给定"我"）
- 位置 1 的目标：预测"北京"（给定"我爱"）
- 位置 2 的目标：预测"天安门"（给定"我爱北京"）
- 位置 3 的目标：预测 EOS（给定"我爱北京天安门"）

每个位置都贡献一个损失，最终取平均。

### Teacher Forcing

注意一个关键细节：训练时，每个位置的输入是**真实答案**，不是模型自己预测的词。

比如位置 1 的输入永远是"爱"（ground truth），不管位置 0 模型预测的是什么。这叫 **Teacher Forcing**（老师强制）——不管学生答什么，老师总是把正确答案写在黑板上让学生继续学。

好处：训练稳定、收敛快。坏处：训练和推理的输入分布不一致（训练用真实词，推理用模型预测词），可能导致 **Exposure Bias**。

---

## 🔍 Causal Mask 在不同 Head 中的表现

> ❓ **问题**：Multi-Head Attention 中，每个 Head 都有相同的 Causal Mask 吗？

**是的**——Causal Mask 对所有 Head 一视同仁。每个 Head 都独立地、只能看到当前位置及之前的 token。

但不同 Head 会在**相同的视野范围内**关注不同的信息：
- Head 1 可能专注于语法关系（主语-谓语）
- Head 2 可能专注于语义相似性
- Head 3 可能专注于位置相近的词

Causal Mask 定义了"视野边界"，而每个 Head 在边界内自由选择看什么。

**打比方**：所有学生都坐在同一个教室里（相同视野），但有的看黑板，有的看书本，有的看笔记（不同关注点）。Causal Mask 就是教室的墙壁——所有人都被同样的墙壁限制，但墙壁内各看各的。

---

## 🏛️ 完整的 Decoder-Only Transformer

> ❓ **问题**：把所有零件组装起来，一个完整的 Decoder-Only Transformer 长什么样？

```
输入: [BOS, "我", "爱", "北京", "天安门"]
        │
        ▼
  Token Embedding + Positional Encoding
        │
        ▼
┌─── Transformer Block × N ────────────────┐
│  x → LayerNorm → Causal Self-Attn → (+x) │
│  x → LayerNorm → FFN → (+x)              │
└───────────────────────────────────────────┘
        │
        ▼
  Final LayerNorm
        │
        ▼
  Linear(d_model → vocab_size)    # 投影到词表大小
        │
        ▼
  Softmax → 概率分布               # 每个位置预测下一个词
```

这就是 GPT 的完整架构。没有 Encoder，没有 Cross-Attention，只有一堆相同的 Block 堆叠。

### GPT 系列的参数对比

| 模型 | 层数 N | d_model | Head 数 | 参数量 |
|------|--------|---------|---------|--------|
| GPT-2 Small | 12 | 768 | 12 | 117M |
| GPT-2 Medium | 24 | 1024 | 16 | 345M |
| GPT-2 Large | 36 | 1280 | 20 | 774M |
| GPT-2 XL | 48 | 1600 | 25 | 1.5B |
| GPT-3 | 96 | 12288 | 96 | 175B |

核心架构完全一样，只是"堆叠数量"不同。这就是 **Scaling Law** 的魅力——架构对了，只要加参数就能变强。

---

## 🎯 Causal Mask 的隐藏代价

> ❓ **问题**：Causal Mask 有什么坏处吗？

### 1. 表达能力受限

双向注意力（BERT）中，"猫"可以通过"吃鱼"来理解"猫"是什么。而 Causal Attention 中，"猫"出现时还没看到"吃鱼"，只能靠自己。

这让 Decoder-Only 模型在**理解类任务**上（如分类、NER）不如 Encoder-Only 模型。

### 2. 前方位置的表示质量低

序列开头的 token 因为看到的上下文少，它们的表示（embedding）质量较差。这对需要全局理解的任务（如情感分析）是不利的。

### 3. 训练效率的问题

每个位置都在做预测，但前方的预测"信息量"很低（上下文少）。一种改进是给后方位置更高的损失权重。

### 但为什么 Decoder-Only 还是赢了？

因为**统一了理解和生成**：

1. 生成能力是 LLM 的核心价值
2. 通过足够大的数据和参数，Decoder-Only 的理解能力也能追上 BERT
3. 一个模型同时做理解和生成（GPT-4），比两个模型各做一样更高效

**打比方**：Causal Mask 像一个"单行道"的视野限制——虽然不如"360度全景"看得多，但"一边走一边看"的方式更适合"边走边说"的生成任务。而且路走得多了（参数大了），单行道也能走得很远。

---

## 🧪 动手实验

今天的代码 `decoder_only.py` 包含：

1. **Causal Mask 可视化** — 看清楚掩码长什么样，理解它如何遮住未来信息
2. **有/无 Causal Mask 的注意力对比** — 同一个序列，看注意力模式的巨大差异
3. **完整 Decoder-Only Block** — 把所有零件组装起来（LayerNorm + Causal Attention + FFN + 残差）
4. **自回归生成** — 用训练好的 mini 模型一步步生成文本
5. **Teacher Forcing vs Free Running** — 对比训练和推理模式的区别

运行方式：
```bash
python3 decoder_only.py
```

---

## 🔑 关键洞察

1. **Causal Mask 的本质**：一个上三角为负无穷的矩阵，通过 Softmax 让未来 token 的注意力权重为 0，实现"只看过去"
2. **为什么 Decoder-Only 赢了**：因果注意力天然匹配"预测下一个词"的任务，统一了理解和生成
3. **训练时并行，推理时串行**：Causal Mask 让训练时可以一次 forward 计算所有位置，但推理时只能逐步生成
4. **Next Token Prediction**：Decoder-Only 的训练目标就是在每个位置预测下一个词，用交叉熵损失优化
5. **架构简单但强大**：GPT 就是 N 个相同的 Block 堆叠，只有 Causal Mask 这一个关键改动

---

## 📝 一句话总结

> Causal Mask 用一个简单的上三角负无穷矩阵，把"全员可见"的 Self-Attention 变成"只看过去"的因果注意力——这一个改动，让 Transformer 从理解工具变成了生成引擎，催生了 GPT 和整个大语言模型时代。

## 📖 关键术语速查

| 术语 | 含义 |
|------|------|
| **Causal Mask** | 因果掩码，上三角为 -inf 的矩阵，阻止看到未来信息 |
| **Autoregressive** | 自回归，逐个生成 token，每次把生成结果追加到输入 |
| **Decoder-Only** | 只使用 Transformer Decoder 的架构（GPT 系列） |
| **Next Token Prediction** | 预测下一个词，Decoder-Only 的训练目标 |
| **Teacher Forcing** | 训练时用真实答案作为输入，而非模型预测值 |
| **Exposure Bias** | 训练和推理输入分布不一致导致的问题 |
| **BOS / EOS** | 序列起始/结束标记 |
| **Scaling Law** | 规模法则——架构相同时，参数越多性能越好 |

---

> 🤔 **今天留下的悬念**：Decoder-Only 推理时是自回归的——每生成一个新词，就要把整个序列重新送入模型计算一次。序列越长，计算量越大，速度越慢。生成 1000 个词需要 1000 次 forward pass？有没有办法优化？明天我们认识 **KV Cache**——用空间换时间的魔法，让推理速度提升数倍。

**下节预告**：Day 13 — KV Cache，如何让自回归生成快起来。

---

*本课程代码开源于 [GitHub](https://github.com/LuChunQi/llm-from-scratch)，欢迎 Star ⭐*
