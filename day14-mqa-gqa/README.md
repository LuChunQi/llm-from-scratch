# Day 14: MQA & GQA — 给 KV Cache 瘦身的"共享经济"

## 🔗 上节回顾

昨天我们拆解了 KV Cache——让自回归生成从 O(N³) 降到 O(N²) 的核心优化。它的思想极其朴素：既然历史 token 的 Key 和 Value 在每一步都不会变，那就算一次存起来复用。但我们也算了一笔账：LLaMA-2-70B 在 4K 序列长度下，单个请求的 KV Cache 就要 **10 GB**，batch_size = 32 时直接飙到 320 GB——比模型参数还多！

我们留了一个悬念：**有没有办法压缩 KV Cache，让它不再这么吃内存？**

今天就来拆解两个"瘦身"利器——**MQA（Multi-Query Attention）** 和 **GQA（Grouped-Query Attention）**，看看 LLaMA-2、GPT-4、Mistral 是怎么用"共享"来省下几十倍内存的。

---

## 🏠 从一个生活类比开始：共享经济

想象你住在一个有 32 户人家的大楼里。

**方案 A：每家自备一套工具箱**（标准 Multi-Head Attention）
- 32 户 × 每套工具箱 500 元 = 16,000 元
- 工具箱占 32 个柜子
- 但每家的锤子、扳手一年也用不了几次

**方案 B：32 户共享 1 套工具箱**（Multi-Query Attention）
- 只买 1 套 = 500 元
- 只占 1 个柜子
- 但高峰期大家都要用，排队排到天荒地老

**方案 C：32 户分成 8 组，每组共享 1 套工具箱**（Grouped-Query Attention）
- 买 8 套 = 4,000 元
- 占 8 个柜子
- 每组 4 户共享，排队时间可接受，开销也大幅减少

这就是 MQA 和 GQA 的核心思想。让我们回到 Transformer 的世界，看它具体怎么运作。

---

## 🔍 先回顾：标准 Multi-Head Attention 的 KV Cache

> ❓ **问题**：标准 MHA 的 KV Cache 为什么这么大？每 byte 都花在了哪里？

在标准的 Multi-Head Attention 中，每个 Head 都有**自己独立的** Q、K、V 投影：

```
输入 x: (B, T, d_model)

# 每个头独立投影
Q = x @ W_q  → (B, T, n_heads, d_head) → 拆成 n_heads 份
K = x @ W_k  → (B, T, n_heads, d_head) → 拆成 n_heads 份
V = x @ W_v  → (B, T, n_heads, d_head) → 拆成 n_heads 份

# 每个头独立做注意力
for head h:
    Q_h, K_h, V_h = Q[:, h], K[:, h], V[:, h]
    attn_h = softmax(Q_h @ K_h.T / sqrt(d_head)) @ V_h
```

KV Cache 存的是每一步的 K 和 V，**按头分开存**：

```
KV Cache 形状（每层）：
  cache_k: (B, n_heads, T, d_head)
  cache_v: (B, n_heads, T, d_head)

总大小（每层）：
  2 × B × n_heads × T × d_head × 2 bytes (float16)
```

**关键观察**：KV Cache 的大小和 `n_heads` 成正比。如果 n_heads = 32，那 K 和 V 就有 32 份。每一份的内容不同，因为每个头学到了不同的注意力模式——有的关注语法，有的关注语义，有的关注位置。

> ❓ **问题**：32 个头的 K 和 V 都不同，能共享吗？共享了不会丢失信息吗？

这个问题的答案，就是 MQA 和 GQA 的分歧点。

---

## 💡 MQA：所有人共享一把钥匙

> ❓ **问题**：如果 32 个头共享同一组 K 和 V，会怎样？

### 核心思想

**Multi-Query Attention（MQA）**，2019 年由 Google 提出，核心改动极其简单：

- **Q（Query）**：每个头仍然独立，n_heads 份
- **K（Key）和 V（Value）**：所有头共享 **1 份**

```
标准 MHA:
  Q: n_heads 份 × (B, T, d_head)
  K: n_heads 份 × (B, T, d_head)    ← 每个头独立的 K
  V: n_heads 份 × (B, T, d_head)    ← 每个头独立的 V

MQA:
  Q: n_heads 份 × (B, T, d_head)
  K: 1 份 × (B, T, d_head)          ← 所有头共享一个 K
  V: 1 份 × (B, T, d_head)          ← 所有头共享一个 V
```

**打比方**：想象一个图书馆，32 个读者（32 个 Head）各自带着自己的问题（Q）来找书。在标准 MHA 中，每个读者有自己的书架索引（K）和书架副本（V）。在 MQA 中，只有一个公共书架索引和一套书——但每个读者带着不同的问题来查，所以他们关注的"重点"仍然不同。

### 实现差异

```python
# 标准 MHA
self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
self.W_k = nn.Linear(d_model, n_heads * d_head, bias=False)  # n_heads × d_head
self.W_v = nn.Linear(d_model, n_heads * d_head, bias=False)  # n_heads × d_head

# MQA
self.W_q = nn.Linear(d_model, n_heads * d_head, bias=False)
self.W_k = nn.Linear(d_model, d_head, bias=False)            # 只有 1 × d_head ！
self.W_v = nn.Linear(d_model, d_head, bias=False)            # 只有 1 × d_head ！
```

注意到区别了吗？W_k 和 W_v 的输出维度从 `n_heads * d_head` 缩减到了 `d_head`。这意味着 K 和 V 的参数量也减少了 n_heads 倍——虽然这只占总参数的一小部分，但对 KV Cache 的影响是巨大的。

### KV Cache 压缩效果

```
标准 MHA KV Cache（每层）：
  cache_k: (B, n_heads, T, d_head)
  cache_v: (B, n_heads, T, d_head)
  总大小: 2 × B × n_heads × T × d_head × 2 bytes

MQA KV Cache（每层）：
  cache_k: (B, 1, T, d_head)          ← 只有一份！
  cache_v: (B, 1, T, d_head)          ← 只有一份！
  总大小: 2 × B × 1 × T × d_head × 2 bytes

压缩比 = n_heads 倍！
```

以 LLaMA-2-70B 为例（n_heads = 64）：
- 标准 MHA：10 GB KV Cache
- MQA：10 GB / 64 ≈ **156 MB**！

从 10 GB 降到 156 MB，batch_size 可以从 1 扩到 64，吞吐量直接起飞。

### MQA 的注意力计算

```python
# MQA 的注意力计算
Q = Q.view(B, T, n_heads, d_head).transpose(1, 2)  # (B, n_heads, T, d_head)
K = K.view(B, T, 1, d_head).transpose(1, 2)          # (B, 1, T, d_head) — 只有 1 个头
V = V.view(B, T, 1, d_head).transpose(1, 2)          # (B, 1, T, d_head) — 只有 1 个头

# 广播（broadcast）：K 和 V 自动复制给所有头
# PyTorch 的广播机制让 (B, 1, T, d_head) 和 (B, n_heads, T, d_head) 运算时自动扩展
scores = Q @ K.transpose(-2, -1)  # (B, n_heads, T, T) — 广播后每个头都算了自己的注意力
attn = softmax(scores / sqrt(d_head))
out = attn @ V                      # (B, n_heads, T, d_head)
```

> 注意：虽然 K 和 V 只有 1 份，但每个头的 Q 不同，所以每个头算出的注意力分数不同，关注的位置不同。**多样性来自 Q 的差异，而不是 K/V 的差异**。

### MQA 的代价

> ❓ **问题**：KV Cache 小了这么多，难道没有代价吗？

当然有。代价就是**表达能力的下降**。

在标准 MHA 中，每个头有独立的 K 和 V，意味着不同头可以关注完全不同的信息。有的头学习"主谓一致"（关注动词和主语的关系），有的头学习"指代消解"（关注代词和先行词的距离），它们各自有独立的索引方式（K）和信息编码方式（V）。

MQA 把所有头的 K 和 V 压缩成 1 份，相当于**所有头必须使用同一套索引和同一套信息编码**。虽然不同的 Q 仍然可以让每个头关注不同的位置，但索引方式（K）和信息内容（V）的多样性丧失了。

实际表现：
- **简单任务**：几乎不影响性能
- **复杂推理、长文本**：可能有轻微的质量下降（通常 1-3%）
- **但在速度和内存的巨大优势面前**，这点质量损失完全可以接受

这就是为什么 Google 的 PaLM、Google 的 Gemini、Meta 的 LLaMA-2 部分、StarCoder 等模型都采用了 MQA 或其变体。

---

## 🎯 GQA：在 MHA 和 MQA 之间找平衡

> ❓ **问题**：MHA 太吃内存，MQA 又太激进——有没有中间方案？

有！**Grouped-Query Attention（GQA）**，2023 年由 Google 在论文《GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints》中提出，被 LLaMA-2 采用后迅速成为主流。

### 核心思想

GQA 把 n_heads 个 Query 头分成 g 组，每组共享同一组 K 和 V。

```
标准 MHA（g = n_heads）：每个头独立 K, V
  Q_1 → K_1, V_1
  Q_2 → K_2, V_2
  ...
  Q_n → K_n, V_n

MQA（g = 1）：所有头共享 K, V
  Q_1 → K, V
  Q_2 → K, V
  ...
  Q_n → K, V

GQA（g 组）：每组共享 K, V
  Q_1, Q_2, Q_3, Q_4 → K_1, V_1       ← 第 1 组（4 个 Q 共享 1 组 KV）
  Q_5, Q_6, Q_7, Q_8 → K_2, V_2       ← 第 2 组
  Q_9, Q_10, Q_11, Q_12 → K_3, V_3   ← 第 3 组
  ...
```

**打比方**：回到共享工具箱的例子。32 户人家分成 8 组，每组 4 户共享一个工具箱。既不像每家自备那么浪费，也不像全部共享那么拥挤。每组的 4 户人家用途相近，共享一套工具够用。

### 三者对比

```
┌──────────────────────────────────────────────────────────┐
│           MHA vs GQA vs MQA 的 KV 分布                    │
│                                                          │
│  n_heads = 8, g = 2 (GQA)                               │
│                                                          │
│  MHA (g=8):  Q1→K1  Q2→K2  Q3→K3  Q4→K4                │
│              Q5→K5  Q6→K6  Q7→K7  Q8→K8                │
│              共 8 组 KV → Cache = 8x                     │
│                                                          │
│  GQA (g=2):  Q1 Q2 Q3 Q4 → K1                            │
│              Q5 Q6 Q7 Q8 → K2                            │
│              共 2 组 KV → Cache = 2x                     │
│                                                          │
│  MQA (g=1):  Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 → K               │
│              共 1 组 KV → Cache = 1x                     │
└──────────────────────────────────────────────────────────┘
```

### 实现细节

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.n_heads = n_heads          # Q 的头数（如 32）
        self.n_kv_heads = n_kv_heads    # K, V 的头数（如 8）—— 这就是 "组数 g"
        self.d_head = d_model // n_heads
        self.n_rep = n_heads // n_kv_heads  # 每组有几个 Q 头（如 32/8 = 4）
        
        # Q: n_heads 份
        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        # K: 只有 n_kv_heads 份！
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        # V: 只有 n_kv_heads 份！
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        # 输出投影
        self.W_o = nn.Linear(n_heads * self.d_head, d_model, bias=False)
    
    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        
        # 投影
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        
        # 拼接缓存的 KV
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        new_kv = (K, V)
        
        # 🔑 关键步骤：扩展 K 和 V，让它们和 Q 的头数匹配
        # K: (B, n_kv_heads, T, d_head) → (B, n_heads, T, d_head)
        K = self._expand_kv(K)
        V = self._expand_kv(V)
        
        # 标准注意力计算
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        attn = F.softmax(scores, dim=-1)
        out = attn @ V
        
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out), new_kv
    
    def _expand_kv(self, x):
        """将 KV 从 n_kv_heads 扩展到 n_heads
        
        (B, n_kv_heads, T, d_head) → (B, n_heads, T, d_head)
        
        做法是将每个 KV 头重复 n_rep 次
        """
        B, n_kv_heads, T, d_head = x.shape
        # 先 reshape: (B, n_kv_heads, 1, T, d_head)
        # 再 expand: (B, n_kv_heads, n_rep, T, d_head)
        # 再 reshape: (B, n_kv_heads * n_rep, T, d_head) = (B, n_heads, T, d_head)
        x = x[:, :, None, :, :].expand(B, n_kv_heads, self.n_rep, T, d_head)
        return x.reshape(B, self.n_heads, T, d_head)
```

### `_expand_kv` 的直觉

这是 GQA 实现中最关键的一步。我们仔细看看它在做什么：

假设 n_heads = 8, n_kv_heads = 2, n_rep = 4：

```
扩展前（K 只有 2 个头）：
  K[0] → 被 Q[0], Q[1], Q[2], Q[3] 共享
  K[1] → 被 Q[4], Q[5], Q[6], Q[7] 共享

扩展后（K 变成 8 个头，通过重复）：
  K[0] → Q[0] 的 K
  K[0] → Q[1] 的 K（同一个 K 的副本）
  K[0] → Q[2] 的 K（同一个 K 的副本）
  K[0] → Q[3] 的 K（同一个 K 的副本）
  K[1] → Q[4] 的 K
  K[1] → Q[5] 的 K（同一个 K 的副本）
  K[1] → Q[6] 的 K（同一个 K 的副本）
  K[1] → Q[7] 的 K（同一个 K 的副本）
```

**注意**：expand 不会真的复制内存！它只是告诉 PyTorch "读取时假装这里有 n_rep 份"，实际数据只有一份。这就是为什么 KV Cache 可以省内存——缓存中只存 n_kv_heads 份，推理时通过 expand "展开"给所有 Q 头用。

---

## 📊 KV Cache 压缩效果对比

> ❓ **问题**：GQA 相比 MHA 和 MQA，到底省了多少？质量损失多少？

### 内存对比

假设模型配置：
- d_model = 4096, n_heads = 32, d_head = 128, L = 32 层
- 序列长度 T = 4096, batch_size = 1, float16

```
KV Cache 大小 = 2 × B × n_kv_heads × T × d_head × L × 2 bytes
```

| 方案 | n_kv_heads | KV Cache 大小 | 相对 MHA |
|------|-----------|--------------|----------|
| MHA | 32 | 2.0 GB | 1× |
| GQA (8 组) | 8 | 512 MB | 1/4× |
| GQA (4 组) | 4 | 256 MB | 1/8× |
| MQA | 1 | 64 MB | 1/32× |

### 实际模型中的应用

| 模型 | n_heads | n_kv_heads | 方案 |
|------|---------|-----------|------|
| GPT-3 | 96 | 96 | MHA |
| LLaMA-1 | 32 | 32 | MHA |
| LLaMA-2 (7B/13B) | 32 | 32 | MHA |
| LLaMA-2 (70B) | 64 | 8 | GQA (8 组) |
| LLaMA-3 (8B) | 32 | 8 | GQA (4 组) |
| LLaMA-3 (70B) | 64 | 8 | GQA (8 组) |
| Mistral (7B) | 32 | 8 | GQA (4 组) |
| PaLM (540B) | 48 | 1 | MQA |
| Gemini | - | - | GQA |
| DeepSeek-V2 | 128 | 16 | GQA (16 组) |

可以看到，**GQA 已经成为主流选择**，MQA 只在 Google 的一些超大模型中使用。

### 性能 vs 质量的 trade-off

Google 的 GQA 论文做了详细实验：

```
质量排名（从高到低）：
  MHA > GQA (g=8) > GQA (g=4) > GQA (g=2) > MQA

速度排名（从快到慢，推理阶段）：
  MQA > GQA (g=2) > GQA (g=4) > GQA (g=8) > MHA

内存排名（从小到大，KV Cache）：
  MQA < GQA (g=2) < GQA (g=4) < GQA (g=8) < MHA
```

但关键发现是：**GQA (g=8) 的质量非常接近 MHA**（通常只差 0.1-0.3% 的困惑度），但 KV Cache 只有 MHA 的 1/8。这个性价比让 GQA 成为了几乎所有现代 LLM 的标配。

> ❓ **问题**：为什么 g=8 是个好选择？8 组够用吗？

Google 的实验表明，8 组（对于 32-64 头的模型）是一个甜蜜点。原因可能在于：

1. **注意力的冗余性**：研究发现 MHA 中很多头学到的 K/V 模式高度相似，分组共享后并不会丢失太多信息
2. **Q 仍然提供多样性**：每组内不同头的 Q 不同，所以即使 K/V 相同，每个头关注的位置仍然不同
3. **训练时学到的"适应"**：模型在训练过程中会自动学习如何利用有限的 KV 资源

---

## 🔄 从已训练的 MHA 模型转换到 GQA

> ❓ **问题**：如果我已经用 MHA 训练好了一个模型，能转换成 GQA 吗？

这是 GQA 论文的一个重要贡献——**Uptraining**。不需要从头训练，可以从已有的 MHA 模型转换：

### 转换步骤

```
原始 MHA 模型有 n_heads 组独立的 W_k 和 W_v

转换为 GQA（n_kv_heads 组）：
1. 把 n_heads 个头的 W_k 分成 n_kv_heads 组
2. 每组内的 W_k 取均值（mean pooling）
3. 同理处理 W_v
4. 用转换后的模型继续训练一小段时间（原始预训练步数的 ~5%）
```

**打比方**：就像 32 个厨师各自研发了自己的菜谱（MHA 的 32 份 W_k/W_v），现在要把他们分成 8 组，每组只保留一份融合版菜谱。融合的方法就是取各家之长（均值），然后让每组的厨师再磨合几天（uptraining）。

### 为什么需要 Uptraining？

直接取均值后的模型性能会下降（因为丢失了一些头特有的信息），但只下降一点。继续训练 5% 的步数，模型就能适应新的 KV 结构，恢复大部分性能。

LLaMA-2-70B 就是这么做的——从 MHA 的检查点转换为 GQA，然后 uptraining。

---

## 🧮 KV Cache 对推理吞吐量的影响

> ❓ **问题**：KV Cache 减小了，推理吞吐量能提升多少？

推理时，GPU 的瓶颈通常是**内存带宽**（memory bandwidth），而不是计算能力。KV Cache 越大，每步从显存读取的数据越多，等待内存传输的时间越长。

### 内存带宽分析

每步 decode 时的内存读取量：

```
标准 MHA:
  读取模型参数: ~140 GB (LLaMA-70B)
  读取 KV Cache: ~10 GB
  总读取: ~150 GB

GQA (8 组):
  读取模型参数: ~140 GB
  读取 KV Cache: ~1.25 GB
  总读取: ~141 GB  ← 省了 6% 的读取量
  
  但更重要的是：KV Cache 变小 → GPU 显存能放更多 → batch_size 更大
  
  batch_size 从 1 → 8:
    MHA: 不可能（10 GB × 8 = 80 GB KV Cache 就超了）
    GQA: 轻松（1.25 GB × 8 = 10 GB KV Cache）
```

**结论**：GQA 的主要好处不是单次推理变快（差距不大），而是**允许更大的 batch_size**，从而大幅提升吞吐量。

### 真实性能数据

根据 vLLM 团队的测试（LLaMA-2-70B，A100 80GB）：

| 方案 | 最大 batch_size | 吞吐量 (tokens/s) |
|------|----------------|-------------------|
| MHA | 8 | ~800 |
| GQA (8 组) | 32 | ~2800 |
| MQA | 64 | ~4500 |

GQA 的吞吐量是 MHA 的 **3.5 倍**，主要归功于 batch_size 从 8 提升到 32。

---

## 🏗️ 代码实现：MHA / GQA / MQA 三合一

> ❓ **问题**：能不能用一套代码实现三种注意力？

可以！关键参数是 `n_kv_heads`：

```python
# MHA: n_kv_heads = n_heads
# GQA: 1 < n_kv_heads < n_heads
# MQA: n_kv_heads = 1

class UnifiedAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None):
        super().__init__()
        # 默认 n_kv_heads = n_heads → 退化为标准 MHA
        self.n_kv_heads = n_kv_heads or n_heads
        ...
```

今天的 `mqa_gqa.py` 包含完整的统一实现，以及三种方案的对比实验。

---

## 🔍 深入理解：为什么共享 K/V 不会严重影响质量？

> ❓ **问题**：直觉上，共享 K/V 会减少模型的"视角"，为什么实际影响这么小？

### 原因 1：Q 才是注意力的"驱动力"

在注意力机制中，Q 决定了"我在找什么"，K 决定了"我能被什么匹配到"。即使 K/V 被共享，不同的 Q 仍然可以让每个头关注不同的位置和模式。

想象你和一个朋友共享同一个图书馆（K/V 相同），但你们要找的书不同（Q 不同）。你找数学书，TA 找文学书——你们的"搜索路径"完全不同，即使图书馆是一样的。

### 原因 2：多头之间的冗余性

研究表明，Transformer 的多头注意力存在大量冗余。很多头学到的 K/V 模式高度相似（特别是在浅层），共享它们不会损失太多信息。

### 原因 3：训练时的自适应

当模型从头开始用 GQA 训练时，它会自动学习如何在有限的 KV 资源下最大化表达能力。就像压缩图片——如果知道只能用 100KB，编码器会聪明地分配 bit。

---

## 🧪 动手实验

今天的代码 `mqa_gqa.py` 包含：

1. **统一注意力实现**：一套代码支持 MHA / GQA / MQA，通过 `n_kv_heads` 参数切换
2. **KV Cache 内存对比**：计算三种方案在不同模型配置下的缓存大小
3. **推理速度对比**：实测三种方案在自回归生成中的速度差异
4. **注意力模式可视化**：看看共享 KV 后，不同头的注意力分布有什么变化
5. **expand 操作详解**：直观展示 KV 是如何被"复制"给多个 Q 头的

运行方式：
```bash
python3 mqa_gqa.py
```

---

## 🔑 关键洞察

1. **GQA 是 MHA 和 MQA 的中间方案**：通过参数 `n_kv_heads` 控制共享程度，在内存和质量之间取平衡
2. **KV Cache 缩减 n_heads / n_kv_heads 倍**：32 头的模型用 8 组 GQA，缓存只有原来的 1/4
3. **MQA 太激进，MHA 太浪费，GQA 刚刚好**：这就是为什么 LLaMA-2/3、Mistral、Gemini 都选择了 GQA
4. **主要收益是吞吐量而非延迟**：KV Cache 变小 → batch_size 可以更大 → 单位时间处理更多请求
5. **expand 不复制内存**：KV Cache 只存 n_kv_heads 份，推理时通过 view/expand "虚拟"展开
6. **Uptraining 可以从 MHA 转换**：不需要从头训练，均值池化 + 少量继续训练即可

---

## 📝 一句话总结

> GQA 让多个注意力头共享同一组 Key 和 Value，用极少的质量损失换取数倍的 KV Cache 压缩和推理吞吐量提升——这就是现代 LLM 推理加速的秘密武器。

## 📖 关键术语速查

| 术语 | 含义 |
|------|------|
| **MHA** | Multi-Head Attention，每个头有独立的 Q、K、V |
| **MQA** | Multi-Query Attention，所有头共享 1 组 K 和 V |
| **GQA** | Grouped-Query Attention，n_heads 个头分 g 组，每组共享 1 组 K 和 V |
| **n_kv_heads** | KV 的头数（组数），控制共享程度。= n_heads 为 MHA，= 1 为 MQA |
| **n_rep** | 每组有几个 Q 头，等于 n_heads / n_kv_heads |
| **expand** | 将 KV 从 n_kv_heads 扩展到 n_heads 的操作，不实际复制内存 |
| **Uptraining** | 从 MHA 模型均值池化 K/V 后继续训练，转换为 GQA 模型 |
| **Memory Bandwidth** | 内存带宽，推理时的主要瓶颈，KV Cache 越大读取越多 |

---

> 🤔 **今天留下的悬念**：GQA 解决了 KV Cache 的"空间"问题（缓存太大），但注意力计算本身的 O(N²) 时间复杂度还在——每步要对新 Q 和所有缓存 K 算点积，序列一长就很慢。有没有办法让注意力计算也加速？明天我们来看 **Flash Attention**——不改变数学结果，但通过巧妙的分块计算和 IO 优化，让注意力速度提升 2-4 倍，这就是 GPT-4 训练的秘密武器。

**下节预告**：Day 15 — Flash Attention，用"聪明地算"代替"算得快"。

---

*本课程代码开源于 [GitHub](https://github.com/LuChunQi/llm-from-scratch)，欢迎 Star ⭐*
