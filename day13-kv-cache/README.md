# Day 13: KV Cache — 让自回归生成快起来的"时间机器"

## 🔗 上节回顾

昨天我们搭建了完整的 Decoder-Only 架构，用 Causal Mask 给 Self-Attention 戴上了"眼罩"，让它只看过去、不看未来。我们还体验了自回归生成——模型一个词一个词地吐出来，每次把新词追加到输入末尾再重新送入模型。

但在实验中，你有没有注意到一个问题？每次生成一个新词，我们都要把**整个序列**重新喂给模型做一次完整的 forward pass。生成 1000 个词就要做 1000 次 forward pass，而且每次的计算量还在不断增长——因为序列越来越长了。

我们留了一个悬念：**有没有办法让推理也能快起来？**

今天我们就来拆解这个优化利器——**KV Cache**。

---

## 🐢 从一个问题开始：自回归生成到底慢在哪？

> ❓ **问题**：假设模型已经生成了 1000 个词，现在要生成第 1001 个。这一步到底在算什么？

让我们打开 Decoder-Only 的计算过程看个仔细。对于第 1001 步，模型需要：

```
输入: [token_1, token_2, ..., token_1000, token_1001]
```

在 Self-Attention 中，每个 token 都要：
1. 算 Q（Query）、K（Key）、V（Value）
2. Q 和所有 K 算注意力分数
3. 用注意力权重加权所有 V

但等等——**token_1 到 token_1000 的 K 和 V，上一步不是已经算过了吗？**

没错！这就是浪费的根源。第 1000 步已经算过 token_1~999 的 K 和 V，第 1001 步又把它们全部重新算了一遍。而且第 1002 步还会再算一遍，第 1003 步又再算一遍……

**打比方**：你在一间图书馆里查资料。每翻一页新的，就把之前翻过的所有页全部重新翻一遍。翻到第 1000 页时，你已经把前 999 页翻了 1000 遍了！这不疯了吗？

实际上，自回归生成中，**已生成 token 的 Key 和 Value 是不会变的**（因为输入没变，权重没变），只有最新 token 的 Q/K/V 需要计算。

> ❓ **问题**：既然旧的 K 和 V 不变，为什么不能存起来复用？

当然可以！这就是 KV Cache 的核心思想。

---

## 💡 KV Cache：把重复计算变成查表

### 核心直觉

> **Key 和 Value 算一次就够了，存下来，以后直接用。**

每次生成一个新 token 时：
- **新 token**：算它的 Q、K、V
- **K 和 V**：存入缓存
- **所有历史的 K 和 V**：从缓存里取，不用重算
- **用新 token 的 Q** 和**所有 K（包括缓存的）** 算注意力分数
- **用注意力权重** 加权**所有 V（包括缓存的）**

**打比方**：还是图书馆的例子。现在你有个笔记本，每翻一页就把关键信息摘抄下来（相当于缓存 K 和 V）。翻新页时，只需要看新页的内容，再结合笔记本上的旧笔记就行了——不用把旧页重新翻一遍。

### 具体流程对比

#### 无 KV Cache（朴素方式）

```
Step 1: 输入 [t1]
        计算 Q1, K1, V1        ← 第 1 次算 t1 的 K, V

Step 2: 输入 [t1, t2]
        计算 Q1, K1, V1        ← 第 2 次算 t1 的 K, V（重复！）
        计算 Q2, K2, V2
        Attention: Q2 对 [K1, K2]

Step 3: 输入 [t1, t2, t3]
        计算 Q1, K1, V1        ← 第 3 次算 t1 的 K, V（又重复！）
        计算 Q2, K2, V2        ← 第 2 次算 t2 的 K, V（重复！）
        计算 Q3, K3, V3
        Attention: Q3 对 [K1, K2, K3]

...

Step N: 输入 [t1, ..., tN]
        计算 Q1~QN 的 K, V     ← 全部重新计算！
```

总计算量（以 QK^T 运算为例）：

```
Step 1: 1 次点积
Step 2: 2 次点积 × 2 个 token = 4
Step 3: 3 次点积 × 3 个 token = 9
...
Step N: N^2 次点积

总计 ≈ 1 + 4 + 9 + ... + N^2 = O(N^3)
```

#### 有 KV Cache

```
Step 1: 输入 [t1]
        计算 Q1, K1, V1
        缓存 K1, V1

Step 2: 输入 [t2]（注意：只输入新 token！）
        计算 Q2, K2, V2
        缓存追加 K2, V2       ← 缓存现在是 [K1, K2], [V1, V2]
        Attention: Q2 对缓存中的 [K1, K2]

Step 3: 输入 [t3]
        计算 Q3, K3, V3
        缓存追加 K3, V3       ← 缓存现在是 [K1, K2, K3], [V1, V2, V3]
        Attention: Q3 对缓存中的 [K1, K2, K3]

...

Step N: 输入 [tN]
        计算 QN, KN, VN
        缓存追加 KN, VN
        Attention: QN 对缓存中的 [K1, ..., KN]
```

每步只有 1 个新 token 需要计算 QKV，和 N 个缓存 K 算注意力：

```
Step 1: 1 次点积
Step 2: 1 次点积（Q2 对 2 个 K）
Step 3: 1 次点积（Q3 对 3 个 K）
...
Step N: 1 次点积（QN 对 N 个 K）

总计 ≈ 1 + 2 + 3 + ... + N = O(N^2)
```

**从 O(N^3) 降到 O(N^2)**！这是一个量级的提升。生成 1000 个 token，计算量从约 10 亿次降到约 100 万次——快了 1000 倍（在 QKV 投影部分）。

> ❓ **问题**：等等，注意力分数的计算（Q 和 K 的点积）还是 O(N^2) 啊，因为每步都要对新 Q 和所有缓存 K 算分数。KV Cache 到底省了什么？

问得好！KV Cache 省的是**重复计算 QKV 投影**的开销。没有 KV Cache 时，每步要对所有历史 token 做完整的 QKV 线性投影（这是大头），而有了 KV Cache，每步只对 1 个新 token 做投影。注意力分数的计算确实是 O(N)，但 QKV 投影从 O(N) 降到 O(1)，这才是真正的节省。

### 数学详解

让我们更形式化地看看每一步的节省。

对于一个序列长度为 T 的生成任务，有 L 层 Transformer、H 个注意力头、d_model 维度：

**无 KV Cache：**

在第 t 步（生成第 t 个 token）：
- 输入所有 t 个 token
- 计算 t 个 token 的 Q, K, V：矩阵乘法，FLOPs ≈ 6 × t × d_model^2（三个投影矩阵，正反各一次）
- 总计 FLOPs for QKV projection across all steps: 6 × d_model^2 × (1 + 2 + ... + T) ≈ **3 × d_model^2 × T^2**

**有 KV Cache：**

在第 t 步：
- 输入 1 个新 token
- 计算 1 个 token 的 Q, K, V：FLOPs ≈ 6 × d_model^2
- 总计 FLOPs for QKV projection across all steps: 6 × d_model^2 × T = **6 × d_model^2 × T**

当 T = 1000 时，QKV 投影部分快了约 **500 倍**。

---

## 🏗️ KV Cache 的实现细节

> ❓ **问题**：KV Cache 听起来简单，但实际实现时有什么坑？

### 1. 缓存的数据结构

KV Cache 本质上就是一个不断增长的张量列表。在 PyTorch 中，通常这样管理：

```python
# 每一层的每个 Head 都有自己的 K 和 V 缓存
# 形状: (batch_size, n_heads, seq_len, d_head)
cache_k = torch.zeros(batch_size, n_heads, max_seq_len, d_head)
cache_v = torch.zeros(batch_size, n_heads, max_seq_len, d_head)

# 每次 forward 后更新缓存位置
cache_pos = 0  # 当前缓存到第几个位置

def update_cache(new_k, new_v):
    global cache_pos
    cache_k[:, :, cache_pos:cache_pos + new_k.shape[2], :] = new_k
    cache_v[:, :, cache_pos:cache_pos + new_v.shape[2], :] = new_v
    cache_pos += new_k.shape[2]
```

### 2. Prefill vs Decode 阶段

实际使用中，生成过程分两个阶段：

**Prefill（预填充）阶段**：处理用户输入的 prompt
- 输入整个 prompt（可能很长）
- 一次性计算所有 prompt token 的 K 和 V，存入缓存
- 这一步是**并行**的，因为所有 token 都已知

**Decode（解码）阶段**：逐个生成新 token
- 每步只输入 1 个新 token
- 计算 Q, K, V，更新缓存
- 用新 Q 和缓存中的所有 K 算注意力
- 这一步是**串行**的

**打比方**：Prefill 就像把一本书快速浏览一遍，把所有要点记在笔记本上。Decode 就像合上书，只靠笔记本上的笔记来写续集。

### 3. 多层缓存

每一层 Transformer 都有自己独立的 KV Cache！

```
Layer 0: cache_k[0], cache_v[0]
Layer 1: cache_k[1], cache_v[1]
...
Layer L-1: cache_k[L-1], cache_v[L-1]
```

为什么每层要独立缓存？因为每层的 K 和 V 含义不同：
- 浅层捕获语法、词法信息
- 深层捕获语义、逻辑信息

第 L 层的输入是第 L-1 层的输出，所以每层的 K 和 V 是不同的——不能跨层复用。

---

## 💾 空间换时间：KV Cache 的代价

> ❓ **问题**：KV Cache 这么好，有没有代价？

天下没有免费的午餐。KV Cache 用**空间换时间**——你省了计算，但多吃了内存。

### 内存开销计算

假设：
- 模型有 L 层，H 个注意力头，d_head 维度
- 序列长度 T
- 用 float16（2 bytes per value）

每个 token 在每层需要缓存的 K 和 V 大小：

```
K: 1 × H × d_head × 2 bytes
V: 1 × H × d_head × 2 bytes
总计: 2 × H × d_head × 2 = 4 × H × d_head bytes per layer per token
```

所有层、所有 token：

```
总缓存大小 = L × T × 4 × H × d_head bytes
           = L × T × 2 × d_model × 2 bytes  （因为 H × d_head = d_model）
           = 4 × L × T × d_model bytes
```

### 真实例子：LLaMA-2-70B

参数：
- L = 80 层
- d_model = 8192
- float16

对于 batch_size = 1，序列长度 T = 4096：

```
KV Cache 大小 = 4 × 80 × 4096 × 8192 bytes
             = 4 × 80 × 4096 × 8192
             = 10,737,418,240 bytes
             ≈ 10 GB
```

**10 GB！** 仅仅是为了缓存 KV。这比模型参数本身还大吗？LLaMA-70B 的参数约 140 GB（float16），所以 KV Cache 占了大约 7%。

但当 batch_size 增大时，KV Cache 线性增长。batch_size = 32 时，KV Cache 就要 320 GB——这已经不可接受了。

> ❓ **问题**：KV Cache 这么吃内存，能不能压缩？这就是明天要讲的 MQA 和 GQA——通过共享 Key 和 Value 来减少缓存大小。先留个印象。

### 内存 vs 计算：一个有趣的 trade-off

| 序列长度 | 无 Cache FLOPs | 有 Cache FLOPs | Cache 内存 |
|---------|---------------|----------------|-----------|
| 128 | 低 | 更低 | 很小 |
| 1024 | 中等 | 低 | 小 |
| 4096 | 高 | 中等 | 中等 |
| 32768 | 极高 | 较高 | 很大 |
| 128K | 无法承受 | 高 | 巨大 |

结论：**短序列时 KV Cache 省的有限，长序列时 KV Cache 是必需品**。对于 128K 上下文的长文本应用，没有 KV Cache 根本跑不动。

---

## 🔧 实现：带 KV Cache 的注意力层

> ❓ **问题**：如何在代码中实现 KV Cache？让我们一步步来。

### 关键变化：新增 `past_kv` 参数

```python
class CausalSelfAttentionWithCache(nn.Module):
    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        
        Q = self.W_q(x)  # (B, T, C)
        K = self.W_k(x)  # (B, T, C)
        V = self.W_v(x)  # (B, T, C)
        
        # 拆分多头
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        
        # 🔑 关键步骤：拼接缓存的 K 和 V
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)  # 在序列维度拼接
            V = torch.cat([past_v, V], dim=2)
        
        # 保存新的 KV 缓存（返回给外部管理）
        new_kv = (K, V)
        
        # 注意力计算（和之前一样）
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        
        # Causal Mask：注意 mask 的大小变了！
        # Q 的长度是 T（新 token 数），K 的长度是 past_len + T
        # mask 形状应该是 (T, past_len + T)
        # 新 token 之间仍然需要 causal mask
        # 新 token 对所有旧 token 不需要 mask（旧 token 都在前面）
        S = K.shape[2]  # 总序列长度 = past_len + T
        causal_mask = torch.triu(torch.ones(T, S, device=x.device), diagonal=S - T + 1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ V
        
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out), new_kv
```

### Causal Mask 的变化

注意上面 `causal_mask` 的计算——这是 KV Cache 实现中最容易出错的地方。

没有缓存时，mask 就是简单的下三角矩阵 `(T, T)`。

有缓存时，K 的长度变成了 `past_len + T`。新 token 的 Q 需要能看到所有旧 token（它们都在"过去"），但新 token 之间仍然要遵守因果约束。

**举个例子**：

假设缓存中有 3 个 token（pos 0, 1, 2），现在输入 2 个新 token（pos 3, 4）：

```
Q 来自 pos 3, 4
K 来自 pos 0, 1, 2, 3, 4

Causal Mask 应该是：
        pos0  pos1  pos2  pos3  pos4
pos3  [  ✓     ✓     ✓     ✓     ✗  ]
pos4  [  ✓     ✓     ✓     ✓     ✓  ]

用 torch.triu 生成：
diagonal = S - T + 1 = 5 - 2 + 1 = 4
mask[i][j] = True 当 j >= 4
即只遮住 pos3→pos4 的"偷看"
```

不过在实际推理中，每步通常只输入 1 个新 token（T=1），此时 `diagonal = S - 1 + 1 = S`，mask 全为 False——因为 1 个新 token 看所有旧 token 没有任何因果问题。

### Prefill 阶段的 Mask

Prefill 时，T 可能很大（整个 prompt），past_len = 0。这时 mask 退化为普通的下三角矩阵。

---

## 📊 实测：有/无 KV Cache 的性能对比

> ❓ **问题**：KV Cache 到底能快多少？让我们用代码实测。

### 计算量对比

假设生成 T 个 token，模型有 L 层，d_model 维度。

**无 Cache 的 QKV 投影 FLOPs**：

```
每步: 6 × t × d_model^2（t 为当前序列长度）
总计: 6 × d_model^2 × Σ(t=1 to T) t = 3 × d_model^2 × T × (T+1)
```

**有 Cache 的 QKV 投影 FLOPs**：

```
每步: 6 × 1 × d_model^2（始终只算 1 个 token）
总计: 6 × d_model^2 × T
```

**加速比** ≈ (T+1)/2 ≈ T/2

所以：
- 生成 100 个 token：约快 50 倍（QKV 投影部分）
- 生成 1000 个 token：约快 500 倍
- 生成 4000 个 token：约快 2000 倍

**打比方**：就像从"每次都从头算"变成了"增量计算"——每次只算新增的那一点点，然后和之前的结果拼起来。

### 不仅仅是 QKV

KV Cache 的好处不仅限于省 QKV 投影。整个 forward pass 中，没有 KV Cache 时：
- Token Embedding：需要查 T 个 token 的 embedding
- 所有层的 Attention：需要算 T 个 token 的 QKV
- FFN：需要对 T 个 token 做前馈计算
- LM Head：需要把 T 个 token 的表示投影回词表

有 KV Cache 时：
- Token Embedding：只需查 1 个 token
- 所有层的 Attention：只需算 1 个 token 的 QKV（K 和 V 拼接缓存）
- FFN：只需处理 1 个 token
- LM Head：只需投影 1 个 token

**每一步的计算量从 O(T) 降到了 O(1)**（QKV 投影和 FFN 部分），只有注意力分数计算仍然是 O(T)（因为要和所有缓存 K 算点积）。

---

## 🌊 KV Cache 和 PagedAttention

> ❓ **问题**：KV Cache 的内存管理还能更高效吗？

### 问题：内存碎片

传统的 KV Cache 管理方式是预分配一个大的连续内存块（max_seq_len 大小）。但实际使用中：
- 大部分序列不会达到最大长度 → **内存浪费**
- 不同序列长度不同 → **碎片化**
- 频繁创建/删除序列 → **分配/释放开销**

**打比方**：就像给每个客人预订一个最大号的房间，不管客人实际上只住一晚还是住一个月。大部分房间都空着，但新的客人来了却可能没房了。

### vLLM 的 PagedAttention

vLLM 团队提出了 **PagedAttention**，借鉴了操作系统的虚拟内存分页机制：

1. 把 KV Cache 切成固定大小的"页"（block），比如每页存 16 个 token 的 KV
2. 用一个页表（block table）映射逻辑位置到物理页
3. 新 token 来了就分配新页，不需要连续的物理内存
4. 序列结束时回收页

好处：
- **近乎零浪费**：只有最后一个页可能有少量空闲
- **灵活共享**：不同序列可以共享相同的 KV 页（比如 beam search）
- **更大 batch size**：省下的内存可以服务更多请求

这个我们会在 Day 40（vLLM 深度解析）中详细展开。

---

## 🔍 KV Cache 的几个细节

### 1. Batch 推理中的 KV Cache

当同时服务多个请求时，每个请求有自己独立的 KV Cache。这就需要精心管理内存——谁的缓存到哪了，谁的位置在哪。

Hugging Face 的 `generate()` 方法中，`past_key_values` 就是一个 tuple of tuples：

```python
# past_key_values 的结构：
# (
#   (layer_0_key, layer_0_value),
#   (layer_1_key, layer_1_value),
#   ...
#   (layer_L_key, layer_L_value),
# )
```

### 2. KV Cache 和 Beam Search

Beam Search 维护多个候选序列。如果每个候选都独立缓存 KV，内存会成倍增长。

优化：多个 beam 可以共享"共同前缀"的 KV Cache。比如 beam 1 是 "The cat sat"，beam 2 是 "The cat ran"，它们可以共享 "The cat" 的 KV Cache。

### 3. 什么时候清空 KV Cache？

- 用户发来新的对话（不续接上一轮）：清空
- 序列达到最大长度：清空（或者用滑动窗口只保留最近的 KV）
- 推理框架（如 vLLM）会在序列结束后自动回收

---

## 🧪 动手实验

今天的代码 `kv_cache.py` 包含：

1. **朴素自回归 vs KV Cache 自回归对比** — 亲手实现两种方式，对比结果和速度
2. **KV Cache 内存占用估算** — 计算不同模型在不同序列长度下的缓存大小
3. **带 KV Cache 的完整注意力层** — 实现一个支持 past_kv 的注意力模块
4. **实际推理速度对比** — 用 MiniGPT 实测有/无 Cache 的性能差异
5. **Prefill + Decode 两阶段演示** — 理解真实推理流程的两个阶段

运行方式：
```bash
python3 kv_cache.py
```

---

## 🔑 关键洞察

1. **KV Cache 的本质**：自回归生成中，历史 token 的 K 和 V 不会变，所以算一次存起来就行——用空间换时间
2. **计算量从 O(T^3) 降到 O(T^2)**：QKV 投影从每步 O(T) 降到 O(1)，是性能提升的主要来源
3. **每层独立缓存**：每一层都有自己的 KV Cache，不能跨层共享
4. **Prefill + Decode**：Prefill 并行处理 prompt 填充缓存，Decode 逐个生成利用缓存
5. **内存是代价**：对于长序列和大模型，KV Cache 可能占用 GB 级内存
6. **PagedAttention 优化内存管理**：vLLM 用分页机制管理 KV Cache，大幅减少内存浪费

---

## 📝 一句话总结

> KV Cache 的思想极其简单——把算过的 Key 和 Value 存起来复用，避免重复计算——但这个简单的优化让自回归生成的速度提升了数百倍，是 LLM 推理不可或缺的基石。

## 📖 关键术语速查

| 术语 | 含义 |
|------|------|
| **KV Cache** | 缓存每层注意力中历史 token 的 Key 和 Value，避免重复计算 |
| **Prefill** | 预填充阶段，并行处理 prompt 的所有 token，填充 KV Cache |
| **Decode** | 解码阶段，逐步生成新 token，每步利用缓存的 KV |
| **PagedAttention** | vLLM 提出的分页式 KV Cache 管理机制，减少内存碎片 |
| **O(N^3) vs O(N^2)** | 无 Cache 和有 Cache 的总计算量级别 |
| **Batch Inference** | 批量推理，多个请求共享 GPU，每个请求有独立 KV Cache |
| **Beam Search** | 束搜索，维护多个候选序列，可共享前缀的 KV Cache |

---

> 🤔 **今天留下的悬念**：KV Cache 虽好，但内存开销惊人——LLaMA-2-70B 在 4K 序列长度下，单个请求的 KV Cache 就要 10 GB。batch_size 一大，内存直接爆炸。有没有办法压缩 KV Cache？明天我们来看 **MQA（Multi-Query Attention）和 GQA（Grouped-Query Attention）**——通过让多个 Head 共享同一组 K 和 V，把缓存大小砍掉几倍甚至几十倍。这就是 LLaMA-2 和 GPT-4 的秘密武器。

**下节预告**：Day 14 — MQA & GQA，让 KV Cache 瘦身。

---

*本课程代码开源于 [GitHub](https://github.com/LuChunQi/llm-from-scratch)，欢迎 Star ⭐*
