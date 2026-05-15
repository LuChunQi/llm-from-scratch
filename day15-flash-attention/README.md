# Day 15: Flash Attention — 不改变数学，却让注意力快 2-4 倍的魔法

## 🔗 上节回顾

昨天我们拆解了 MQA 和 GQA——通过让多个注意力头共享同一组 Key 和 Value，KV Cache 被压缩了 n_heads / n_kv_heads 倍。LLaMA-2-70B 用 GQA 把 KV Cache 从 10 GB 降到 1.25 GB，推理吞吐量提升 3.5 倍。

但我们也意识到一个更深层的问题：**即使缓存了 KV，注意力计算的 O(N²) 时间复杂度仍然存在**——每一步 decode，新的 Q 要和所有历史 K 算点积。序列一长，计算量和显存读写都是平方级增长。

今天我们来看一个更精妙的方案——**Flash Attention**。它不改变注意力机制的数学定义，输出的每一个数值都和标准实现一模一样，但通过"聪明地安排计算顺序"，让速度提升 2-4 倍，显存占用从 O(N²) 降到 O(N)。

> 这不是魔法，这是对硬件的深刻理解。

---

## 🏭 从工厂流水线说起：为什么"算得对"不等于"算得快"

> ❓ **问题**：注意力公式的数学很简单，GPU 算力这么强，为什么还是慢？

假设你开了一家家具厂（GPU），生产"注意力牌"沙发。

**原料**：Q（客户需求单）、K（库存清单）、V（实际家具）

**生产流程（标准实现）**：
1. 把 K 的整个仓库搬到工厂大厅（读 K 到 SRAM）
2. Q 和 K 做点积 → 得到一张巨大的"匹配度评分表" S（N×N 矩阵）
3. 把评分表 S 存到仓库（写 S 到 HBM）
4. 从仓库搬回来（读 S）
5. 对 S 做 softmax → 得到注意力权重 P（N×N 矩阵）
6. 把 P 存到仓库（写 P 到 HBM）
7. 从仓库搬回来（读 P）
8. P 和 V 相乘 → 得到最终产品 O

看到问题了吗？**N×N 的中间矩阵 S 和 P 需要在工厂大厅和仓库之间来回搬 4 次**。

### GPU 内存层次：为什么"搬东西"这么要命

```
┌──────────────────────────────────────────────────┐
│              GPU 内存层次                          │
│                                                  │
│  SRAM（片上缓存）                                 │
│  ├── 大小：约 20 MB（A100）                       │
│  ├── 带宽：~19 TB/s                              │
│  └── 延迟：~1 ns                                 │
│  ↑ 超快，但太小了                                 │
│                                                  │
│  HBM（高带宽内存，就是常说的"显存"）               │
│  ├── 大小：40-80 GB                              │
│  ├── 带宽：~2 TB/s                               │
│  └── 延迟：~100 ns                               │
│  ↑ 够大，但慢 10 倍                              │
└──────────────────────────────────────────────────┘
```

**关键数字**：SRAM 的带宽是 HBM 的 **~10 倍**。所以同样大小的数据，从 SRAM 读比从 HBM 读快 10 倍。

但 SRAM 只有 20 MB。一个序列长度 4096、64 头的注意力，中间矩阵 S 和 P 的大小是：

```
S: 64 × 4096 × 4096 × 4 bytes (float32) = 4 GB
P: 同样 4 GB
```

4 GB >> 20 MB。中间矩阵根本放不进 SRAM，只能放在 HBM 里，每步都要慢吞吞地从 HBM 读写。

> ❓ **问题**：能不能不存这些巨大的中间矩阵 S 和 P？

这就是 Flash Attention 要解决的核心问题。

---

## 💡 核心洞察：分块计算 + 在线 Softmax

> ❓ **问题**：Softmax 需要看到整行数据才能算（分母是所有 exp 的和），怎么"分块"？

标准注意力公式：

```
S = Q × K^T                    # (N, N) 的评分矩阵
P = softmax(S, dim=-1)         # 每行归一化 → (N, N)
O = P × V                      # (N, d) 的输出
```

Softmax 的定义是：

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

问题在于分母 `Σ_j exp(x_j)`——它需要遍历整行所有元素。如果你只看到行的一部分（一个"块"），你算不出正确的 softmax。

这就好比：**你知道全班 5 个同学的分数，要算每个人的分数占比。如果老师一次只给你看 2 个人的分数，你怎么算？**

### 在线 Softmax（Online Softmax）——关键突破

2018 年，Milakov 等人提出了"在线 Softmax"算法，它的核心思想是：**可以逐块更新 softmax 结果，不需要先看完所有数据**。

原理：

```
假设你已经处理了前 k 个元素：
  running_max = max(x_1, ..., x_k)
  running_sum = Σ_{j=1}^{k} exp(x_j - running_max)

现在来了第 k+1 个元素 x_{k+1}：
  new_max = max(running_max, x_{k+1})
  
  更新 running_sum：
  running_sum = running_sum × exp(running_max - new_max) + exp(x_{k+1} - new_max)
  
  这里的 exp(running_max - new_max) 是"修正因子"——
  因为 max 变了，之前算的 exp 都要调整。
```

**打比方**：你在算班级平均分。一开始以为最高分是 90，后来发现有个 95 的同学。不用重新算所有人的分数——只需要把之前的数据"校正"一下（因为 max 变了，所有 exp 值都偏了），再加上新同学的分数就行。

### Flash Attention 的算法

有了在线 Softmax，Flash Attention 的流程就清楚了：

```
把 Q、K、V 切成小块（block），每块大小 B_r、B_c

初始化：
  O = 0（输出）
  l = 0（每行的 exp 之和）
  m = -∞（每行的最大值）

外层循环：遍历 K、V 的块
  for each block (K_j, V_j):

    内层循环：遍历 Q 的块
    for each block Q_i:
      
      1. 算当前块的注意力分数
         S_ij = Q_i × K_j^T          # 只有小块，放得进 SRAM！
      
      2. 算当前块的行最大值
         m_ij = rowmax(S_ij)
      
      3. 算当前块的 exp 和
         P_ij = exp(S_ij - m_ij)     # 用当前块的 max 做数值稳定
         l_ij = rowsum(P_ij)
      
      4. 更新全局统计量
         m_new = max(m_i, m_ij)      # 新的全局 max
         l_new = exp(m_i - m_new) * l_i + exp(m_ij - m_new) * l_ij
                ↑ 校正旧的 sum       ↑ 加上新的 sum
      
      5. 更新输出
         O_i = (l_i * exp(m_i - m_new) * O_i + P_ij * V_j) / l_new
               ↑ 校正旧的输出          ↑ 加上新的贡献
```

**关键**：整个过程只需要在 SRAM 中保持 Q_i、K_j、V_j、S_ij、P_ij 这些小块。中间结果 S 和 P 从来不写入 HBM！

### 数学等价性

通过在线 Softmax 的校正因子，最终算出的 O 和标准实现**完全一致**——逐元素相等，没有任何近似。

```
标准实现：
  S = Q × K^T
  P = softmax(S)
  O = P × V

Flash Attention：
  逐块计算，但通过 online softmax 校正
  最终 O = softmax(Q × K^T) × V  ← 完全相同！
```

**注意**：这里说的是 **Flash Attention（v1/v2）**，不是 Flash Attention 的近似版本。有些论文提出的"Linear Attention"或"Performer"等是近似方法，会改变数学结果。Flash Attention 是精确的。

---

## 📊 为什么 Flash Attention 更快？

> ❓ **问题**：Flash Attention 的计算量并没有减少（还是 N² 次乘法），为什么实际更快？

答案是：**减少了对 HBM 的读写次数**。

### IO 分析

```
标准注意力的 HBM 读写：
  读 Q:       N × d           （从 HBM 读 Q）
  读 K:       N × d           （从 HBM 读 K）
  写 S:       N × N           （QK^T 结果写回 HBM）
  读 S:       N × N           （读回来做 softmax）
  写 P:       N × N           （softmax 结果写回 HBM）
  读 P:       N × N           （读回来乘 V）
  读 V:       N × d           （从 HBM 读 V）
  写 O:       N × d           （最终输出）
  
  总 HBM 读写 ≈ 4N² + 4Nd ≈ 4N²（当 N >> d 时）

Flash Attention 的 HBM 读写：
  读 Q:       N × d           （从 HBM 分块读 Q）
  读 K:       N × d           （从 HBM 分块读 K）
  读 V:       N × d           （从 HBM 分块读 V）
  写 O:       N × d           （最终输出写回 HBM）
  
  总 HBM 读写 ≈ 4Nd ≈ O(Nd)
```

**比例**：标准实现的 IO 是 O(N²)，Flash Attention 的 IO 是 O(Nd)。当 N = 4096, d = 128 时：

```
标准：4 × 4096² = 67M 次读写
Flash：4 × 4096 × 128 = 2M 次读写

快 33 倍！
```

当然，这是纯 IO 的理论分析。实际中 Flash Attention 比标准实现快 2-4 倍（因为计算本身也需要时间，而且 GPU 有计算/IO 重叠优化）。但 2-4 倍的加速对于训练和推理都是巨大的——训练时间直接缩短一半以上。

### 显存从 O(N²) 降到 O(N)

标准实现需要存 N×N 的 S 和 P 矩阵（用于反向传播），显存占用 O(N²)。

Flash Attention 不需要存这些中间矩阵——通过**重计算（recomputation）**在反向传播时重新算一遍 S 和 P（比从 HBM 读还快）。显存从 O(N²) 降到 O(N)。

```
N = 8192, d = 128, n_heads = 32, float16

标准注意力中间矩阵：
  S: 32 × 8192 × 8192 × 2 bytes = 4 GB
  P: 同样 4 GB
  总计：8 GB（而且必须同时存在于显存中！）

Flash Attention：
  只存 Q, K, V, O：32 × 8192 × 128 × 2 bytes × 4 ≈ 256 MB
  总计：~256 MB
```

从 8 GB 降到 256 MB，**省了 30 倍显存**。这意味着：
- 可以用更长的序列（从 2K → 8K → 32K）
- 可以用更大的 batch_size
- 或者两者兼得

---

## 🔄 Flash Attention 2：更快更优雅

> ❓ **问题**：Flash Attention 已经很快了，v2 又改进了什么？

Flash Attention 2（2023 年，Tri Dao）进一步优化了两个关键点：

### 改进 1：减少非矩阵乘法操作

GPU 有专门的矩阵乘法单元（Tensor Core），速度极快。但 softmax、masking 等操作不能用 Tensor Core，只能用普通 CUDA Core，慢很多。

Flash Attention v1 在内层循环中做了很多"非矩阵乘法"操作（更新 m、l 等）。v2 重新安排了循环顺序，把矩阵乘法集中在 SRAM 中批量完成，减少了中间的非矩阵乘法操作。

### 改进 2：更好的并行性

v1 按序列长度 N 做外层并行——当 batch_size 和 n_heads 不够大时，GPU 的 SM（流式多处理器）会空闲。

v2 在 Q 的维度上做并行，让每个线程块处理一段 Q 的行。这样即使 batch_size = 1，也能充分利用 GPU。

```
v1：外层循环 K, V → 内层循环 Q
  → 并行维度：batch × n_heads
  → 问题：batch=1 时只有 n_heads 个并行任务

v2：外层循环 Q → 内层循环 K, V
  → 并行维度：batch × n_heads × (N / block_size)
  → 即使 batch=1, n_heads=32, N=4096, block=128
  → 并行任务 = 32 × 32 = 1024 个！
```

**结果**：Flash Attention 2 比标准注意力快 **2-4 倍**（比 v1 又快约 50%）。

---

## 🧩 Flash Attention 的反向传播

> ❓ **问题**：前向传播可以分块，反向传播怎么办？

反向传播需要 S 和 P 的值来计算梯度。标准做法是把这些中间矩阵存在显存中——这就是为什么训练时显存爆炸。

Flash Attention 的策略是：**不存中间矩阵，反向时重新计算**。

```
反向传播流程：
1. 读入 Q, K, V, O（前向的输出，这些必须存）
2. 读入 dO（来自上一层的梯度）
3. 用 O 和 dO 快速算出 dP = dO × V^T  ← 不需要 P 本身！
4. 重计算 S = Q × K^T（分块重计算，在 SRAM 中）
5. 用 S 和 P 重计算 softmax 的梯度
6. 计算 dQ, dK, dV
```

**为什么不存中间矩阵更快？**

因为从 HBM 读写 N×N 的矩阵比重新算一遍还慢！

```
从 HBM 读 S:  N × N × 4 bytes 的带宽消耗
重计算 S:     Q × K^T 在 SRAM 中算，利用 Tensor Core

当 N 大时，计算速度 > IO 速度
所以"重计算"比"读缓存"更快
```

这颠覆了很多人的直觉——通常缓存是为了避免重复计算，但在 GPU 上，有时候重计算比读写缓存更快。这就是 **"算术密度"（arithmetic intensity）** 的概念。

---

## 🎯 实际应用：Flash Attention 无处不在

Flash Attention 已经成为所有主流 LLM 训练和推理框架的标配：

| 框架/模型 | 使用方式 |
|-----------|---------|
| PyTorch 2.0+ | `torch.nn.functional.scaled_dot_product_attention`（内置 Flash Attention） |
| Hugging Face Transformers | 自动使用 Flash Attention（如果检测到兼容硬件） |
| vLLM | 默认 Flash Attention |
| LLaMA 训练 | 使用 Flash Attention |
| GPT-4 训练 | 使用 Flash Attention |
| Megatron-LM | 使用 Flash Attention |

你不需要手写 Flash Attention——PyTorch 2.0+ 已经内置了，只需一行代码：

```python
# PyTorch 2.0+ 内置的 Flash Attention
import torch.nn.functional as F

output = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
# PyTorch 会自动选择 Flash Attention（如果硬件支持）
```

---

## 🔍 理解"算术密度"

> ❓ **问题**：为什么 GPU 上"重计算"有时比"读缓存"更快？

这是理解 Flash Attention（以及很多 GPU 优化）的关键概念。

**算术密度** = 计算量 / IO 量

```
矩阵乘法 C = A × B（M×K × K×N → M×N）：
  计算量：2 × M × K × N FLOPs
  IO 量：读取 A + 读取 B + 写入 C = (M×K + K×N + M×N) × bytes
  
  当 M, N, K 很大时：
    算术密度 ≈ 2MNK / (MN + NK + KM) ≈ 2K（假设 M ≈ N）
    → 很高！计算远多于 IO → 矩阵乘法是"计算密集型"

Softmax：
  计算量：3N FLOPs（exp + sum + div）
  IO 量：3N reads/writes
  
  算术密度 ≈ 1
  → 很低！每个数据只做一两个操作 → softmax 是"IO 密集型"
```

**核心洞察**：GPU 的 Tensor Core 算矩阵乘法非常快（每秒几百 TFLOPS），但读写 HBM 相对慢（每秒几 TB）。所以如果一个操作的计算量远大于 IO 量（高算术密度），GPU 的计算能力就充分发挥了。如果一个操作 IO 量很大而计算量小（低算术密度），那大部分时间都在等数据搬运。

Flash Attention 的策略就是把计算密集的矩阵乘法集中在 SRAM 里做，避免低算术密度的大矩阵读写。

---

## 🏗️ 代码实现：模拟 Flash Attention 的分块计算

> ❓ **问题**：真正的 Flash Attention 需要用 CUDA 写，但能先用 Python 理解核心思想吗？

当然可以！今天的 `flash_attention.py` 包含：

1. **标准注意力实现**：教科书式版本，作为对照
2. **分块注意力实现**：用 Python 模拟 Flash Attention 的分块计算流程
3. **数值验证**：确认分块实现和标准实现的输出完全一致
4. **IO 量估算**：计算标准实现 vs 分块实现的 HBM 读写量
5. **显存对比**：展示中间矩阵的显存占用
6. **速度对比**：对比 PyTorch 内置的 SDPA（包含 Flash Attention）和手动实现

运行方式：
```bash
python3 flash_attention.py
```

---

## 🔑 关键洞察

1. **Flash Attention 不改变数学结果**——输出和标准注意力完全一致，没有任何近似
2. **核心思想是减少 HBM 读写**——通过分块计算和在线 Softmax，避免存储 N×N 的中间矩阵
3. **在线 Softmax 是关键**——允许逐块更新 softmax 结果，通过"修正因子"保持数学等价性
4. **显存从 O(N²) 降到 O(N)**——不再存储 S 和 P，反向传播时重计算
5. **"重计算比读缓存更快"**——在 GPU 上，计算是廉价的，IO 是昂贵的
6. **已经成为行业标配**——PyTorch 2.0+ 内置，所有主流框架默认使用
7. **Flash Attention 2 更快**——改进了并行策略和非矩阵乘法操作的占比

---

## 📝 一句话总结

> Flash Attention 不改变注意力机制的任何数学定义，只是通过分块计算把中间矩阵留在 GPU 的快速 SRAM 中，用更少的 HBM 读写换来 2-4 倍的速度提升和 O(N) 的显存占用——这是对"计算快不如 IO 少"的完美诠释。

## 📖 关键术语速查

| 术语 | 含义 |
|------|------|
| **Flash Attention** | 分块计算注意力的算法，不改变数学结果，减少 HBM 读写 |
| **SRAM** | GPU 片上缓存，~20MB，带宽 ~19TB/s，超快但很小 |
| **HBM** | 高带宽内存（显存），40-80GB，带宽 ~2TB/s，够大但相对慢 |
| **在线 Softmax** | 逐块更新 softmax 的算法，通过修正因子保持数值等价 |
| **分块（Tiling）** | 把大矩阵切成小块，每次只在 SRAM 中处理一小块 |
| **重计算（Recomputation）** | 不存中间结果，需要时重新计算——在 GPU 上比读缓存更快 |
| **算术密度** | 计算量 / IO 量，越高越能发挥 GPU 的计算能力 |
| **Tensor Core** | GPU 专门的矩阵乘法单元，做矩阵乘法极快 |
| **SDPA** | Scaled Dot-Product Attention，PyTorch 内置的注意力函数，自动使用 Flash Attention |

---

> 🤔 **今天留下的悬念**：前 15 天我们一块一块拆解了 Transformer 的每个零件——Self-Attention、多头、位置编码、FFN、LayerNorm、残差、Decoder-Only、KV Cache、GQA、Flash Attention。每个零件都理解了，但它们是怎么拼在一起变成一个完整模型的？训练的过程又是什么样的？明天我们进入第三模块，从**预训练**开始——用一个小数据集从头训练一个玩具级 GPT，看看 Transformer 从随机权重到能生成文本的神奇过程。

**下节预告**：Day 16 — 预训练，让一个模型从零开始学会"说人话"。

---

*本课程代码开源于 [GitHub](https://github.com/LuChunQi/llm-from-scratch)，欢迎 Star ⭐*
