# Day 21: QLoRA & LoRA 变体 — 量化与低秩的化学反应

## 🔗 上节回顾

昨天我们学习了 LoRA——用低秩分解的方式，给模型打一个"小补丁"。冻结原始权重，只训练两个小矩阵 A 和 B，用不到 0.1% 的参数就能近似全量微调的效果。但 LoRA 解决的只是"训练参数太多"的问题，模型权重本身还是 FP16 的 14GB——一张 8GB 显存的消费级显卡仍然塞不下。

今天我们就来解决最后这个障碍——**QLoRA**，把模型从 16-bit 压缩到 4-bit 存储，再配合 LoRA 微调。7B 模型只需 ~6GB 显存，消费级显卡就能跑微调。同时我们会一览 LoRA 家族的其他重要变体。

---

## 🔢 先搞懂量化：从 FP16 到 INT4

> ❓ **问题**：什么叫"量化"？为什么一个 FP16 的权重能"压缩"到 4-bit？

量化（Quantization）就是把数值的精度从高降低。就像把一张高清照片压缩成 JPEG——看起来差不多，但文件小了很多。

### 精度对比

```
FP32（32-bit 浮点）：  一个数用 32 位表示
  范围：±3.4 × 10^38
  精度：约 7 位有效数字

FP16（16-bit 浮点）：  一个数用 16 位表示
  范围：±6.5 × 10^4
  精度：约 3 位有效数字

BF16（16-bit 浮点）：  一个数用 16 位表示
  范围：±3.4 × 10^38（和 FP32 一样大）
  精度：约 2 位有效数字
  → 牺牲精度换范围，训练时更稳定

INT8（8-bit 整数）：    一个数用 8 位表示
  范围：-128 到 127（或 0 到 255）

INT4（4-bit 整数）：    一个数用 4 位表示
  范围：-8 到 7（或 0 到 15）

一个 7B 模型的存储大小：
  FP32：  7B × 4 字节 = 28 GB
  FP16：  7B × 2 字节 = 14 GB
  INT8：  7B × 1 字节 =  7 GB
  INT4：  7B × 0.5 字节 = 3.5 GB  ← 这才是消费级显卡能塞下的！
```

### 线性量化的基本原理

最简单的量化方式是"线性映射"：

```
FP16 到 INT8 的映射：

1. 找到权重矩阵的最大值和最小值
   例如：min = -3.2, max = 5.6

2. 计算缩放因子 scale 和零点 zero_point
   scale = (max - min) / (2^8 - 1) = 8.8 / 255 ≈ 0.0345
   zero_point = round(-min / scale) = round(92.8) = 93

3. 量化：INT8 值 = round(FP16 值 / scale) + zero_point
   例如：3.0 → round(87.0) + 93 = 180
        -1.5 → round(-43.5) + 93 = 50

4. 反量化：FP16 值 = (INT8 值 - zero_point) × scale
   例如：180 → (180 - 93) × 0.0345 = 3.001 ≈ 3.0 ✅
        50  → (50 - 93) × 0.0345 = -1.484 ≈ -1.5 ✅

→ 会有微小误差（量化误差），但通常不影响模型质量太多
```

---

## 🧊 QLoRA 的核心创新：不是普通量化

> ❓ **问题**：直接把模型量化到 INT4 再加 LoRA，不就完了吗？为什么 QLoRA 还需要特殊设计？

问题没那么简单。普通 INT4 量化的精度损失太大，直接用来训练会导致微调效果显著下降。QLoRA 论文（Dettmers et al., 2023）提出了三个关键创新来解决这个问题：

### 创新 1：NF4（4-bit NormalFloat）

普通量化假设数值均匀分布，但神经网络权重实际接近正态分布。NF4 专门为正态分布设计了量化级别：

```
普通 INT4 量化：  -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
                  → 均匀间隔，但权重大部分集中在 0 附近，浪费了精度

NF4 量化：        16 个量化级别，按正态分布的分位数排列
                  → 在 0 附近更密集，在两端更稀疏
                  → 完美匹配权重的实际分布
                  → 同样 4-bit，但精度更高！

类比：
  普通量化 = 用均匀的刻度尺量温度（-40°C 到 50°C 每格 5°C）
  NF4     = 在常温区域（0-30°C）刻度更密，极端温度刻度更疏
            → 因为我们最关心的是常温范围
```

### 创新 2：双重量化（Double Quantization）

量化本身也有"元数据"——每个权重块需要存储 scale 和 zero_point。这些元数据虽然小，但在 7B 模型里积少成多：

```
不做双重量化：
  每 64 个权重一组，存一个 scale (FP32) + 一个 zero_point (FP32)
  7B 参数的元数据：7B / 64 × 8 字节 ≈ 875 MB

双重量化：
  把这些 scale/zero_point 也量化一次！
  元数据从 FP32 量化到 INT8，再配合一组新的 scale
  7B 参数的元数据：≈ 300 MB

节省 ≈ 575 MB，别小看这 0.5 GB，在消费级显卡上每一 MB 都很珍贵
```

### 创新 3：分页优化器（Paged Optimizer）

LoRA 的优化器状态平时在 GPU 上，但显存不够时怎么办？QLoRA 借助 NVIDIA 统一内存：

```
普通方式：
  优化器状态一直在 GPU 显存里 → 显存不够就 OOM ❌

分页优化器：
  优化器状态默认在 GPU 上
  当显存快满时，自动把部分状态"页面"搬到 CPU 内存
  需要时再搬回来
  → 像操作系统的虚拟内存/页面置换一样
  → 显存不够也不 OOM，只是稍慢一点 ✅

类比：你桌子（GPU）放不下所有书（优化器状态）
  → 把暂时不用的书放书架（CPU 内存）
  → 需要时再去书架拿
  → 桌子永远够用，只是偶尔要起身拿书
```

### 三个创新组合起来

```
                     普通量化       QLoRA 的做法
模型权重存储：       INT4           NF4（正态分布优化）
量化元数据：         FP32           双重量化（再量化一次）
优化器状态：         全在 GPU       分页管理（CPU/GPU 自动换页）
LoRA 部分：         FP16           BF16（训练更稳定）

最终效果：
  7B 模型占用：从 14GB (FP16) → 约 3.5-4GB (NF4)
  LoRA 权重/梯度/优化器：约 50-100 MB
  激活值 + 上下文：约 1-2 GB
  总计：约 6-7 GB

→ 一张 RTX 3060 (12GB) 或 RTX 4060 (8GB) 就能微调 7B 模型！
```

---

## 🔬 QLoRA 的完整流程

> ❓ **问题**：量化模型 + LoRA 训练，具体怎么配合？计算时用 4-bit 还是 16-bit？

QLoRA 的精妙之处在于：**存储用 4-bit，计算用 BF16**。

```
┌──────────────────────────────────────────────┐
│  QLoRA 前向传播                               │
│                                              │
│  1. 从 NF4 存储中取出权重                     │
│     W_nf4 (4-bit) → 反量化 → W_bf16          │
│                                              │
│  2. 计算：y = W_bf16 × x（BF16 精度计算）     │
│                                              │
│  3. 计算 LoRA 部分：y += (α/r) × B × A × x   │
│     （同样是 BF16 精度）                       │
│                                              │
│  4. 输出 y，计算损失                          │
│                                              │
│  注意：                                       │
│  - 模型权重在内存中始终是 NF4（节省空间）       │
│  - 计算时临时反量化为 BF16（保证精度）          │
│  - 只有 LoRA 部分有梯度                       │
│  - 梯度和优化器状态是 BF16/FP32               │
└──────────────────────────────────────────────┘
```

### 为什么要"存 4 算 16"？

```
如果全程 4-bit 计算：
  → 精度太低，梯度不准确，微调效果很差 ❌

如果全程 16-bit 存储：
  → 7B 模型 14GB，消费级显卡塞不下 ❌

QLoRA 的折中：
  → 存储压缩到 4-bit（省空间）
  → 计算时反量化到 16-bit（保精度）
  → 反量化有额外计算开销，但远比显存溢出好
```

---

## 🛠️ QLoRA 代码实战

> ❓ **问题**：QLoRA 在代码中怎么用？从零写一个简化版理解原理。

### 简化版 NF4 量化

真正的 NF4 需要精确的分位数计算，这里我们用一个简化版理解原理：

```python
import torch
import torch.nn as nn
import math

class NF4Linear(nn.Module):
    """简化版 NF4 量化线性层（教学用途）"""
    
    def __init__(self, in_features, out_features, block_size=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        # 先创建 FP32 权重，然后量化
        weight = torch.randn(out_features, in_features) * 0.02
        self.register_buffer('weight_4bit', torch.zeros_like(weight, dtype=torch.uint8))
        self.register_buffer('scale', torch.zeros(
            (out_features * in_features + block_size - 1) // block_size
        ))
        self.register_buffer('zero_point', torch.zeros_like(self.scale))
        
        # 量化权重
        self._quantize(weight)
    
    def _quantize(self, weight_fp32):
        """将 FP32 权重量化为 NF4"""
        flat = weight_fp32.flatten()
        n = flat.shape[0]
        
        for i in range(0, n, self.block_size):
            block = flat[i:i + self.block_size]
            idx = i // self.block_size
            
            # 计算该 block 的缩放参数
            block_min = block.min().item()
            block_max = block.max().item()
            self.scale[idx] = (block_max - block_min) / 15.0
            self.zero_point[idx] = block_min
            
            # 量化到 0-15（4-bit）
            quantized = ((block - block_min) / self.scale[idx]).round().clamp(0, 15).to(torch.uint8)
            self.weight_4bit[i:i + self.block_size] = quantized
    
    def _dequantize(self):
        """将 NF4 权重反量化为 FP32"""
        flat_4bit = self.weight_4bit.flatten()
        n = flat_4bit.shape[0]
        result = torch.zeros(n, dtype=torch.float32)
        
        for i in range(0, n, self.block_size):
            idx = i // self.block_size
            block = flat_4bit[i:i + self.block_size]
            result[i:i + self.block_size] = block.float() * self.scale[idx] + self.zero_point[idx]
        
        return result.view(self.out_features, self.in_features)
    
    def forward(self, x):
        # 反量化然后计算（实际框架会优化这一步）
        weight_fp32 = self._dequantize()
        return x @ weight_fp32.t()
```

### 完整 QLoRA 示例

```python
class QLoRALinear(nn.Module):
    """NF4 量化 + LoRA 的组合"""
    
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        # NF4 量化权重（冻结）
        self.quantized = NF4Linear(in_features, out_features)
        for p in self.quantized.parameters():
            p.requires_grad = False
        
        # LoRA 部分（可训练，BF16/FP32）
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.scaling = alpha / r
    
    def forward(self, x):
        # 反量化计算 + LoRA
        return self.quantized(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

### 显存对比实测

```python
def measure_memory():
    """测量不同配置下的显存占用"""
    d = 4096  # hidden size
    
    configs = {
        "FP16 全量": (d * d * 2, d * d * 2, d * d * 8),
        "LoRA (FP16)": (d * d * 2, 8 * 2 * d * 2, 8 * 2 * d * 8),
        "QLoRA (NF4)": (d * d * 0.5, 8 * 2 * d * 2, 8 * 2 * d * 8),
    }
    
    for name, (weight, grad, optim) in configs.items():
        total_mb = (weight + grad + optim) / 1024 / 1024
        print(f"{name:20s}: {total_mb:8.1f} MB")

measure_memory()
```

---

## 🧬 LoRA 家族全览：不止 QLoRA

> ❓ **问题**：LoRA 提出后，社区涌现了哪些变体？各自解决了什么问题？

QLoRA 解决的是"显存"问题，但 LoRA 还有其他维度的改进空间：

### 1. AdaLoRA — 自适应分配秩

```
问题：LoRA 所有层用同样的 rank r，但不同层的重要性不同

AdaLoRA 的思路：
  → 给重要层分配更高的 rank，不重要层分配更低的 rank
  → 就像考试分配时间：擅长的科目少花时间，薄弱的科目多花时间

怎么判断"重要性"？
  → 通过训练过程中梯度的"能量"（奇异值大小）
  → 梯度能量大的层 → 正在积极学习 → 需要更高的 rank
  → 梯度能量小的层 → 已经收敛 → 低 rank 就够

效果：
  在相同参数预算下，AdaLoRA 比 LoRA 效果更好
  因为参数被"聪明地"分配到了最需要的地方
```

### 2. DoRA — 分解方向与大小

```
问题：LoRA 把 ΔW 当成一个整体来学，但权重的变化可以分解为两部分

DoRA（Weight-Decomposed Low-Rank Adaptation）的思路：
  把权重分解为：
    W = m × (V / ||V||)
    其中 m 是大小（magnitude），V 是方向（direction）

  - 方向 V 用 LoRA 来调整（低秩分解）
  - 大小 m 用单独的可学习标量来调整
  
类比：
  你在调整一个箭头（权重向量）
  LoRA：只能同时改变箭头的方向和长度
  DoRA：方向和长度分别调整，更灵活

效果：
  更接近全量微调的效果
  特别是在较大的 rank 设置下优势明显
```

### 3. LoRA+ — 不对称学习率

```
问题：LoRA 的 A 和 B 用相同的学习率，但它们的角色不同

LoRA+ 的发现：
  - A 矩阵：负责"提取"输入的特征（类似于"读"）
  - B 矩阵：负责"组合"输出（类似于"写"）
  - B 初始化为 0，需要更快地从零开始学习
  - A 初始化正常，不需要那么激进

解决方案：给 B 更高的学习率
  lr_B = 16 × lr_A  （论文推荐的典型比例）

效果：
  训练速度提升约 2 倍
  最终效果也有小幅提升

这个发现看似简单，但它揭示了 A 和 B 在训练中的不对称性
——这不只是工程 trick，有理论分析支持
```

### 4. rsLoRA — 重新思考缩放

```
问题：LoRA 的缩放因子 α/r 在 rank 很大时不合理

原版 LoRA：  输出贡献 = (α/r) × B × A × x
rsLoRA：    输出贡献 = (α/√r) × B × A × x

为什么 √r 更好？
  当 r 增大时，BA 的 Frobenius 范数大约按 √r 增长
  用 1/r 缩放会"过度压制"高 rank LoRA 的贡献
  用 1/√r 缩放让不同 rank 的 LoRA 在同一"音量"上
  
效果：
  在高 rank（r ≥ 32）时表现更稳定
  低 rank 时和原版 LoRA 差别不大
```

### 5. PiSSA — 主成分初始化

```
问题：LoRA 的 A 随机初始化、B 初始化为 0，意味着初始时 LoRA 没有贡献

PiSSA 的思路：
  不从零开始！用 SVD 找到原始权重中最重要的 r 个成分
  把它们"分离"出来作为 LoRA 的初始值
  
  W = W_remaining + (B_init × A_init)
  其中 B_init × A_init 是 W 的前 r 个主成分

效果：
  初始时 LoRA 就已经捕捉了原始权重中最重要的部分
  训练更快收敛
  效果通常优于标准 LoRA

类比：
  标准 LoRA = 找一个新手，从零开始学
  PiSSA     = 找一个有基础的实习生，已经有部分能力了
```

---

## 📊 实战对比：不同方案的显存和效果

> ❓ **问题**：这些方案放在一起，显存和效果差多少？

### 显存对比（7B 模型，单卡微调）

```
方案                模型存储    LoRA/训练   总显存需求    硬件门槛
────────────────────────────────────────────────────────────
全量 FP16 微调       14 GB      72 GB      ~86 GB       2-4× A100
LoRA FP16           14 GB      ~0.5 GB    ~18 GB       1× RTX 4090
QLoRA (NF4)         3.5 GB     ~0.5 GB    ~6 GB        1× RTX 3060 12GB
QLoRA + 双重量化     3.5 GB     ~0.3 GB    ~5 GB        1× RTX 3060 8GB*
QLoRA + 分页优化器   3.5 GB     ~0.3 GB    ~4-5 GB      1× RTX 4060 8GB*

*需要足够的 CPU 内存做分页
```

### 效果对比（在常见基准上的典型表现）

```
方案              参数量      效果（相对全量微调）
──────────────────────────────────────────────
全量微调           100%       基准线
LoRA r=8          0.1%       略低 1-2%
LoRA r=16         0.2%       基本持平
QLoRA r=8         0.1%       略低 1-3%（量化损失）
QLoRA r=32        0.4%       几乎持平
AdaLoRA           0.1%       ≈ LoRA r=16 的效果
DoRA              0.2%       略优于 LoRA r=16
PiSSA             0.1%       ≈ LoRA r=16，收敛更快

结论：QLoRA 的量化损失很小（1-3%），换来的是显存降低 70%+
      对于大多数实际应用，这点损失完全可以接受
```

---

## 🎯 怎么选？决策树

> ❓ **问题**：面对这么多方案，我该选哪个？

```
你的 GPU 有多少显存？
│
├─ ≥ 80 GB (A100/H100)
│   └─ 直接全量微调或 LoRA FP16（不差钱就直接上）
│
├─ 24 GB (RTX 4090/3090)
│   └─ LoRA FP16（7B 模型轻松，13B 也能跑）
│
├─ 12-16 GB (RTX 3060/4060Ti)
│   └─ QLoRA NF4（7B 模型很舒服，13B 勉强能跑）
│
├─ 8 GB (RTX 4060/3060 Laptop)
│   └─ QLoRA + 双重量化 + 分页优化器（只能跑 7B）
│
└─ < 8 GB
    └─ 考虑用更小的模型（1-3B），或者用云端算力

选好了存储方案后：
│
├─ 效果优先 → DoRA 或 PiSSA（更好效果，同参数量）
├─ 速度优先 → LoRA+（更快收敛）
├─ 资源受限 → AdaLoRA（自适应 rank，最高效利用参数）
└─ 不确定   → 标准 QLoRA（最成熟、生态最好）
```

---

## 🔑 关键洞察

1. **QLoRA = NF4 量化 + LoRA** — 存储用 4-bit 省空间，计算时反量化到高精度保质量
2. **NF4 不是普通量化** — 它利用了权重服从正态分布的先验，比均匀量化精度更高
3. **三个关键创新缺一不可** — NF4 + 双重量化 + 分页优化器，三者配合才能把显存压到最低
4. **LoRA 变体各有侧重** — AdaLoRA 自适应 rank、DoRA 分解方向大小、LoRA+ 不对称学习率、PiSSA 主成分初始化
5. **QLoRA 已经是消费级微调的事实标准** — HuggingFace 的 BitsAndBytes 集成让它用起来非常简单
6. **量化损失很小（1-3%）** — 换来的显存节省远超精度损失，对大多数应用来说完全值得

---

## 📝 一句话总结

> QLoRA 的核心洞察是：模型权重用 NF4（正态分布优化的 4-bit 格式）存储，计算时反量化为高精度，再配合 LoRA 微调——7B 模型的微调从需要 86GB 显存（多张 A100）降低到约 6GB（一张 RTX 3060），代价仅为 1-3% 的精度损失。加上 AdaLoRA、DoRA、LoRA+、PiSSA 等变体在不同维度上的优化，LoRA 家族已经成为参数高效微调的完整工具箱。

## 📖 关键术语速查

| 术语 | 含义 |
|------|------|
| **QLoRA** | 量化 LoRA，用 4-bit 存储模型 + LoRA 微调 |
| **NF4** | 4-bit NormalFloat，按正态分布分位数设计的 4-bit 格式 |
| **双重量化** | 对量化的 scale/zero_point 再做一次量化，进一步节省空间 |
| **分页优化器** | 利用统一内存在 GPU/CPU 间自动换页优化器状态 |
| **反量化** | 将低精度数值恢复为高精度，用于计算 |
| **量化误差** | 量化前后数值的差异，是精度损失的主要来源 |
| **AdaLoRA** | 自适应 LoRA，不同层用不同的 rank |
| **DoRA** | 权重分解 LoRA，方向和大小分别调整 |
| **LoRA+** | 不对称学习率，B 矩阵用更高学习率 |
| **rsLoRA** | 用 1/√r 替代 1/r 缩放，高 rank 更稳定 |
| **PiSSA** | 用 SVD 主成分初始化 LoRA，收敛更快 |
| **BitsAndBytes** | HuggingFace 的量化库，QLoRA 的常用后端 |

---

> 🤔 **今天留下的悬念**：QLoRA 用 NF4 量化把模型从 14GB 压到 3.5GB，效果只损失 1-3%。但这引出一个更深层的问题：量化到底是怎么做到"压缩这么狠但效果几乎不变"的？INT8、INT4、GPTQ、AWQ、GGUF……这些五花八门的量化方法各有什么优劣？明天的主题——**量化基础**，我们深入理解量化的数学原理，手写一个线性量化器，亲眼看看精度损失到底有多大。

**下节预告**：Day 22 — 量化基础，从 FP16 到 INT4 的数学之旅。

---

*本课程代码开源于 [GitHub](https://github.com/LuChunQi/llm-from-scratch)，欢迎 Star ⭐*
