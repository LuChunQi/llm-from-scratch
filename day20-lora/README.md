# Day 20: LoRA（Low-Rank Adaptation）— 用 0.1% 的参数微调大模型

## 🔗 上节回顾

昨天我们学习了 DPO——用简单的分类损失替代 RLHF 中复杂的 PPO 流程，让对齐训练变得和 SFT 一样稳定。但我们一直回避一个残酷的现实：**无论是 SFT、RLHF 还是 DPO，你都需要在训练时更新模型的所有参数**。一个 7B 参数的模型，光权重就占 14GB 显存，加上梯度和优化器状态，一张 24GB 的消费级显卡根本塞不下。

今天我们就来解决这个问题——**LoRA（Low-Rank Adaptation，低秩适配）**，它不修改原始模型的任何参数，而是给模型打一个"小补丁"，用不到 0.1% 的参数量就能达到全量微调的效果。

---

## 🏋️ 全量微调的困境：为什么 7B 模型比 7GB 显存大得多？

> ❓ **问题**：一个 7B 参数的模型，权重占 7GB 空间，为什么 24GB 显存的 4090 都训练不了？

答案在于：**训练 ≠ 推理**。推理只需要模型权重，但训练还需要梯度和优化器状态。

```
全量微调 7B 模型的显存清单：

1. 模型权重（FP16）       = 7B × 2 字节 = 14 GB
2. 梯度（FP16）           = 7B × 2 字节 = 14 GB
3. Adam 优化器状态        = 7B × 8 字节 = 56 GB
   - 一阶动量 m (FP32)    = 7B × 4 字节 = 28 GB
   - 二阶动量 v (FP32)    = 7B × 4 字节 = 28 GB
4. 激活值（前向传播中间结果）≈ 2-4 GB

总计 ≈ 86-90 GB！

→ 即使 A100 80GB 也单卡跑不了
→ 需要 2-4 张 A100 才能全量微调一个 7B 模型
→ 70B 模型？几十张 A100 起步
```

这就像你想给一栋大楼换个内部装修风格（微调），但每次都得把整栋楼拆了重建（更新所有参数）——太浪费了。

**关键洞察**：微调时，模型权重的变化量其实很小。我们需要的不是更新 7B 个参数，而是找到一种方式，只用很少的参数就能表示这种"小变化"。

---

## 💡 核心直觉：矩阵的低秩分解

> ❓ **问题**：怎么用很少的参数表示一个"小的权重变化"？

LoRA 的核心思想来自一个数学事实：**一个大的矩阵变化，往往可以用两个小矩阵的乘积来近似**。

### 打个比方：RGB 调色板

想象你有一张 1000×1000 的高清照片（100 万像素），你想微调它的色调。你当然可以逐像素修改（全量微调），但更聪明的方式是：**只调整 RGB 三个通道的曲线**。三个参数就能改变整张照片的色调——因为颜色变化有一个"低秩结构"。

### 数学原理

原始权重矩阵 W ∈ ℝ^(d×k)，微调时我们想学的变化量 ΔW：

```
全量微调：  W_new = W + ΔW         → ΔW 也是 d×k 的，需要 d×k 个参数

LoRA：      W_new = W + B × A       → B ∈ ℝ^(d×r), A ∈ ℝ^(r×k)

当 r << min(d, k) 时：
- 全量参数：d × k
- LoRA 参数：d × r + r × k = r × (d + k)

例如 d = k = 4096, r = 8：
- 全量参数：4096 × 4096 = 16,777,216
- LoRA 参数：8 × (4096 + 4096) = 65,536
- 压缩比：256 倍！
```

**这就是 LoRA 的魔法**：用一个极小的"瓶颈"（秩 r）把参数量压缩几百倍。

### 具体长什么样？

```
原始线性层：y = Wx                    W ∈ ℝ^(4096 × 4096)

LoRA 版本：  y = Wx + BAx            B ∈ ℝ^(4096 × 8), A ∈ ℝ^(8 × 4096)
             ↑     ↑
           冻结   可训练（LoRA 补丁）
```

原始权重 W 完全不动，我们只训练 A 和 B 这两个小矩阵。**推理时**，可以把 BA 合并回 W（`W_new = W + BA`），不增加任何推理延迟。

---

## 🏗️ LoRA 的完整架构

> ❓ **问题**：LoRA 只是一个矩阵分解的数学技巧吗？在实际的 Transformer 中怎么用？

LoRA 不止是数学，它是一套完整的工程方案。让我们看看它怎么接入 Transformer。

### 插入位置：Q、K、V 还是全都要？

原始论文只在 Attention 的 Q、V 投影矩阵上加 LoRA，但后来的实践表明**在所有线性层上加 LoRA 效果最好**：

```
Transformer 中的一个 Block：

     Input
       ↓
  ┌─────────────────┐
  │  Self-Attention  │
  │  ├─ Wq (d×d)    │  ← LoRA ✅
  │  ├─ Wk (d×d)    │  ← LoRA ✅
  │  ├─ Wv (d×d)    │  ← LoRA ✅
  │  └─ Wo (d×d)    │  ← LoRA ✅
  └─────────────────┘
       ↓
  ┌─────────────────┐
  │     FFN          │
  │  ├─ W1 (d×4d)   │  ← LoRA ✅（可选）
  │  └─ W2 (4d×d)   │  ← LoRA ✅（可选）
  └─────────────────┘

每个线性层 W 都可以挂一对 LoRA 矩阵 (B, A)
```

### 初始化策略

LoRA 的初始化有一个精妙的设计：

```
矩阵 A：使用 Kaiming 均匀初始化（或正态初始化）
         → 和普通的权重初始化一样

矩阵 B：初始化为零矩阵！
         → 这保证训练开始时 BA = 0，即 LoRA 的贡献为零
         → 模型从原始权重的行为开始，不会"起步就跑偏"

这个设计非常关键：
- 如果 B 不初始化为 0，模型一开始的行为就被 LoRA 扰乱了
- B = 0 意味着"我还没学会，先别添乱"
- 训练过程中 B 逐渐从 0 变大，LoRA 的贡献逐渐增加
```

### 缩放因子 α

LoRA 还有一个超参数 α（缩放系数），用来控制 LoRA 的"影响力"：

```
实际输出：y = Wx + (α/r) × BAx

- α/r 是缩放因子
- r = 8, α = 16 → 缩放因子 = 2
- r = 8, α = 8  → 缩放因子 = 1

为什么要缩放？
- 不同 rank r 的 LoRA，ΔW 的"量级"不同
- r 越大，BA 的值可能越大（因为有更多自由度）
- 缩放因子让不同 r 的 LoRA 在同一量级上比较
- 实践中 α 通常设为 r 的 1-2 倍
```

---

## 📊 LoRA 的参数量计算

> ❓ **问题**：LoRA 说"只用 0.1% 的参数"，这个数字是怎么来的？我来算给你看。

以 LLaMA-7B 为例：

```
LLaMA-7B 的结构：
- 32 个 Transformer Block
- 每个 Block 有 Attention（Wq, Wk, Wv, Wo）+ FFN（W1, W2, W3）
- hidden_size = 4096

只在 Attention 层加 LoRA（rank = 8）：

每个 Block 的 LoRA 参数：
- Wq: 4096 × 8 + 8 × 4096 = 65,536
- Wk: 4096 × 8 + 8 × 4096 = 65,536
- Wv: 4096 × 8 + 8 × 4096 = 65,536
- Wo: 4096 × 8 + 8 × 4096 = 65,536
- 每个 Block：4 × 65,536 = 262,144

32 个 Block：32 × 262,144 = 8,388,608 ≈ 8.4M

总参数：7B = 7,000M
LoRA 比例：8.4M / 7,000M ≈ 0.12%

→ 确实是 0.1% 的参数量！
```

**显存对比**：

```
全量微调 7B：
- 权重 + 梯度 + 优化器 ≈ 86 GB
- 需要 2-4 × A100 (80GB)

LoRA 微调 7B（rank=8）：
- 模型权重（冻结）≈ 14 GB（不需要梯度）
- LoRA 权重 ≈ 16 MB
- LoRA 梯度 ≈ 16 MB
- LoRA 优化器 ≈ 64 MB
- 激活值 ≈ 2-4 GB
- 总计 ≈ 16-18 GB
- 一张 RTX 4090 (24GB) 就够了！✅

从 4 张 A100 → 1 张 4090，成本从 $60,000 降到 $1,500！
```

---

## 🔬 LoRA 为什么有效？低秩假说

> ❓ **问题**：凭什么一个 rank=8 的小矩阵就能近似全量微调的效果？8 这么小的秩够用吗？

这背后有一个重要的经验发现：**微调过程中的权重变化量 ΔW 天然就是低秩的**。

### 实验证据

研究人员做了这样的实验：

```
1. 全量微调一个模型
2. 收集训练过程中每一步的 ΔW = W_final - W_init
3. 对 ΔW 做奇异值分解（SVD）
4. 看看 ΔW 的"有效秩"是多少

结果：ΔW 的大部分信息集中在极少数几个奇异值上！

例如一个 4096×4096 的权重矩阵：
- SVD 有 4096 个奇异值
- 但前 10 个奇异值就能解释 90%+ 的方差
- 前 100 个能解释 99%+

→ ΔW 的"内在维度"远小于它的表面维度
→ rank=8 到 rank=64 就足够捕捉大部分微调信号
```

### 直觉理解

为什么微调的变化是低秩的？因为**微调的本质是"轻微调整行为"，不是"重写知识"**：

```
类比：驾照考试

预训练 = 考驾照（学会了所有驾驶技能）
微调   = 适应新车型（方向盘重一点，刹车灵敏一点）

适应新车型不需要你重新学一遍驾驶——你只需要微调几个"参数"：
- 方向盘力度
- 刹车灵敏度
- 油门响应速度

这就像 LoRA：只调几个维度就能"适应新任务"

相比之下，全量微调就像"把整本驾驶手册重写一遍"——太浪费了
```

---

## 🔧 LoRA 的变体：不只是 A×B

> ❓ **问题**：LoRA 只有这一种形式吗？后来有什么改进？

LoRA 提出后，社区涌现了大量变体，针对不同的场景做了优化：

### LoRA 家族一览

```
LoRA（原版，2021）
├─ 只在 Attention 的 Q/V 上加
└─ rank 通常 4-64

AdaLoRA（2023）
├─ 不同层的 rank 可以不同
├─ 重要层给高 rank，不重要层给低 rank
└─ 自适应分配参数预算

QLoRA（2023）
├─ 模型用 4-bit 量化存储
├─ LoRA 部分仍然是 FP16
└─ 进一步降低显存（7B 模型只需 ~6 GB！）

LoRA+（2024）
├─ A 和 B 使用不同的学习率
├─ B 的学习率更高（因为 B 初始化为 0，需要更快学习）
└─ 训练速度和效果都有提升

DoRA（2024）
├─ 把权重分解为"方向"和"大小"两部分
├─ LoRA 只调整方向，大小用单独的标量
└─ 效果更接近全量微调

rsLoRA（2024）
├─ 缩放因子从 α/r 改为 α/√r
├─ 高 rank 时表现更稳定
└─ 理论分析表明 √r 更合理
```

其中 **QLoRA** 是最实用的变体——我们明天会专门讲它。

---

## 🎯 LoRA 实战：手写一个 LoRA 层

> ❓ **问题**：LoRA 说起来简单，实际代码怎么写？

LoRA 的核心代码其实只有几十行！让我们一步步实现：

### Step 1: LoRA 线性层

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """LoRA 包装的线性层"""
    
    def __init__(self, original_linear, r=8, alpha=16):
        super().__init__()
        # 原始线性层（冻结）
        self.original = original_linear
        self.original.weight.requires_grad = False
        
        d = original_linear.out_features
        k = original_linear.in_features
        
        # LoRA 矩阵
        self.lora_A = nn.Parameter(torch.empty(r, k))
        self.lora_B = nn.Parameter(torch.zeros(d, r))
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 初始化为 0（关键！）
        
        self.scaling = alpha / r
    
    def forward(self, x):
        # 原始输出 + LoRA 调整
        return self.original(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```

就这么简单！一个 `LoRALinear` 就是一个完整的 LoRA 层。

### Step 2: 注入到 Transformer

```python
def inject_lora(model, r=8, alpha=16, target_modules=["q_proj", "v_proj"]):
    """把模型中的指定线性层替换为 LoRA 版本"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查是否是需要注入的目标模块
            if any(target in name for target in target_modules):
                # 获取父模块和属性名
                *path, attr = name.split('.')
                parent = model
                for p in path:
                    parent = getattr(parent, p)
                
                # 替换为 LoRA 层
                lora_layer = LoRALinear(module, r=r, alpha=alpha)
                setattr(parent, attr, lora_layer)
    
    # 冻结所有非 LoRA 参数
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False
    
    return model
```

### Step 3: 合并权重（推理时零成本）

```python
def merge_lora(model):
    """把 LoRA 权重合并回原始权重，推理时零额外开销"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            # 合并：W_new = W + (α/r) × B @ A
            module.original.weight.data += (
                module.scaling * module.lora_B @ module.lora_A
            )
```

---

## 🧪 LoRA 训练流程

完整的 LoRA 微调流程如下：

```
┌─────────────────────────────────────────────┐
│  Step 1: 加载预训练模型                      │
│  model = AutoModelForCausalLM.from_pretrained("llama-7b")  │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│  Step 2: 注入 LoRA                          │
│  model = inject_lora(model, r=8)            │
│  → 冻结原始权重，只训练 A、B 矩阵            │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│  Step 3: 正常训练（和全量微调完全一样！）      │
│  optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()))  │
│  for batch in dataloader:                   │
│      loss = model(batch).loss               │
│      loss.backward()                        │
│      optimizer.step()                       │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│  Step 4: 保存 LoRA 权重（只需几 MB！）       │
│  torch.save({k: v for k, v in model.state_dict().items() if 'lora_' in k},  │
│             "lora_weights.pt")               │
└──────────────────────┬──────────────────────┘
                       ↓
┌─────────────────────────────────────────────┐
│  Step 5: 推理（可选合并）                    │
│  merge_lora(model)                          │
│  → 推理速度 = 原始模型，零额外开销            │
└─────────────────────────────────────────────┘
```

### 一个精妙之处：多 LoRA 切换

因为 LoRA 权重只有几 MB，你可以为一个基座模型训练多个 LoRA：

```
基座模型：LLaMA-7B（14 GB）

LoRA 权重：
├── lora-coding.pt     (16 MB)  — 代码助手
├── lora-chat.pt       (16 MB)  — 聊天助手
├── lora-translate.pt  (16 MB)  — 翻译助手
└── lora-creative.pt   (16 MB)  — 创意写作

切换场景只需：
1. 加载基座模型（14 GB，一次性的）
2. 加载对应的 LoRA 权重（16 MB，瞬间切换）
3. 合并 → 推理

这就像一个万能遥控器，基座是遥控器本体，每个 LoRA 是一个"频道"！
```

---

## 📈 超参数指南

LoRA 的效果主要受三个超参数影响：

### Rank（r）

```
rank = 4：    适合简单任务（如风格迁移）
rank = 8：    适合中等任务（如 SFT），最常用
rank = 16：   适合复杂任务
rank = 32-64： 接近全量微调的效果，但参数量也在增加

推荐：
- 先从 r=8 开始
- 效果不够好就翻倍（16、32）
- 通常 r=16 已经非常好了
```

### Alpha（α）

```
α = r：     缩放因子 = 1，LoRA 影响力适中
α = 2r：    缩放因子 = 2，LoRA 影响力较大（最常用）
α = r/2：   缩放因子 = 0.5，LoRA 影响力较小

推荐：α = 2r（即 α=16, r=8 或 α=32, r=16）
```

### Target Modules

```
只加 Q/V：      参数最少，效果已经不错（原始论文）
加 Q/K/V/O：    参数更多，效果更好（HuggingFace 默认推荐）
加所有线性层：   参数最多，效果最好（最新实践推荐）

推荐：至少加 Q/K/V/O，资源允许就加所有线性层
```

---

## 🔑 关键洞察

1. **LoRA 的核心思想是低秩分解** — 权重变化 ΔW 天然是低秩的，用两个小矩阵就能近似
2. **冻结原始权重，只训练 LoRA 补丁** — 参数量从 7B 降到 8M（0.1%），显存从 86GB 降到 16GB
3. **B 初始化为零是关键** — 保证训练开始时 LoRA 不干扰原始模型的行为
4. **推理时可以合并权重** — 合并后和原始模型完全一样，零额外开销
5. **多 LoRA 切换是杀手锏** — 一个基座模型 + 多个几 MB 的 LoRA 权重，瞬间切换不同能力
6. **LoRA 已经成为微调的标准方法** — HuggingFace PEFT 库、LLaMA-Factory 等都默认使用 LoRA

---

## 📝 一句话总结

> LoRA 的核心洞察是：微调时权重变化量 ΔW 是低秩的，所以不需要更新全部参数——只需给每个线性层挂一对小矩阵 A 和 B，用不到 0.1% 的参数就能近似全量微调的效果。冻结原始权重、只训练 LoRA 补丁，显存需求降低 5 倍以上，而且推理时可以合并回原始权重，零额外开销。

## 📖 关键术语速查

| 术语 | 含义 |
|------|------|
| **LoRA** | Low-Rank Adaptation，低秩适配，用小矩阵近似权重变化 |
| **秩（rank, r）** | LoRA 瓶颈维度，通常 4-64，控制 LoRA 的"表达能力" |
| **α（alpha）** | 缩放系数，控制 LoRA 对输出的影响力，通常 α = 2r |
| **ΔW** | 微调前后的权重变化量，LoRA 用 BA 近似它 |
| **低秩** | 矩阵的"有效维度"远小于其表面维度 |
| **合并（merge）** | 推理时把 BA 加回原始权重，消除额外计算 |
| **inject** | 把原始线性层替换为 LoRA 版本的过程 |
| **target modules** | 指定哪些线性层需要注入 LoRA（如 q_proj, v_proj） |
| **PEFT** | Parameter-Efficient Fine-Tuning，参数高效微调的统称 |
| **QLoRA** | LoRA + 4-bit 量化，明天详细讲解 |

---

> 🤔 **今天留下的悬念**：LoRA 把可训练参数降到了 0.1%，但模型权重本身还是 FP16 的 14GB——有没有办法把模型本身也压缩一下？如果用 4-bit 量化存储模型，再配合 LoRA 微调，7B 模型能不能塞进一张 8GB 显存的消费级显卡？明天的主题——**QLoRA**，量化 + LoRA 的完美结合，让你用最便宜的显卡微调最大的模型。

**下节预告**：Day 21 — QLoRA & LoRA 变体，量化与低秩的化学反应。

---

*本课程代码开源于 [GitHub](https://github.com/LuChunQi/llm-from-scratch)，欢迎 Star ⭐*
