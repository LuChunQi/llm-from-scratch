# 🧠 LLM From Scratch — 从零理解大语言模型

> 50天，从底层原理到上层应用，系统掌握大语言模型。

## 这是什么？

一份面向开发者的 LLM 系统学习课程。每天一篇，从"语言模型到底是什么"讲到"部署一个完整的 RAG 系统"。

- **不是调 API 教程** — 我们讲底层原理
- **不是论文复读机** — 生动类比 + 逐行代码
- **从零开始** — 假设你会 Python，但不了解深度学习

## 课程路线

### 模块一：基础感知（Day 1-5）
语言模型入门 → Tokenization → 词嵌入 → RNN的局限 → Transformer 登场

### 模块二：Transformer 核心机制（Day 6-15）
Self-Attention → Multi-Head → 位置编码 → FFN → LayerNorm → 残差连接 → Decoder-Only → KV Cache → GQA → Flash Attention

### 模块三：训练与优化（Day 16-25）
预训练 → SFT → RLHF → DPO → LoRA → QLoRA → 量化 → MoE → 分布式训练

### 模块四：蒸馏与压缩（Day 26-30）
知识蒸馏 → LLM蒸馏 → 剪枝 → Speculative Decoding → 压缩全景

### 模块五：现代架构与前沿（Day 31-38）
长上下文 → 多模态 → Mamba/SSM → RWKV → RAG → Agent → ACP/MCP

### 模块六：工程实践（Day 39-45）
vLLM → TensorRT-LLM → LangChain → 部署 → 评测 → 安全对齐

### 模块七：从零实战（Day 46-50）
手写 Mini-Transformer → LoRA 微调实战 → RAG 系统 → 总结

## 代码结构

```
day01-what-is-lm/          # Day 1: 语言模型是什么？从掷骰子说起
├── README.md              # 当日讲解
└── ngram_lm.py            # 配套代码：N-gram 语言模型

day02-bpe-tokenizer/      # Day 2: Tokenization — 文字怎么变成数字
├── README.md
└── bpe_tokenizer.py       # 配套代码：BPE 分词器

day03-embedding/           # Day 3: 词嵌入（Word Embedding）
├── README.md
└── skip_gram.py            # 配套代码：Skip-gram (Word2Vec)

day04-rnn-to-attention/    # Day 4: 从 RNN 到 Attention
├── README.md
└── rnn_to_attention.py     # 配套代码：RNN/LSTM/Attention 对比

day05-transformer-intro/  # Day 5: Transformer 登场
├── README.md
└── transformer_skeleton.py # 配套代码：Transformer Encoder 骨架

day06-self-attention/    # Day 6: Self-Attention — 自注意力机制
├── README.md
└── self_attention.py      # 配套代码：纯 NumPy + PyTorch 实现

day07-multi-head-attention/ # Day 7: Multi-Head Attention — 多头注意力
├── README.md
└── multi_head_attention.py # 配套代码：多头注意力 + 可视化 + 对比实验

day08-positional-encoding/ # Day 8: 位置编码 — Sinusoidal / RoPE / ALiBi
├── README.md
└── positional_encoding.py  # 配套代码：三种位置编码实现 + 对比实验

day09-feed-forward-network/ # Day 9: Feed-Forward Network — Transformer 的消化系统
├── README.md
└── feed_forward_network.py  # 配套代码：FFN实现 + 激活函数对比 + 键值记忆实验

day10-layer-norm/          # Day 10: Layer Normalization + 残差连接
├── README.md
└── layer_norm.py            # 配套代码：LayerNorm/RMSNorm + 梯度实验 + Transformer Block

day11-residual-connection/  # Day 11: 残差连接深入
├── README.md
└── residual_connection.py   # 配套代码：梯度可视化 + 信息保留 + 残差流分析 + 变体对比

day12-decoder-only/          # Day 12: Decoder-Only 架构 — Causal Mask 与自回归生成
├── README.md
└── decoder_only.py           # 配套代码：Causal Mask + Decoder Block + 自回归生成

day13-kv-cache/              # Day 13: KV Cache — 让自回归生成快起来的"时间机器"
├── README.md
└── kv_cache.py               # 配套代码：KV Cache 实现 + 速度对比 + 内存估算 + Prefill/Decode 演示

day14-mqa-gqa/               # Day 14: MQA & GQA — 给 KV Cache 瘦身的"共享经济"
├── README.md
└── mqa_gqa.py                # 配套代码：MHA/GQA/MQA 三合一实现 + 内存对比 + 速度实测 + expand 详解

day15-flash-attention/        # Day 15: Flash Attention — 不改变数学，却让注意力快 2-4 倍的魔法
├── README.md
└── flash_attention.py        # 配套代码：分块注意力模拟 + 在线 Softmax 演示 + IO/显存估算 + 数值验证

day16-pretraining/            # Day 16: 预训练 — 让 Transformer 从零学会"说人话"
├── README.md
└── pretraining.py            # 配套代码：完整 MiniGPT 预训练 + 字符级Tokenizer + 损失曲线 + 文本生成

day17-sft/                    # Day 17: SFT（监督微调）— 让模型从"会说话"到"会听话"
├── README.md
└── sft.py                     # 配套代码：SFT 数据构造 + Chat Template + 损失掩码 + 微调训练 + 灾难性遗忘演示

day18-rlhf/                   # Day 18: RLHF（基于人类反馈的强化学习）
├── README.md
└── rlhf.py                    # 配套代码：Reward Model + PPO 训练 + Bradley-Terry 损失 + KL 惩罚 + 奖励曲线

day19-dpo/                    # Day 19: DPO（直接偏好优化）— 绕过强化学习，直接优化偏好
├── README.md
└── dpo.py                     # 配套代码：DPO 损失函数 + 隐式奖励 + 偏好训练 + 对比实验

...
```

## 环境

- Python 3.10+
- PyTorch 2.0+
- 其他依赖在各 Day 目录的 README 中说明

## License

MIT