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

...
```

## 环境

- Python 3.10+
- PyTorch 2.0+
- 其他依赖在各 Day 目录的 README 中说明

## License

MIT
