#!/usr/bin/env python3
"""
Day 16: 预训练 — 从头训练一个玩具级 GPT

完整实现：
1. 字符级 Tokenizer
2. 完整的 Decoder-Only Transformer 模型（GPT 架构）
3. 训练循环：含 warmup、梯度裁剪、余弦学习率衰减
4. 训练过程损失曲线打印
5. 不同训练阶段的文本采样，观察能力涌现
6. 从随机权重到生成莎士比亚风格文本的全过程

运行：python3 pretraining.py
依赖：pip install torch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import time

# ============================================================
# 超参数配置
# ============================================================

# 设备选择：有 GPU 用 GPU，没有用 CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 超参数 —— 玩具级配置（GPT-3 的 1/1000000）
config = {
    'vocab_size': 65,       # Shakespeare 中所有唯一字符数
    'd_model': 128,         # 嵌入维度（GPT-3: 12288）
    'n_heads': 4,           # 注意力头数（GPT-3: 96）
    'n_layers': 4,          # Transformer Block 层数（GPT-3: 96）
    'block_size': 64,       # 上下文长度（GPT-3: 2048）
    'dropout': 0.1,         # Dropout 率
    'batch_size': 32,       # 批大小（CPU 用 32；GPU 可增至 64-128）
    'learning_rate': 3e-4,  # 学习率
    'max_iters': 2000,      # 训练迭代次数（CPU 环境；GPU 可增至 5000+）
    'eval_interval': 500,   # 每 N 步评估一次
    'warmup_iters': 100,    # 学习率预热步数
}

print("=" * 70)
print("🧠 Day 16: 预训练 — 从头训练一个玩具级 GPT")
print("=" * 70)
print(f"\n📱 设备: {device}")
print(f"📋 配置: d_model={config['d_model']}, n_heads={config['n_heads']}, "
      f"n_layers={config['n_layers']}, block_size={config['block_size']}")


# ============================================================
# 第一部分：数据准备 —— Shakespeare 文本
# ============================================================
print("\n" + "─" * 70)
print("📌 第一部分：数据准备")
print("─" * 70)

# Shakespeare 数据集 — Karpathy 的经典 Tiny Shakespeare
# 约 1MB，包含所有莎士比亚戏剧的文本
# 下载方式: gh api /repos/karpathy/char-rnn/git/blobs/7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9 --jq '.content' | base64 -d > input.txt
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.txt")

if not os.path.exists(DATA_PATH):
    # 尝试在当前工作目录查找
    DATA_PATH = "input.txt"
    if not os.path.exists(DATA_PATH):
        print("❌ 找不到 input.txt，请先下载数据集")
        print("   方法: gh api /repos/karpathy/char-rnn/git/blobs/7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9 --jq '.content' | base64 -d > input.txt")
        exit(1)
print("📂 使用本地数据集")

# 读取文本
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"📄 数据集大小: {len(text):,} 个字符")
print(f"📄 前 200 个字符预览:\n{text[:200]}\n")

# ============================================================
# 第二部分：字符级 Tokenizer
# ============================================================
print("─" * 70)
print("📌 第二部分：字符级 Tokenizer")
print("─" * 70)

# 构建词汇表：所有唯一字符
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"🔤 词汇表大小: {vocab_size}（唯一字符数）")
print(f"🔤 词汇表: {''.join(chars)}")

# 字符 ↔ 数字的映射
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# 编码/解码函数
def encode(s):
    """字符串 → token ID 列表"""
    return [char_to_idx[ch] for ch in s]

def decode(ids):
    """token ID 列表 → 字符串"""
    return ''.join(idx_to_char[i] for i in ids)

# 演示
sample = "Hello, World!"
encoded = encode(sample)
decoded = decode(encoded)
print(f"\n🔤 编码演示: '{sample}' → {encoded}")
print(f"🔤 解码验证: {encoded} → '{decoded}'")

# 编码整个数据集
data = torch.tensor(encode(text), dtype=torch.long)
print(f"📊 编码后数据: {data.shape} 个 token（tensor）")

# 划分训练集/验证集（90%/10%）
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]
print(f"📊 训练集: {len(train_data):,} tokens, 验证集: {len(val_data):,} tokens")


# ============================================================
# 第三部分：数据加载器 —— 构造 (输入, 目标) 对
# ============================================================
print("\n" + "─" * 70)
print("📌 第三部分：数据加载器")
print("─" * 70)

def get_batch(split):
    """
    随机采样一个 batch 的训练数据。
    
    每个样本：输入 = [x_0, x_1, ..., x_{T-1}]
              目标 = [x_1, x_2, ..., x_T]（左移一位）
    
    意义：位置 i 的输入预测位置 i+1 的 token
    """
    data_source = train_data if split == 'train' else val_data
    # 随机选 batch_size 个起始位置
    ix = torch.randint(len(data_source) - config['block_size'], (config['batch_size'],))
    # 构造输入
    x = torch.stack([data_source[i:i + config['block_size']] for i in ix])
    # 构造目标（输入左移一位）
    y = torch.stack([data_source[i + 1:i + config['block_size'] + 1] for i in ix])
    return x.to(device), y.to(device)

# 演示一个 batch
x_demo, y_demo = get_batch('train')
print(f"📦 Batch 输入 shape: {x_demo.shape}  (batch_size × block_size)")
print(f"📦 Batch 目标 shape: {y_demo.shape}")
print(f"📦 输入前 10 个 token: {x_demo[0, :10].tolist()}")
print(f"📦 目标前 10 个 token: {y_demo[0, :10].tolist()}")
print(f"📦 解码输入: '{decode(x_demo[0, :20].tolist())}'")
print(f"📦 解码目标: '{decode(y_demo[0, :20].tolist())}'")
print(f"   ↑ 注意：目标就是输入左移一位！每个位置预测下一个字符")


# ============================================================
# 第四部分：模型定义 —— 完整的 Decoder-Only Transformer
# ============================================================
print("\n" + "─" * 70)
print("📌 第四部分：模型定义 — Decoder-Only Transformer (GPT)")
print("─" * 70)

class SelfAttention(nn.Module):
    """
    多头自注意力机制（带 Causal Mask）。
    
    拼合了 Day 6（Self-Attention）、Day 7（Multi-Head）、Day 12（Causal Mask）的知识。
    """
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads  # 每个头的维度
        
        # Q, K, V 的投影矩阵（合并为一个大的线性层，效率更高）
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal Mask: 下三角矩阵，防止"偷看未来"
        # 1 = 可以看到, 0 = 不能看到
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )
    
    def forward(self, x):
        B, T, C = x.shape  # batch_size, seq_len, d_model
        
        # 计算 Q, K, V（一次矩阵乘法搞定，比分三次快）
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)  # 各 (B, T, C)
        
        # 重塑为多头格式: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数: Q × K^T / sqrt(d_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # (B, n_heads, T, T)
        
        # 应用 Causal Mask：把未来位置的分数设为 -inf（softmax 后变 0）
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float('-inf')
        )
        
        # Softmax 归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # 加权求和: 注意力权重 × V
        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)
        
        # 合并多头: (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # 输出投影 + Dropout
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    """
    前馈网络（FFN）—— Transformer 的"消化系统"。
    
    Day 9 学过：FFN 的作用是对注意力层提取的信息做非线性变换。
    内部维度通常是 d_model 的 4 倍（先扩展再压缩）。
    """
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),   # 扩展: 128 → 512
            nn.GELU(),                          # 激活函数（GPT 用 GELU）
            nn.Linear(4 * d_model, d_model),    # 压缩: 512 → 128
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    一个完整的 Transformer Block。
    
    拼合了 Day 10（LayerNorm）、Day 11（残差连接）的知识：
    x → LayerNorm → Attention → + 残差 → LayerNorm → FFN → + 残差
    """
    def __init__(self, d_model, n_heads, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)   # Pre-LN（GPT 风格）
        self.attn = SelfAttention(d_model, n_heads, block_size, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, dropout)
    
    def forward(self, x):
        # 残差连接 + 注意力
        x = x + self.attn(self.ln1(x))
        # 残差连接 + FFN
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """
    完整的 GPT 模型 — Decoder-Only Transformer。
    
    架构:
    Token Embedding + Position Embedding
    → N × TransformerBlock
    → LayerNorm
    → Linear(vocab_size)
    → 概率分布
    
    所有零件都是前 15 天学过的！现在它们终于拼在一起了。
    """
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        
        # Token Embedding: 每个 token → d_model 维向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Position Embedding: 每个位置 → d_model 维向量
        self.position_embedding = nn.Embedding(block_size, d_model)
        
        # Dropout
        self.drop = nn.Dropout(dropout)
        
        # N 个 Transformer Block
        self.blocks = nn.Sequential(
            *[TransformerBlock(d_model, n_heads, block_size, dropout) 
              for _ in range(n_layers)]
        )
        
        # 最终的 LayerNorm
        self.ln_f = nn.LayerNorm(d_model)
        
        # 输出层: d_model → vocab_size（语言模型头）
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享（Weight Tying）：Embedding 和输出层共享权重
        # 这是一个常见技巧——输入和输出都在同一个"词汇空间"，
        # 所以让它们用同一套权重既合理又省参数
        self.token_embedding.weight = self.lm_head.weight
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化：正态分布，标准差 0.02（GPT 系列标准做法）"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        """
        前向传播。
        
        参数:
            idx: 输入 token IDs, shape (B, T)
            targets: 目标 token IDs, shape (B, T)，可选
        
        返回:
            logits: (B, T, vocab_size) 每个位置的预测
            loss: 如果提供了 targets，返回交叉熵损失
        """
        B, T = idx.shape
        
        # Token Embedding + Position Embedding
        tok_emb = self.token_embedding(idx)               # (B, T, d_model)
        pos_emb = self.position_embedding(
            torch.arange(T, device=device)
        )                                                   # (T, d_model)
        x = self.drop(tok_emb + pos_emb)                   # (B, T, d_model)
        
        # Transformer Blocks
        x = self.blocks(x)                                  # (B, T, d_model)
        
        # 最终 LayerNorm
        x = self.ln_f(x)                                    # (B, T, d_model)
        
        # 输出层 → logits
        logits = self.lm_head(x)                            # (B, T, vocab_size)
        
        # 计算损失（如果提供了目标）
        loss = None
        if targets is not None:
            # 交叉熵损失：衡量预测和真实之间的差距
            # view(-1, ...) 把 batch 和 seq_len 维度展平
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),   # (B*T, vocab_size)
                targets.view(-1)                      # (B*T,)
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归生成文本。
        
        参数:
            idx: 初始上下文, shape (B, T)
            max_new_tokens: 要生成的新 token 数
            temperature: 采样温度（越高越随机，越低越确定）
            top_k: 只从概率最高的 k 个 token 中采样（可选）
        
        返回:
            生成的完整序列（含初始上下文）
        """
        self.eval()  # 切换到评估模式（关闭 Dropout）
        
        for _ in range(max_new_tokens):
            # 如果序列太长，截断到 block_size（我们的模型只训练了这么长的上下文）
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # 前向传播，得到最后一个位置的 logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # 只取最后一个位置: (B, vocab_size)
            
            # 温度缩放：控制随机性
            # temperature > 1 → 更随机（更有创意）
            # temperature < 1 → 更确定（更保守）
            # temperature → 0 → 贪心（总是选概率最高的）
            logits = logits / temperature
            
            # Top-K 过滤：只保留概率最高的 K 个 token
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Softmax → 概率分布
            probs = F.softmax(logits, dim=-1)
            
            # 从概率分布中采样一个 token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # 拼接到序列末尾
            idx = torch.cat([idx, idx_next], dim=1)  # (B, T+1)
        
        self.train()  # 恢复训练模式
        return idx


# 创建模型
model = MiniGPT(
    vocab_size=config['vocab_size'],
    d_model=config['d_model'],
    n_heads=config['n_heads'],
    n_layers=config['n_layers'],
    block_size=config['block_size'],
    dropout=config['dropout'],
).to(device)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🏗️ 模型架构: MiniGPT")
print(f"📊 总参数量: {total_params:,}")
print(f"📊 可训练参数: {trainable_params:,}")
print(f"📊 参数量约: {total_params / 1e6:.2f}M（GPT-3: 175,000M）")
print(f"📊 模型大小约: {total_params * 4 / 1024 / 1024:.1f} MB (float32)")


# ============================================================
# 第五部分：训练前的生成（随机权重）
# ============================================================
print("\n" + "─" * 70)
print("📌 第五部分：训练前生成（随机权重 — 模型在胡言乱语）")
print("─" * 70)

# 用随机权重生成一段文本（作为对照）
context = torch.zeros(1, 1, dtype=torch.long, device=device)  # 起始 token = '\n'
random_output = model.generate(context, max_new_tokens=200, temperature=0.8)[0].tolist()
print(f"🎲 随机模型输出:\n{decode(random_output)}")
print("   ↑ 完全是乱码！因为权重是随机的，预测也是随机的")


# ============================================================
# 第六部分：学习率调度（Warmup + Cosine Decay）
# ============================================================
print("\n" + "─" * 70)
print("📌 第六部分：学习率调度 — Warmup + Cosine Decay")
print("─" * 70)

def get_lr(iteration):
    """
    学习率调度函数：Warmup + Cosine Decay
    
    阶段 1: Warmup（预热）
        - 前 warmup_iters 步，学习率从 0 线性增加到 max_lr
        - 防止训练初期梯度不稳定导致参数震荡
    
    阶段 2: Cosine Decay（余弦衰减）
        - 从 max_lr 按余弦曲线衰减到 min_lr
        - 训练后期用小学习率精细调整参数
    
    阶段 3: 恒定最小学习率
        - 衰减完成后保持 min_lr
    """
    max_lr = config['learning_rate']
    min_lr = max_lr * 0.1  # 最小学习率 = 10% 的最大学习率
    warmup_iters = config['warmup_iters']
    decay_iters = config['max_iters']
    
    # 阶段 1: Warmup
    if iteration < warmup_iters:
        return max_lr * (iteration + 1) / warmup_iters
    # 阶段 3: 超过衰减周期，保持最小学习率
    if iteration > decay_iters:
        return min_lr
    # 阶段 2: Cosine Decay
    progress = (iteration - warmup_iters) / (decay_iters - warmup_iters)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# 打印几个关键点的学习率
print("📈 学习率调度:")
for i in [0, 50, 100, 500, 1000, 2000, 3000]:
    print(f"   Step {i:4d}: lr = {get_lr(i):.6f}")


# ============================================================
# 第七部分：训练循环
# ============================================================
print("\n" + "─" * 70)
print("📌 第七部分：训练循环")
print("─" * 70)

# 优化器（AdamW — Adam 的改进版，更好的权重衰减）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config['learning_rate'],
    betas=(0.9, 0.95),  # GPT 系列标准配置
    weight_decay=0.1,     # 权重衰减（L2 正则化）
)

@torch.no_grad()
def estimate_loss():
    """评估训练集和验证集上的平均损失"""
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(20):  # 评估 20 个 batch 取平均
            x, y = get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out

# 训练循环
print("🚀 开始训练...\n")
print(f"{'Step':>6} | {'Train Loss':>10} | {'Val Loss':>10} | {'Perplexity':>12} | {'LR':>10}")
print("-" * 65)

train_losses = []  # 记录训练损失，用于绘制曲线
val_losses = []

t0 = time.time()
best_val_loss = float('inf')

for iter_num in range(config['max_iters']):
    # 每 eval_interval 步评估一次
    if iter_num % config['eval_interval'] == 0 or iter_num == config['max_iters'] - 1:
        losses = estimate_loss()
        ppl = math.exp(losses['val'])  # 困惑度 = exp(loss)
        lr = get_lr(iter_num)
        print(f"{iter_num:>6d} | {losses['train']:>10.4f} | {losses['val']:>10.4f} | {ppl:>12.2f} | {lr:>10.6f}")
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        
        # 保存最佳模型
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
    
    # 更新学习率（warmup + cosine decay）
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # 采样一个 batch
    xb, yb = get_batch('train')
    
    # 前向传播
    logits, loss = model(xb, yb)
    
    # 反向传播
    optimizer.zero_grad(set_to_none=True)  # 清空梯度（比 zero_grad() 更高效）
    loss.backward()
    
    # 梯度裁剪：防止梯度爆炸
    # 把所有参数的梯度范数限制在 1.0 以内
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 更新权重
    optimizer.step()

t1 = time.time()
print(f"\n⏱️ 训练完成！耗时: {t1 - t0:.1f} 秒")
print(f"📊 最终训练损失: {train_losses[-1]:.4f}")
print(f"📊 最终验证损失: {val_losses[-1]:.4f}")
print(f"📊 最终困惑度: {math.exp(val_losses[-1]):.2f}")
print(f"   （困惑度 = 模型平均在每个位置考虑多少个候选词，越低越好）")


# ============================================================
# 第八部分：损失曲线可视化
# ============================================================
print("\n" + "─" * 70)
print("📌 第八部分：损失曲线（ASCII 可视化）")
print("─" * 70)

def plot_loss_ascii(train_losses, val_losses, width=60, height=15):
    """用 ASCII 字符画一条损失曲线"""
    all_losses = train_losses + val_losses
    max_loss = max(all_losses)
    min_loss = min(all_losses)
    loss_range = max_loss - min_loss if max_loss != min_loss else 1.0
    
    print(f"\n损失曲线（Train=■, Val=□）")
    print(f"最大损失: {max_loss:.4f}, 最小损失: {min_loss:.4f}")
    
    # 为每行构建内容
    for row in range(height, 0, -1):
        threshold = min_loss + loss_range * row / height
        line = f"{threshold:6.3f} |"
        
        # 训练损失
        n_train = len(train_losses)
        train_positions = []
        for i, loss in enumerate(train_losses):
            col = int(i / max(n_train - 1, 1) * (width - 1))
            loss_height = (loss - min_loss) / loss_range * height
            train_positions.append((col, loss_height))
        
        # 验证损失
        val_positions = []
        for i, loss in enumerate(val_losses):
            col = int(i / max(len(val_losses) - 1, 1) * (width - 1))
            loss_height = (loss - min_loss) / loss_range * height
            val_positions.append((col, loss_height))
        
        for col in range(width):
            char = ' '
            for pos, h in train_positions:
                if pos == col and abs(h - row) < 0.6:
                    char = '■'
                    break
            if char == ' ':
                for pos, h in val_positions:
                    if pos == col and abs(h - row) < 0.6:
                        char = '□'
                        break
            line += char
        
        print(line)
    
    print(f"{'':>6} +" + "─" * width)
    print(f"{'':>6}  Step 0" + " " * (width - 15) + f"Step {config['max_iters']}")

plot_loss_ascii(train_losses, val_losses)


# ============================================================
# 第九部分：训练后的生成（见证"涌现"）
# ============================================================
print("\n" + "─" * 70)
print("📌 第九部分：训练后生成 — 模型学会了莎士比亚风格！")
print("─" * 70)

# 不同温度的生成对比
print("\n🌡️ 不同温度的生成对比:\n")

context = torch.zeros(1, 1, dtype=torch.long, device=device)

for temp in [0.3, 0.8, 1.5]:
    output = model.generate(context, max_new_tokens=300, temperature=temp, top_k=40)[0]
    text_out = decode(output.tolist())
    print(f"🌡️ Temperature = {temp}（{'保守/确定' if temp < 0.5 else '平衡' if temp < 1.0 else '创意/随机'}）:")
    print(f"{'-' * 50}")
    # 只打印前 300 个字符
    print(text_out[:300])
    print()

# 用一个莎士比亚风格的起始文本
print("\n🎭 用特定上下文生成:\n")
prompt = "ROMEO:"
prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
output = model.generate(prompt_ids, max_new_tokens=200, temperature=0.8, top_k=40)[0]
print(f"输入: '{prompt}'")
print(f"生成:\n{decode(output.tolist())[:300]}\n")


# ============================================================
# 第十部分：模型参数解剖
# ============================================================
print("─" * 70)
print("📌 第十部分：模型参数解剖 — 权重都在哪里？")
print("─" * 70)

# 统计每层的参数量
print(f"\n{'层':>30} | {'参数量':>10} | {'占比':>6}")
print("-" * 55)

layer_params = {}
for name, param in model.named_parameters():
    # 提取层类别
    parts = name.split('.')
    if 'token_embedding' in name:
        layer = 'Token Embedding'
    elif 'position_embedding' in name:
        layer = 'Position Embedding'
    elif 'blocks' in name:
        block_idx = int(parts[1]) + 1
        sublayer = parts[2]
        if sublayer == 'attn' or sublayer == 'ln1':
            layer = f'Block {block_idx} - Attention'
        elif sublayer == 'ffn' or sublayer == 'ln2':
            layer = f'Block {block_idx} - FFN'
        else:
            layer = f'Block {block_idx} - Other'
    elif 'ln_f' in name:
        layer = 'Final LayerNorm'
    elif 'lm_head' in name:
        layer = 'LM Head (shared w/ emb)'
    else:
        layer = name
    
    if layer not in layer_params:
        layer_params[layer] = 0
    layer_params[layer] += param.numel()

for layer, count in layer_params.items():
    pct = count / total_params * 100
    print(f"{layer:>30} | {count:>10,} | {pct:>5.1f}%")

print("-" * 55)
print(f"{'总计':>30} | {total_params:>10,} | 100.0%")


# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("🎉 Day 16 总结")
print("=" * 70)
print("""
我们做了什么？

1. 📊 准备数据：Shakespeare 全文 → 字符级编码 → 训练/验证集划分
2. 🔤 字符级 Tokenizer：每个字符 = 1 个 token，词汇表 65 个
3. 🏗️ 搭建模型：MiniGPT = Token Embed + Pos Embed + 4×Transformer Block + LM Head
4. 🔄 训练循环：前向 → 损失 → 反向 → 梯度裁剪 → 更新，重复 3000 次
5. 📉 学习率调度：Warmup（预热）+ Cosine Decay（余弦衰减）
6. 🎭 文本生成：从随机乱码到莎士比亚风格，见证"涌现"

关键洞察：

- 预训练 = 大量阅读 + 下一个词预测，逼迫模型学会语言
- Transformer 所有零件终于拼在一起了！
- 损失下降 = 模型对正确答案越来越"不惊讶"
- 能力涌现：统计规律 → 单词结构 → 短语搭配 → 句子模式
- 和 GPT-3/GPT-4 的差距 = 规模，不是原理

下一步：SFT（监督微调）— 让模型学会"听懂指令"。
""")
