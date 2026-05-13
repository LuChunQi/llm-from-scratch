"""
Day 12: Decoder-Only 架构 — Causal Mask 与自回归生成
=====================================================

本实验包含 5 个部分：
1. Causal Mask 可视化 — 看清楚掩码长什么样
2. 有/无 Causal Mask 的注意力对比
3. 完整 Decoder-Only Block 实现
4. 自回归文本生成
5. Teacher Forcing vs Free Running 对比
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


# ============================================================
# 1. Causal Mask 可视化
# ============================================================
def demo_causal_mask():
    """展示 Causal Mask 的结构和作用"""
    print("=" * 70)
    print("📌 实验 1: Causal Mask 可视化")
    print("=" * 70)

    seq_len = 6  # 序列长度为 6

    # 方法：用 torch.triu 生成上三角掩码
    # diagonal=1 表示从主对角线上一格开始取上三角
    # 结果：上三角（不含对角线）为 True，其余为 False
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

    print("\n🔹 Causal Mask 矩阵 (True = 被遮住，不能看到):")
    print("   列代表 '被看的位置'，行代表 '正在看的位置'")
    print("   行标 = 当前位置，列标 = 目标位置\n")

    # 打印表头
    header = "      " + "  ".join([f"pos{i}" for i in range(seq_len)])
    print(header)
    for i in range(seq_len):
        row_str = f"pos{i}  "
        for j in range(seq_len):
            if mask[i][j]:
                row_str += "  ✗   "  # 被遮住
            else:
                row_str += "  ✓   "  # 可以看到
        print(row_str)

    print("\n🔹 解读：")
    print("   pos0 只能看到 pos0（自己）")
    print("   pos1 可以看到 pos0 和 pos1")
    print("   pos5 可以看到 pos0~pos5（全部）")
    print("   这就是'只看过去，不看未来'")

    # 展示掩码对注意力分数的影响
    print("\n🔹 模拟注意力分数被掩码处理的过程：")
    torch.manual_seed(42)
    scores = torch.randn(seq_len, seq_len)
    print("\n   原始注意力分数 (QK^T / sqrt(d)):")
    print("   ", np.array2string(scores.numpy(), precision=2, suppress_small=True))

    # 应用掩码：把 True 的位置设为 -inf
    masked_scores = scores.masked_fill(mask, float('-inf'))
    print("\n   加掩码后 (上三角 → -inf):")
    print("   ", np.array2string(masked_scores.numpy(), precision=2, suppress_small=True))

    # Softmax 后
    attn_weights = F.softmax(masked_scores, dim=-1)
    print("\n   Softmax 后的注意力权重:")
    print("   ", np.array2string(attn_weights.numpy(), precision=3, suppress_small=True))

    print("\n   ✅ 被掩码的位置权重精确为 0.000 → 零信息泄漏！")

    # 统计每个位置能看到多少 token
    print("\n🔹 每个位置可以看到的 token 数量：")
    visible_counts = (~mask).sum(dim=-1)
    for i in range(seq_len):
        bar = "█" * visible_counts[i].item()
        print(f"   pos{i}: {visible_counts[i].item()} tokens  {bar}")

    print()


# ============================================================
# 2. 有/无 Causal Mask 的注意力对比
# ============================================================
def demo_masked_vs_unmasked():
    """对比有/无 Causal Mask 的注意力模式"""
    print("=" * 70)
    print("📌 实验 2: 有/无 Causal Mask 的注意力模式对比")
    print("=" * 70)

    torch.manual_seed(42)
    seq_len = 6
    d_model = 8

    # 模拟 Q, K, V
    x = torch.randn(seq_len, d_model)
    W_q = torch.randn(d_model, d_model) * 0.3
    W_k = torch.randn(d_model, d_model) * 0.3
    W_v = torch.randn(d_model, d_model) * 0.3

    Q = x @ W_q  # (6, 8)
    K = x @ W_k  # (6, 8)
    V = x @ W_v  # (6, 8)

    scores = Q @ K.T / math.sqrt(d_model)  # (6, 6)

    # 无掩码的全局注意力
    attn_full = F.softmax(scores, dim=-1)

    # 有 Causal Mask 的注意力
    causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    masked_scores = scores.masked_fill(causal_mask, float('-inf'))
    attn_causal = F.softmax(masked_scores, dim=-1)

    print("\n🔹 无掩码的全局注意力（BERT 风格）：")
    print("   每个 token 都能看到所有其他 token\n")
    tokens = ["[BOS]", "我", "爱", "北", "京", "[EOS]"]
    header = "         " + "  ".join([f"{t:>5}" for t in tokens])
    print(header)
    for i, token in enumerate(tokens):
        row = f"  {token:>5}  "
        for j in range(seq_len):
            row += f" {attn_full[i][j]:.3f}"
        print(row)

    print("\n🔹 有 Causal Mask 的注意力（GPT 风格）：")
    print("   每个 token 只能看到自己和之前的 token\n")
    print(header)
    for i, token in enumerate(tokens):
        row = f"  {token:>5}  "
        for j in range(seq_len):
            if attn_causal[i][j] == 0:
                row += "   ---"
            else:
                row += f" {attn_causal[i][j]:.3f}"
        print(row)

    print("\n🔹 关键区别：")
    print("   全局注意力：'北' 可以看到 '[EOS]' 来理解上下文")
    print("   因果注意力：'北' 只能看到 '[BOS] 我 爱 北'，不能偷看 '京' 和 '[EOS]'")
    print("   → 全局注意力更适合理解任务，因果注意力更适合生成任务")
    print()


# ============================================================
# 3. 完整 Decoder-Only Block
# ============================================================
class CausalSelfAttention(nn.Module):
    """带 Causal Mask 的多头自注意力"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads  # 每个 head 的维度

        # Q, K, V 的线性投影（一次算完所有 head）
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        返回: (batch, seq_len, d_model)
        """
        B, T, C = x.shape  # batch, seq_len, d_model

        # 线性投影并拆分成多头
        # (B, T, C) → (B, T, C) → (B, T, n_heads, d_head) → (B, n_heads, T, d_head)
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 计算注意力分数: QK^T / sqrt(d_head)
        # (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) → (B, n_heads, T, T)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        # 🔑 关键步骤：应用 Causal Mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Softmax 得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)  # (B, n_heads, T, T)

        # 加权求和
        # (B, n_heads, T, T) @ (B, n_heads, T, d_head) → (B, n_heads, T, d_head)
        out = attn_weights @ V

        # 合并多头: (B, n_heads, T, d_head) → (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 输出投影
        return self.W_o(out)


class FeedForward(nn.Module):
    """前馈网络：线性 → GELU → 线性"""

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # 标准的 4 倍扩展
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class DecoderOnlyBlock(nn.Module):
    """完整的 Decoder-Only Transformer Block（Pre-Norm 版本）"""

    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        # Pre-Norm: LayerNorm 在子层之前
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        结构: x → LN → Causal Attn → (+x) → LN → FFN → (+x)
        """
        # 残差连接 + 因果自注意力
        x = x + self.attn(self.ln1(x))  # 注意：先 LayerNorm，再 Attention，再加回 x
        # 残差连接 + FFN
        x = x + self.ffn(self.ln2(x))   # 先 LayerNorm，再 FFN，再加回 x
        return x


class MiniGPT(nn.Module):
    """迷你 GPT 模型：Embedding + N × DecoderBlock + LM Head"""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token Embedding: 把 token ID 映射到 d_model 维向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # Position Embedding: 可学习的位置编码
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # N 层 Decoder Block
        self.blocks = nn.ModuleList([
            DecoderOnlyBlock(d_model, n_heads) for _ in range(n_layers)
        ])

        # 最终的 LayerNorm
        self.ln_final = nn.LayerNorm(d_model)

        # 语言模型头：把 d_model 维向量投影回 vocab_size 维的 logits
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享：token embedding 和 lm_head 共享权重
        self.token_embedding.weight = self.lm_head.weight

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """使用 GPT-2 风格的初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx):
        """
        idx: (batch, seq_len) — token ID 序列
        返回: (batch, seq_len, vocab_size) — 每个位置预测下一个 token 的 logits
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"序列长度 {T} 超过最大长度 {self.max_seq_len}"

        # Token Embedding + Position Embedding
        tok_emb = self.token_embedding(idx)            # (B, T, d_model)
        pos = torch.arange(0, T, device=idx.device)    # (T,)
        pos_emb = self.pos_embedding(pos)               # (T, d_model)
        x = tok_emb + pos_emb                           # (B, T, d_model)

        # 通过 N 层 Decoder Block
        for block in self.blocks:
            x = block(x)  # 每层都有 Causal Mask

        # 最终 LayerNorm
        x = self.ln_final(x)  # (B, T, d_model)

        # 投影到词表大小
        logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits


def demo_decoder_only_block():
    """演示完整的 Decoder-Only Block"""
    print("=" * 70)
    print("📌 实验 3: 完整 Decoder-Only Block")
    print("=" * 70)

    torch.manual_seed(42)

    d_model = 32
    n_heads = 4
    seq_len = 6
    batch_size = 2

    x = torch.randn(batch_size, seq_len, d_model)

    # 创建 Decoder-Only Block
    block = DecoderOnlyBlock(d_model, n_heads)

    print(f"\n🔹 Block 配置:")
    print(f"   d_model = {d_model}, n_heads = {n_heads}, d_head = {d_model // n_heads}")
    print(f"   输入形状: {x.shape}")

    # Forward
    out = block(x)
    print(f"   输出形状: {out.shape}")
    print(f"   ✅ 输入输出形状一致，残差连接保证了维度不变")

    # 统计参数量
    n_params = sum(p.numel() for p in block.parameters())
    print(f"   参数量: {n_params:,}")

    print()


# ============================================================
# 4. 自回归文本生成
# ============================================================
def demo_autoregressive_generation():
    """用 MiniGPT 做自回归生成"""
    print("=" * 70)
    print("📌 实验 4: 自回归文本生成")
    print("=" * 70)

    torch.manual_seed(42)

    # 构建一个简单的"词表"和训练数据
    # 用一个简单的童谣作为训练语料
    text = "我爱北京天安门天安门上太阳升伟大领袖毛主席指引我们向前进我爱北京天安门"
    # 重复多次让模型过拟合来学习模式
    text = text * 20

    # 构建词表
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    print(f"\n🔹 语料信息:")
    print(f"   词表大小: {vocab_size} 个字符: {''.join(chars)}")
    print(f"   语料长度: {len(text)} 字符")

    # 编码整个语料
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    print(f"   编码后: {data[:20].tolist()} ...")

    # 创建模型
    model = MiniGPT(
        vocab_size=vocab_size,
        d_model=64,        # 嵌入维度
        n_heads=4,         # 注意力头数
        n_layers=3,        # 层数
        max_seq_len=64,    # 最大序列长度
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔹 模型配置:")
    print(f"   d_model=64, n_heads=4, n_layers=3")
    print(f"   总参数量: {total_params:,}")

    # 训练模型（简单的过拟合训练，让它学会这个模式）
    print(f"\n🔹 开始训练（让模型过拟合学会这个童谣的模式）...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    block_size = 32  # 训练时的上下文窗口大小

    model.train()
    for step in range(500):
        # 随机采样一个片段
        ix = torch.randint(0, len(data) - block_size, (1,)).item()
        x = data[ix:ix + block_size].unsqueeze(0)      # (1, block_size)
        y = data[ix + 1:ix + block_size + 1].unsqueeze(0)  # 下一个 token

        # Forward
        logits = model(x)  # (1, block_size, vocab_size)

        # 计算交叉熵损失
        loss = F.cross_entropy(
            logits.view(-1, vocab_size),  # (block_size, vocab_size)
            y.view(-1)                     # (block_size,)
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"   Step {step+1:4d} | Loss: {loss.item():.4f}")

    print(f"\n   ✅ 训练完成！最终 Loss: {loss.item():.4f}")

    # 自回归生成
    print(f"\n🔹 自回归生成过程:")
    model.eval()

    # 用 "我爱" 作为起始 prompt
    prompt = "我爱北京天安"
    context = torch.tensor([char_to_idx[c] for c in prompt], dtype=torch.long).unsqueeze(0)

    print(f"   输入 prompt: '{prompt}'")
    print(f"   生成过程:")

    generated = prompt
    max_new_tokens = 30

    with torch.no_grad():
        for i in range(max_new_tokens):
            # 取最后 block_size 个 token（防止超出最大长度）
            idx_cond = context[:, -block_size:]
            # Forward pass
            logits = model(idx_cond)
            # 只取最后一个位置的 logits
            logits = logits[:, -1, :]  # (1, vocab_size)
            # 用温度缩放后的 softmax 采样
            probs = F.softmax(logits / 0.8, dim=-1)  # 温度 0.8 让分布更尖锐
            # 采样下一个 token
            next_idx = torch.multinomial(probs, num_samples=1)  # (1, 1)
            # 解码看看
            next_char = idx_to_char[next_idx.item()]
            generated += next_char
            # 追加到 context
            context = torch.cat([context, next_idx], dim=1)

            if (i + 1) % 5 == 0 or i == 0:
                print(f"   Step {i+1:2d}: 已生成 '{generated}'")

    print(f"\n   🎉 最终生成: '{generated}'")
    print()


# ============================================================
# 5. Teacher Forcing vs Free Running 对比
# ============================================================
def demo_teacher_forcing_vs_free_running():
    """对比 Teacher Forcing 和 Free Running 的区别"""
    print("=" * 70)
    print("📌 实验 5: Teacher Forcing vs Free Running")
    print("=" * 70)

    torch.manual_seed(42)

    vocab_size = 10
    d_model = 16
    seq_len = 8

    model = MiniGPT(vocab_size=vocab_size, d_model=d_model, n_heads=2, n_layers=2, max_seq_len=32)

    # 模拟训练数据：一个简单的模式 [0, 1, 2, 3, 4, 5, 6, 7]
    train_seq = torch.arange(seq_len).unsqueeze(0)  # (1, 8)

    print(f"\n🔹 训练序列: {train_seq.tolist()[0]}")
    print(f"   训练目标: 每个位置预测下一个数字")

    # --- Teacher Forcing ---
    # 训练时：输入真实序列，计算每个位置的损失
    print(f"\n🔹 Teacher Forcing（训练模式）：")
    print(f"   输入:     {train_seq.tolist()[0]}")
    print(f"   目标:     {train_seq.tolist()[0][1:]} + [EOS]")

    logits = model(train_seq)  # (1, 8, vocab_size)
    # 每个位置的预测
    predictions = logits.argmax(dim=-1).squeeze(0).tolist()
    print(f"   预测:     {predictions}")
    print(f"   → 每个位置同时计算预测，一次 forward 搞定所有位置")
    print(f"   → 输入永远是正确的序列，不依赖模型自己的预测")

    # --- Free Running ---
    # 推理时：用模型自己的预测作为下一步的输入
    print(f"\n🔹 Free Running（推理模式 / 自回归）：")

    # 从第一个 token 开始
    context = train_seq[:, :1]  # 只取 [0]
    print(f"   起始输入: [{context.tolist()[0][0]}]")

    generated = [context.tolist()[0][0]]
    with torch.no_grad():
        for step in range(seq_len - 1):
            logits = model(context)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # 取概率最大的
            generated.append(next_token.item())
            context = torch.cat([context, next_token], dim=1)
            print(f"   Step {step+1}: 输入 {context.tolist()[0]} → 预测 {next_token.item()}")

    print(f"\n   Teacher Forcing 预测: {predictions}")
    print(f"   Free Running 生成:   {generated}")
    print(f"   真实序列:            {train_seq.tolist()[0]}")

    print(f"\n   📝 关键区别:")
    print(f"   - Teacher Forcing: 每步输入都是真实的 → 预测可能看起来还行")
    print(f"   - Free Running: 每步输入是上一步的预测 → 一步错，步步错（误差累积）")
    print(f"   - 这就是 Exposure Bias 问题！模型在训练时从没见过自己的错误输出")
    print()


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "🧠" * 30)
    print("   Day 12: Decoder-Only 架构 — Causal Mask 与自回归生成")
    print("🧠" * 30 + "\n")

    # 实验 1: Causal Mask 可视化
    demo_causal_mask()

    # 实验 2: 有/无 Causal Mask 的注意力对比
    demo_masked_vs_unmasked()

    # 实验 3: 完整 Decoder-Only Block
    demo_decoder_only_block()

    # 实验 4: 自回归文本生成
    demo_autoregressive_generation()

    # 实验 5: Teacher Forcing vs Free Running
    demo_teacher_forcing_vs_free_running()

    print("=" * 70)
    print("🏁 所有实验完成！")
    print("=" * 70)
    print("\n🔑 核心要点回顾:")
    print("1. Causal Mask = 上三角 -inf 矩阵 → Softmax 后未来位置权重为 0")
    print("2. Decoder-Only = Causal Self-Attention + FFN + 残差 + LayerNorm")
    print("3. 训练时并行（一次 forward 所有位置），推理时串行（逐步生成）")
    print("4. Teacher Forcing: 训练用真实输入 vs Free Running: 推理用自己预测")
    print("5. Next Token Prediction 是 Decoder-Only 的唯一训练目标")
    print()


if __name__ == "__main__":
    main()
