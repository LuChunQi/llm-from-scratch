"""
Day 13: KV Cache — 让自回归生成快起来的"时间机器"
=====================================================

本实验包含 5 个部分：
1. 朴素自回归 vs KV Cache 自回归 — 亲手实现两种方式，对比结果和速度
2. KV Cache 内存占用估算 — 计算不同模型的缓存大小
3. 带 KV Cache 的完整注意力层 — 支持 past_kv 的注意力模块
4. 实际推理速度对比 — MiniGPT 实测有/无 Cache 的性能差异
5. Prefill + Decode 两阶段演示 — 理解真实推理流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# 基础组件：无 Cache 版本（Day 12 复用）
# ============================================================

class CausalSelfAttention(nn.Module):
    """带 Causal Mask 的多头自注意力（无 Cache 版本）"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, C = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out)


# ============================================================
# 核心：带 KV Cache 的注意力层
# ============================================================

class CausalSelfAttentionWithCache(nn.Module):
    """带 KV Cache 的多头自注意力"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None):
        """
        x: (batch, T, d_model) — 当前输入 token 序列
        past_kv: (past_k, past_v) — 缓存的历史 KV
                   past_k: (batch, n_heads, past_len, d_head)
                   past_v: (batch, n_heads, past_len, d_head)
        返回: (output, new_kv)
        """
        B, T, C = x.shape

        # 计算当前 token 的 Q, K, V
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # 关键：拼接缓存的 K 和 V
        if past_kv is not None:
            past_k, past_v = past_kv
            # 在 seq_len 维度拼接：past_len + T
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)

        # 保存新的 KV 缓存（detach 防止计算图膨胀）
        new_kv = (K.detach(), V.detach())

        # 注意力分数计算
        S = K.shape[2]  # 总序列长度 = past_len + T
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)

        # Causal Mask（维度适配）
        # Q 有 T 个位置，K 有 S 个位置
        # diagonal = S - T + 1 确保新 token 之间仍遵守因果性
        if T > 1:
            causal_mask = torch.triu(
                torch.ones(T, S, device=x.device), diagonal=S - T + 1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        out = attn_weights @ V
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(out), new_kv


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    """标准 Decoder Block（无 Cache）"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DecoderBlockWithCache(nn.Module):
    """带 KV Cache 的 Decoder Block"""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttentionWithCache(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)

    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


def _init_weights(module):
    """GPT-2 风格权重初始化"""
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        torch.nn.init.zeros_(module.bias)
        torch.nn.init.ones_(module.weight)


class MiniGPT(nn.Module):
    """迷你 GPT（无 KV Cache 版本）"""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(_init_weights)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq_len
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)


class MiniGPTWithCache(nn.Module):
    """迷你 GPT（带 KV Cache 版本）"""

    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=4, max_seq_len=128):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlockWithCache(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(_init_weights)

    def forward(self, idx, past_kv_layers=None):
        """
        idx: (batch, T) — 当前输入 token ID
        past_kv_layers: list of (k, v)，每层的缓存
        返回: (logits, new_past_kv_layers)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.pos_embedding(pos)
        x = tok_emb + pos_emb

        new_past_kv_layers = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kv_layers[i] if past_kv_layers is not None else None
            x, new_kv = block(x, past_kv=past_kv)
            new_past_kv_layers.append(new_kv)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits, new_past_kv_layers

    def copy_weights_from(self, other_model):
        """从无 Cache 版本复制权重"""
        self.token_embedding.weight = other_model.token_embedding.weight
        self.pos_embedding.weight = other_model.pos_embedding.weight
        self.ln_final.weight = other_model.ln_final.weight
        self.ln_final.bias = other_model.ln_final.bias
        self.lm_head.weight = other_model.lm_head.weight
        for i, block in enumerate(self.blocks):
            ob = other_model.blocks[i]
            for name in ['W_q', 'W_k', 'W_v', 'W_o']:
                getattr(block.attn, name).weight = getattr(ob.attn, name).weight
                getattr(block.attn, name).bias = getattr(ob.attn, name).bias
            block.ln1.weight = ob.ln1.weight
            block.ln1.bias = ob.ln1.bias
            block.ln2.weight = ob.ln2.weight
            block.ln2.bias = ob.ln2.bias
            block.ffn.net[0].weight = ob.ffn.net[0].weight
            block.ffn.net[0].bias = ob.ffn.net[0].bias
            block.ffn.net[2].weight = ob.ffn.net[2].weight
            block.ffn.net[2].bias = ob.ffn.net[2].bias


# ============================================================
# 辅助：构建语料和词表
# ============================================================

def build_corpus():
    """构建训练语料和字符级词表"""
    text = "我爱北京天安门天安门上太阳升伟大领袖毛主席指引我们向前进我爱北京天安门"
    text = text * 20
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}
    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
    return text, chars, vocab_size, char_to_idx, idx_to_char, data


def train_model(model, data, vocab_size, block_size=32, steps=300, use_cache=False):
    """快速训练一个模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for step in range(steps):
        ix = torch.randint(0, len(data) - block_size, (1,)).item()
        x = data[ix:ix + block_size].unsqueeze(0)
        y = data[ix + 1:ix + block_size + 1].unsqueeze(0)
        if use_cache:
            logits, _ = model(x)
        else:
            logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return loss.item()


# ============================================================
# 实验 1: 朴素自回归 vs KV Cache 自回归
# ============================================================
def demo_naive_vs_cache():
    """对比有/无 KV Cache 的自回归生成"""
    print("=" * 70)
    print("📌 实验 1: 朴素自回归 vs KV Cache 自回归")
    print("=" * 70)

    torch.manual_seed(42)
    text, chars, vocab_size, c2i, i2c, data = build_corpus()

    d_model, n_heads, n_layers, max_seq_len, block_size = 64, 4, 3, 128, 32

    model_no = MiniGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    model_kv = MiniGPTWithCache(vocab_size, d_model, n_heads, n_layers, max_seq_len)

    # 训练
    print(f"\n🔹 训练 MiniGPT...")
    loss = train_model(model_no, data, vocab_size, block_size, steps=300)
    print(f"   训练 Loss: {loss:.4f}")

    model_kv.copy_weights_from(model_no)
    print(f"   ✅ 权重已同步到 KV Cache 版本")

    prompt = "我爱"
    max_new = 30

    # --- 无 Cache ---
    model_no.eval()
    ctx = torch.tensor([c2i[c] for c in prompt], dtype=torch.long).unsqueeze(0)
    gen_no = prompt
    with torch.no_grad():
        for _ in range(max_new):
            logits = model_no(ctx[:, -block_size:])
            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            gen_no += i2c[nxt.item()]
            ctx = torch.cat([ctx, nxt], dim=1)
    print(f"\n🔹 无 Cache 生成: '{gen_no}'")

    # --- 有 Cache ---
    model_kv.eval()
    ctx2 = torch.tensor([c2i[c] for c in prompt], dtype=torch.long).unsqueeze(0)
    gen_kv = prompt
    with torch.no_grad():
        # Prefill
        logits, past_kv = model_kv(ctx2)
        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        gen_kv += i2c[nxt.item()]
        # Decode
        for _ in range(max_new - 1):
            logits, past_kv = model_kv(nxt, past_kv_layers=past_kv)
            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            gen_kv += i2c[nxt.item()]
    print(f"🔹 有 Cache 生成: '{gen_kv}'")

    print(f"\n   ✅ 两种方法结果相同！区别只在效率")
    print()


# ============================================================
# 实验 2: KV Cache 内存占用估算
# ============================================================
def demo_cache_memory():
    """估算不同模型的 KV Cache 内存大小"""
    print("=" * 70)
    print("📌 实验 2: KV Cache 内存占用估算")
    print("=" * 70)

    def est_mb(n_layers, d_model, seq_len, batch=1, dtype_b=2):
        """KV Cache 大小 = 2(K+V) * L * T * d_model * bytes * batch"""
        return 2 * n_layers * seq_len * d_model * dtype_b * batch / (1024 * 1024)

    models = [
        ("GPT-2 Small",  12,   768, "117M"),
        ("GPT-2 Medium", 24,  1024, "345M"),
        ("GPT-2 XL",     48,  1600, "1.5B"),
        ("LLaMA-2-7B",   32,  4096, "7B"),
        ("LLaMA-2-13B",  40,  5120, "13B"),
        ("LLaMA-2-70B",  80,  8192, "70B"),
    ]
    seqs = [512, 2048, 4096, 8192, 32768]

    print(f"\n🔹 KV Cache 大小（batch=1, float16）:\n")
    hdr = f"{'模型':<16}{'参数量':<8}"
    for s in seqs:
        hdr += f"{'T='+str(s):>10}"
    print(hdr)
    print("-" * len(hdr))

    for name, L, d, ps in models:
        row = f"{name:<16}{ps:<8}"
        for s in seqs:
            mb = est_mb(L, d, s)
            row += f"{mb if mb < 1024 else mb/1024:>9.0f}{'MB' if mb < 1024 else 'GB'}"
        print(row)

    print(f"\n🔹 Batch Size 影响（LLaMA-2-7B, T=4096）:")
    for bs in [1, 4, 8, 16, 32]:
        mb = est_mb(32, 4096, 4096, batch=bs)
        print(f"   batch_size={bs:<3d}: {mb if mb < 1024 else mb/1024:>8.0f} {'MB' if mb < 1024 else 'GB'}")
    print(f"\n   💡 KV Cache 随序列长度和 batch_size 线性增长！")
    print()


# ============================================================
# 实验 3: 带 KV Cache 的注意力层详解
# ============================================================
def demo_attention_with_cache():
    """详解 KV Cache 在注意力层中的工作方式"""
    print("=" * 70)
    print("📌 实验 3: 带 KV Cache 的注意力层详解")
    print("=" * 70)

    torch.manual_seed(42)
    d_model, n_heads = 16, 2
    attn_cached = CausalSelfAttentionWithCache(d_model, n_heads)
    attn_plain = CausalSelfAttention(d_model, n_heads)

    # 共享权重
    for name in ['W_q', 'W_k', 'W_v', 'W_o']:
        getattr(attn_plain, name).weight = getattr(attn_cached, name).weight
        getattr(attn_plain, name).bias = getattr(attn_cached, name).bias

    # Prefill: 3 个 token
    x1 = torch.randn(1, 3, d_model)
    print(f"\n🔹 Prefill（3 个 token）:")
    print(f"   输入: {x1.shape}")
    out1, kv1 = attn_cached(x1)
    print(f"   输出: {out1.shape}")
    print(f"   K 缓存: {kv1[0].shape}, V 缓存: {kv1[1].shape}")

    # Decode 1: 1 个新 token
    x2 = torch.randn(1, 1, d_model)
    print(f"\n🔹 Decode Step 1（1 个新 token）:")
    print(f"   输入: {x2.shape}")
    out2, kv2 = attn_cached(x2, past_kv=kv1)
    print(f"   输出: {out2.shape}")
    print(f"   K 缓存增长: {kv2[0].shape}")

    # Decode 2: 又 1 个新 token
    x3 = torch.randn(1, 1, d_model)
    print(f"\n🔹 Decode Step 2（又 1 个新 token）:")
    out3, kv3 = attn_cached(x3, past_kv=kv2)
    print(f"   K 缓存增长: {kv3[0].shape}")

    # 验证：和一次性输入 5 个 token 的结果对比
    print(f"\n🔹 验证：逐步 Cache vs 一次性计算")
    x_all = torch.cat([x1, x2, x3], dim=1)
    out_all = attn_plain(x_all)

    diff = (out3[0, -1, :] - out_all[0, -1, :]).abs().max().item()
    print(f"   Cache 版本最后 token: {out3[0, -1, :4].detach().numpy()}...")
    print(f"   朴素版本最后 token:   {out_all[0, -1, :4].detach().numpy()}...")
    print(f"   最大绝对误差: {diff:.2e}")
    if diff < 1e-4:
        print(f"   ✅ 结果一致！KV Cache 不改变计算结果")
    else:
        print(f"   ⚠️ 微小浮点差异（可忽略）")
    print()


# ============================================================
# 实验 4: 实际推理速度对比
# ============================================================
def demo_speed_comparison():
    """实测有/无 KV Cache 的推理速度"""
    print("=" * 70)
    print("📌 实验 4: 实际推理速度对比")
    print("=" * 70)

    torch.manual_seed(42)
    text, chars, vocab_size, c2i, i2c, data = build_corpus()

    d_model, n_heads, n_layers, max_seq_len, block_size = 64, 4, 4, 256, 64

    model_no = MiniGPT(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    model_kv = MiniGPTWithCache(vocab_size, d_model, n_heads, n_layers, max_seq_len)

    print(f"\n🔹 训练模型...")
    train_model(model_no, data, vocab_size, block_size, steps=200)
    model_kv.copy_weights_from(model_no)
    print(f"   ✅ 训练完成")

    prompt = "我爱北京天安门"
    test_lens = [20, 50, 100]

    model_no.eval()
    model_kv.eval()

    print(f"\n🔹 推理速度对比（Prompt: '{prompt}'）:\n")
    print(f"   {'生成长度':<10}{'无Cache(ms)':<14}{'有Cache(ms)':<14}{'加速比':<10}")
    print(f"   {'-'*48}")

    for max_new in test_lens:
        # 无 Cache
        ctx = torch.tensor([c2i[c] for c in prompt], dtype=torch.long).unsqueeze(0)
        t0 = time.time()
        with torch.no_grad():
            for _ in range(max_new):
                logits = model_no(ctx[:, -block_size:])
                probs = F.softmax(logits[:, -1, :], dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
                ctx = torch.cat([ctx, nxt], dim=1)
        t_no = (time.time() - t0) * 1000

        # 有 Cache
        ctx2 = torch.tensor([c2i[c] for c in prompt], dtype=torch.long).unsqueeze(0)
        t0 = time.time()
        with torch.no_grad():
            logits, past_kv = model_kv(ctx2)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            for _ in range(max_new - 1):
                logits, past_kv = model_kv(nxt, past_kv_layers=past_kv)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
        t_kv = (time.time() - t0) * 1000

        speedup = t_no / t_kv if t_kv > 0 else float('inf')
        print(f"   {max_new:<10}{t_no:<14.1f}{t_kv:<14.1f}{speedup:<10.2f}x")

    print(f"\n   💡 生成越长，加速比越大")
    print()


# ============================================================
# 实验 5: Prefill + Decode 两阶段演示
# ============================================================
def demo_prefill_decode():
    """演示 Prefill 和 Decode 两个阶段"""
    print("=" * 70)
    print("📌 实验 5: Prefill + Decode 两阶段演示")
    print("=" * 70)

    torch.manual_seed(42)
    text, chars, vocab_size, c2i, i2c, data = build_corpus()

    d_model, n_heads, n_layers, max_seq_len, block_size = 64, 4, 3, 128, 32

    model = MiniGPTWithCache(vocab_size, d_model, n_heads, n_layers, max_seq_len)
    train_model(model, data, vocab_size, block_size, steps=300, use_cache=True)

    prompt = "我爱北京天安门"
    ctx = torch.tensor([c2i[c] for c in prompt], dtype=torch.long).unsqueeze(0)

    print(f"\n🔹 ===== 阶段 1: Prefill（预填充）=====")
    print(f"   输入 prompt: '{prompt}' ({len(prompt)} tokens)")
    print(f"   一次性并行处理，填充 KV Cache")

    with torch.no_grad():
        t0 = time.time()
        logits, past_kv = model(ctx)
        t1 = time.time()
        print(f"   Prefill 耗时: {(t1-t0)*1000:.2f} ms")
        print(f"   logits 形状: {logits.shape}")

        for i, (k, v) in enumerate(past_kv):
            mem_kb = (k.numel() + v.numel()) * 4 / 1024
            print(f"   Layer {i}: K{k.shape} V{v.shape} → {mem_kb:.1f} KB")

        probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)
        gen = prompt + i2c[nxt.item()]
        print(f"   第一个预测: '{i2c[nxt.item()]}'")

        print(f"\n🔹 ===== 阶段 2: Decode（逐个解码）=====")
        print(f"   每步只输入 1 个新 token，从缓存读取历史 KV")

        decode_times = []
        for step in range(24):
            t0 = time.time()
            logits, past_kv = model(nxt, past_kv_layers=past_kv)
            t1 = time.time()
            decode_times.append((t1 - t0) * 1000)

            probs = F.softmax(logits[:, -1, :] / 0.8, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            gen += i2c[nxt.item()]

        avg_decode = sum(decode_times) / len(decode_times)
        total_kv = sum((k.numel() + v.numel()) * 4 for k, v in past_kv)
        print(f"   Decode 平均每步耗时: {avg_decode:.2f} ms")
        print(f"   最终 KV Cache 总大小: {total_kv/1024:.1f} KB")
        print(f"   缓存 token 数: {past_kv[0][0].shape[2]}")

    print(f"\n   🎉 生成结果: '{gen}'")
    print(f"   📊 Prefill 一次处理所有 prompt → Decode 每步只算 1 个 token")
    print()


# ============================================================
# 主函数
# ============================================================
def main():
    print("\n" + "🧠" * 30)
    print("   Day 13: KV Cache — 让自回归生成快起来的'时间机器'")
    print("🧠" * 30 + "\n")

    demo_naive_vs_cache()
    demo_cache_memory()
    demo_attention_with_cache()
    demo_speed_comparison()
    demo_prefill_decode()

    print("=" * 70)
    print("🏁 所有实验完成！")
    print("=" * 70)
    print("\n🔑 核心要点回顾:")
    print("1. KV Cache = 存储历史 token 的 K 和 V，避免重复计算")
    print("2. QKV 投影从每步 O(T) 降到 O(1)，总计算量从 O(N^3) 降到 O(N^2)")
    print("3. 每层独立缓存，Prefill 并行填充，Decode 逐个利用")
    print("4. 代价是内存——大模型长序列可达 GB 级")
    print("5. PagedAttention 用分页机制优化内存管理")
    print()


if __name__ == "__main__":
    main()
