#!/usr/bin/env python3
"""
Day 14: MQA & GQA — MHA / GQA / MQA 三合一实现与对比实验

运行方式：python3 mqa_gqa.py

实验内容：
1. 统一注意力实现（一套代码支持 MHA / GQA / MQA）
2. KV Cache 内存占用对比
3. 推理速度实测（自回归生成）
4. 注意力模式可视化
5. expand 操作详解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


# ============================================================
# 1. 统一注意力实现：MHA / GQA / MQA 三合一
# ============================================================

class UnifiedAttention(nn.Module):
    """
    统一的注意力实现，通过 n_kv_heads 参数控制方案：
    - n_kv_heads == n_heads  → 标准 MHA
    - 1 < n_kv_heads < n_heads → GQA
    - n_kv_heads == 1 → MQA
    """

    def __init__(self, d_model, n_heads, n_kv_heads=None):
        """
        参数:
            d_model: 模型维度
            n_heads: Q 的头数（Query heads）
            n_kv_heads: K/V 的头数（Key/Value heads），默认等于 n_heads（MHA）
        """
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        # 默认 n_kv_heads = n_heads，即标准 MHA
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.d_head = d_model // n_heads  # 每个头的维度

        # 每个 Q 头共享的 KV 头数量
        self.n_rep = n_heads // self.n_kv_heads

        # Q 投影：n_heads 个头，每个头 d_head 维
        self.W_q = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        # K 投影：只有 n_kv_heads 个头！这是省内存的关键
        self.W_k = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        # V 投影：只有 n_kv_heads 个头！
        self.W_v = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        # 输出投影
        self.W_o = nn.Linear(n_heads * self.d_head, d_model, bias=False)

    def _expand_kv(self, x):
        """
        将 KV 从 n_kv_heads 扩展到 n_heads（虚拟复制，不占额外内存）
        
        输入: (B, n_kv_heads, T, d_head)
        输出: (B, n_heads, T, d_head)
        
        原理: 每个 KV 头被 n_rep 个 Q 头共享
        通过 view + expand 实现零拷贝扩展
        """
        B, n_kv, T, d_head = x.shape
        if self.n_rep == 1:
            # n_kv_heads == n_heads，无需扩展（标准 MHA）
            return x
        # 第一步：插入一个维度 → (B, n_kv, 1, T, d_head)
        # 第二步：expand 填充 → (B, n_kv, n_rep, T, d_head)
        # 第三步：reshape 合并 → (B, n_kv * n_rep, T, d_head) = (B, n_heads, T, d_head)
        x = x[:, :, None, :, :].expand(B, n_kv, self.n_rep, T, d_head)
        return x.reshape(B, n_kv * self.n_rep, T, d_head)

    def forward(self, x, past_kv=None, use_causal_mask=True):
        """
        前向传播

        参数:
            x: 输入张量 (B, T, d_model)
            past_kv: 上一层的 KV 缓存 (past_k, past_v)，用于自回归推理加速
            use_causal_mask: 是否使用因果 mask（Decoder-Only 需要）

        返回:
            out: 输出张量 (B, T, d_model)
            new_kv: 更新后的 KV 缓存 (new_k, new_v)
        """
        B, T, C = x.shape

        # --- 步骤1: QKV 投影 ---
        # Q: 所有 n_heads 个头各自投影
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # K: 只有 n_kv_heads 个头投影（节省参数和计算！）
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        # V: 只有 n_kv_heads 个头投影
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # --- 步骤2: 拼接 KV 缓存 ---
        if past_kv is not None:
            past_k, past_v = past_kv
            # 在序列维度上拼接: 过去的 KV + 当前新算的 KV
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        new_kv = (K, V)  # 返回更新后的 KV，供下一步使用

        # --- 步骤3: 扩展 KV 以匹配 Q 的头数 ---
        # 关键！n_kv_heads → n_heads 的虚拟复制
        K_expanded = self._expand_kv(K)  # (B, n_heads, S, d_head)
        V_expanded = self._expand_kv(V)  # (B, n_heads, S, d_head)
        S = K_expanded.shape[2]  # 总序列长度 = past_len + T

        # --- 步骤4: 注意力计算 ---
        # Q @ K^T → 注意力分数矩阵
        scores = Q @ K_expanded.transpose(-2, -1) / math.sqrt(self.d_head)

        # 因果 mask：确保每个位置只能看到它之前的位置
        if use_causal_mask:
            # mask 形状: (T, S)，遮住右上角
            causal_mask = torch.triu(
                torch.ones(T, S, device=x.device), diagonal=S - T + 1
            ).bool()
            scores = scores.masked_fill(causal_mask, float('-inf'))

        # Softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)

        # 加权求和 V
        out = attn_weights @ V_expanded  # (B, n_heads, T, d_head)

        # --- 步骤5: 合并多头 + 输出投影 ---
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.W_o(out)

        return out, new_kv, attn_weights

    def count_parameters(self):
        """统计注意力层的参数量"""
        return sum(p.numel() for p in self.parameters())

    def kv_cache_bytes(self, seq_len, batch_size=1, dtype_bytes=2):
        """
        计算单层 KV Cache 的内存占用（字节）
        
        KV Cache 存的是 n_kv_heads 个头的 K 和 V
        """
        # 2 = K 和 V 各一份
        return 2 * batch_size * self.n_kv_heads * seq_len * self.d_head * dtype_bytes


# ============================================================
# 2. KV Cache 内存占用对比
# ============================================================

def compare_kv_cache_memory():
    """对比 MHA / GQA / MQA 三种方案的 KV Cache 内存占用"""
    print("=" * 70)
    print("📊 KV Cache 内存占用对比")
    print("=" * 70)

    # 模拟不同规模的模型配置
    configs = [
        {"name": "小模型 (7B 级)", "d_model": 4096, "n_heads": 32, "n_layers": 32},
        {"name": "中模型 (13B 级)", "d_model": 5120, "n_heads": 40, "n_layers": 40},
        {"name": "大模型 (70B 级)", "d_model": 8192, "n_heads": 64, "n_layers": 80},
    ]

    seq_len = 4096  # 序列长度
    batch_size = 1

    for cfg in configs:
        print(f"\n{'─' * 70}")
        print(f"模型: {cfg['name']} (d={cfg['d_model']}, H={cfg['n_heads']}, L={cfg['n_layers']})")
        print(f"序列长度: {seq_len}, batch_size: {batch_size}, dtype: float16")
        print(f"{'─' * 70}")

        d_model = cfg["d_model"]
        n_heads = cfg["n_heads"]
        d_head = d_model // n_heads

        # 三种方案
        schemes = [
            ("MHA", n_heads),
            ("GQA-8组", min(8, n_heads)),
            ("GQA-4组", min(4, n_heads)),
            ("MQA", 1),
        ]

        print(f"{'方案':<12} {'n_kv_heads':<12} {'单层缓存':<15} {'全模型缓存':<15} {'相对MHA':<10}")
        print(f"{'─' * 64}")

        mha_total = None
        for name, n_kv_heads in schemes:
            # 单层 KV Cache: 2 * B * n_kv_heads * T * d_head * 2 bytes
            single_layer = 2 * batch_size * n_kv_heads * seq_len * d_head * 2
            # 全模型
            total = single_layer * cfg["n_layers"]

            if mha_total is None:
                mha_total = total
                ratio_str = "1.0×"
            else:
                ratio_str = f"{total / mha_total:.4f}×"

            # 格式化大小
            def fmt_size(b):
                if b >= 1e9:
                    return f"{b / 1e9:.2f} GB"
                elif b >= 1e6:
                    return f"{b / 1e6:.2f} MB"
                else:
                    return f"{b / 1e3:.2f} KB"

            print(f"{name:<12} {n_kv_heads:<12} {fmt_size(single_layer):<15} {fmt_size(total):<15} {ratio_str:<10}")


# ============================================================
# 3. 推理速度对比（自回归生成）
# ============================================================

def compare_inference_speed():
    """实测 MHA / GQA / MQA 在自回归生成中的速度差异"""
    print("\n" + "=" * 70)
    print("⏱️ 推理速度对比（自回归生成）")
    print("=" * 70)

    torch.manual_seed(42)

    # 模拟一个小型 Transformer 注意力层
    d_model = 512
    n_heads = 32
    d_head = d_model // n_heads
    seq_len = 128  # prompt 长度
    gen_len = 50   # 生成长度

    # 三种方案
    schemes = {
        "MHA (32 KV heads)": UnifiedAttention(d_model, n_heads, n_kv_heads=32),
        "GQA (8 KV heads)": UnifiedAttention(d_model, n_heads, n_kv_heads=8),
        "MQA (1 KV head)": UnifiedAttention(d_model, n_heads, n_kv_heads=1),
    }

    x_prompt = torch.randn(1, seq_len, d_model)
    x_single = torch.randn(1, 1, d_model)  # decode 阶段，每次输入 1 个 token

    print(f"\n配置: d_model={d_model}, n_heads={n_heads}, prompt={seq_len}, 生成={gen_len}")
    print(f"{'─' * 60}")
    print(f"{'方案':<25} {'参数量':<12} {'耗时':<12} {'KV Cache 内存':<15}")
    print(f"{'─' * 60}")

    for name, attn in schemes.items():
        attn.eval()

        # 计算参数量
        n_params = attn.count_parameters()

        # 计算 KV Cache 内存
        kv_mem = attn.kv_cache_bytes(seq_len + gen_len)

        # --- Prefill 阶段：处理整个 prompt ---
        with torch.no_grad():
            _, kv_cache, _ = attn(x_prompt, past_kv=None, use_causal_mask=True)

        # --- Decode 阶段：逐个生成 ---
        start_time = time.time()
        with torch.no_grad():
            for _ in range(gen_len):
                # 每次输入 1 个 token + 上一步的 KV Cache
                _, kv_cache, _ = attn(x_single, past_kv=kv_cache, use_causal_mask=True)
        elapsed = time.time() - start_time

        def fmt_size(b):
            return f"{b / 1024:.1f} KB" if b < 1e6 else f"{b / 1e6:.2f} MB"

        print(f"{name:<25} {n_params:<12,} {elapsed:.4f}s{'':<5} {fmt_size(kv_mem):<15}")

    print(f"\n💡 观察: GQA 和 MQA 的 KV Cache 明显更小，推理速度也有提升")


# ============================================================
# 4. 注意力模式可视化
# ============================================================

def visualize_attention_patterns():
    """可视化 MHA / GQA / MQA 的注意力模式差异"""
    print("\n" + "=" * 70)
    print("👁️ 注意力模式可视化")
    print("=" * 70)

    torch.manual_seed(42)

    d_model = 64
    n_heads = 8
    seq_len = 10

    x = torch.randn(1, seq_len, d_model)

    schemes = {
        "MHA (8 KV heads)": UnifiedAttention(d_model, n_heads, n_kv_heads=8),
        "GQA (2 KV heads)": UnifiedAttention(d_model, n_heads, n_kv_heads=2),
        "MQA (1 KV head)": UnifiedAttention(d_model, n_heads, n_kv_heads=1),
    }

    for name, attn in schemes.items():
        attn.eval()
        with torch.no_grad():
            _, _, attn_weights = attn(x, use_causal_mask=True)

        print(f"\n{name}:")
        print(f"注意力权重形状: {attn_weights.shape}  (B, n_heads, T, T)")

        # 展示几个头的注意力模式（取前 4 个头）
        n_show = min(4, n_heads)
        for h in range(n_show):
            # 取最后一个 token 对所有 token 的注意力
            last_token_attn = attn_weights[0, h, -1, :].tolist()
            bar = " → ".join([f"{v:.3f}" for v in last_token_attn])
            print(f"  Head {h} (last token): [{bar}]")

    print(f"\n💡 观察:")
    print(f"  MHA: 每个 head 有不同的注意力分布（每个头独立 K/V）")
    print(f"  GQA: 同组内的 head 注意力相似但不完全相同（Q 不同）")
    print(f"  MQA: 所有 head 注意力差异来自 Q 的不同（K/V 完全共享）")


# ============================================================
# 5. Expand 操作详解
# ============================================================

def demonstrate_expand():
    """直观展示 KV expand 操作"""
    print("\n" + "=" * 70)
    print("🔧 Expand 操作详解：KV 如何从 n_kv_heads 扩展到 n_heads")
    print("=" * 70)

    # 创建一个小例子：2 个 KV head，要扩展到 8 个 Q head
    n_kv_heads = 2
    n_heads = 8
    n_rep = n_heads // n_kv_heads  # = 4
    d_head = 4
    T = 3  # 序列长度

    print(f"\n配置: n_kv_heads={n_kv_heads}, n_heads={n_heads}, n_rep={n_rep}")
    print(f"      每个 KV head 被 {n_rep} 个 Q head 共享\n")

    # 创建模拟 KV
    torch.manual_seed(0)
    K = torch.randn(1, n_kv_heads, T, d_head)
    print(f"原始 K 形状: {K.shape}  (B=1, n_kv_heads={n_kv_heads}, T={T}, d_head={d_head})")
    print(f"原始 K 内容:")

    for kv_h in range(n_kv_heads):
        print(f"  KV Head {kv_h}: {K[0, kv_h].tolist()}")

    # Step 1: 插入维度
    K_step1 = K[:, :, None, :, :]
    print(f"\nStep 1 — 插入维度: {K.shape} → {K_step1.shape}")

    # Step 2: expand
    K_step2 = K_step1.expand(1, n_kv_heads, n_rep, T, d_head)
    print(f"Step 2 — expand:   {K_step1.shape} → {K_step2.shape}")

    # Step 3: reshape
    K_final = K_step2.reshape(1, n_heads, T, d_head)
    print(f"Step 3 — reshape:  {K_step2.shape} → {K_final.shape}")

    print(f"\n扩展后 K 的内容:")
    for q_h in range(n_heads):
        kv_h = q_h // n_rep
        print(f"  Q Head {q_h} → 用 KV Head {kv_h} 的数据: {K_final[0, q_h].tolist()}")

    print(f"\n💡 关键观察:")
    print(f"  - Q Head 0~3 都用 KV Head 0 的数据（完全相同）")
    print(f"  - Q Head 4~7 都用 KV Head 1 的数据（完全相同）")
    print(f"  - 内存中实际只存了 {n_kv_heads} 份数据，不是 {n_heads} 份")
    print(f"  - 这就是 GQA 省 KV Cache 的原理！")


# ============================================================
# 6. 真实模型配置分析
# ============================================================

def analyze_real_models():
    """分析真实世界模型的 GQA 配置"""
    print("\n" + "=" * 70)
    print("🌍 真实模型的 GQA 配置分析")
    print("=" * 70)

    models = [
        {"name": "LLaMA-1 (7B)", "d": 4096, "n_heads": 32, "n_kv": 32, "layers": 32},
        {"name": "LLaMA-2 (7B)", "d": 4096, "n_heads": 32, "n_kv": 32, "layers": 32},
        {"name": "LLaMA-2 (70B)", "d": 8192, "n_heads": 64, "n_kv": 8, "layers": 80},
        {"name": "LLaMA-3 (8B)", "d": 4096, "n_heads": 32, "n_kv": 8, "layers": 32},
        {"name": "LLaMA-3 (70B)", "d": 8192, "n_heads": 64, "n_kv": 8, "layers": 80},
        {"name": "Mistral (7B)", "d": 4096, "n_heads": 32, "n_kv": 8, "layers": 32},
        {"name": "PaLM (540B)", "d": 18432, "n_heads": 48, "n_kv": 1, "layers": 118},
    ]

    seq_len = 4096
    batch_size = 1

    print(f"\n序列长度: {seq_len} | batch_size: {batch_size} | dtype: float16")
    print(f"{'─' * 85}")
    print(f"{'模型':<20} {'方案':<8} {'n_rep':<7} {'参数省':<10} {'KV Cache':<12} {'相对MHA':<10}")
    print(f"{'─' * 85}")

    for m in models:
        d = m["d"]
        nh = m["n_heads"]
        nkv = m["n_kv"]
        layers = m["layers"]
        d_head = d // nh

        # 方案名称
        if nkv == nh:
            scheme = "MHA"
        elif nkv == 1:
            scheme = "MQA"
        else:
            scheme = "GQA"

        n_rep = nh // nkv

        # 参数节省：W_k 和 W_v 的参数减少
        # MHA: 2 * d * nh * d_head = 2 * d^2
        # GQA: 2 * d * nkv * d_head
        param_ratio = 1 - (2 * d * nkv * d_head) / (2 * d * nh * d_head)

        # KV Cache
        kv_bytes = 2 * batch_size * nkv * seq_len * d_head * 2 * layers
        mha_bytes = 2 * batch_size * nh * seq_len * d_head * 2 * layers
        kv_ratio = kv_bytes / mha_bytes

        def fmt_size(b):
            if b >= 1e9:
                return f"{b / 1e9:.2f} GB"
            elif b >= 1e6:
                return f"{b / 1e6:.1f} MB"
            else:
                return f"{b / 1e3:.1f} KB"

        print(f"{m['name']:<20} {scheme:<8} {n_rep:<7} {param_ratio*100:.1f}%{'':<5} {fmt_size(kv_bytes):<12} {kv_ratio:.4f}×")

    print(f"\n💡 结论:")
    print(f"  - LLaMA-1 和 LLaMA-2 小模型仍用 MHA（模型小，KV Cache 不是瓶颈）")
    print(f"  - LLaMA-2 70B 开始用 GQA-8组，KV Cache 缩减到 1/8")
    print(f"  - LLaMA-3 全系列标配 GQA，哪怕是 8B 小模型")
    print(f"  - PaLM 用激进的 MQA，KV Cache 缩减到 1/48")


# ============================================================
# 7. GQA 的 forward 过程逐步追踪
# ============================================================

def trace_gqa_forward():
    """逐步追踪 GQA 的 forward 过程，展示每一步的形状变化"""
    print("\n" + "=" * 70)
    print("🔍 GQA Forward 逐步追踪")
    print("=" * 70)

    torch.manual_seed(42)

    d_model = 16
    n_heads = 4
    n_kv_heads = 2
    d_head = d_model // n_heads  # = 4
    T = 3

    print(f"\n配置: d_model={d_model}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, d_head={d_head}")
    print(f"      每个 KV head 被 {n_heads // n_kv_heads} 个 Q head 共享")
    print(f"      输入序列长度 T={T}\n")

    # 创建 GQA 层
    attn = UnifiedAttention(d_model, n_heads, n_kv_heads=n_kv_heads)
    attn.eval()

    x = torch.randn(1, T, d_model)
    print(f"输入 x: {x.shape}")

    with torch.no_grad():
        # 手动追踪每一步
        B = 1

        # QKV 投影
        Q = attn.W_q(x).view(B, T, n_heads, d_head).transpose(1, 2)
        K = attn.W_k(x).view(B, T, n_kv_heads, d_head).transpose(1, 2)
        V = attn.W_v(x).view(B, T, n_kv_heads, d_head).transpose(1, 2)

        print(f"\nQ 投影后: {Q.shape}  ← {n_heads} 个 Q head")
        print(f"K 投影后: {K.shape}  ← 只有 {n_kv_heads} 个 KV head！")
        print(f"V 投影后: {V.shape}  ← 只有 {n_kv_heads} 个 KV head！")

        # Expand KV
        K_exp = attn._expand_kv(K)
        V_exp = attn._expand_kv(V)

        print(f"\nK 扩展后: {K_exp.shape}  ← 从 {n_kv_heads} 扩展到 {n_heads}")
        print(f"V 扩展后: {V_exp.shape}  ← 从 {n_kv_heads} 扩展到 {n_heads}")

        # 注意力计算
        scores = Q @ K_exp.transpose(-2, -1) / math.sqrt(d_head)
        print(f"\n注意力分数: {scores.shape}  ← Q({n_heads}头) × K^T({n_heads}头)")

        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        scores_masked = scores.masked_fill(causal_mask, float('-inf'))

        attn_w = F.softmax(scores_masked, dim=-1)
        print(f"注意力权重: {attn_w.shape}")

        out = attn_w @ V_exp
        print(f"注意力输出: {out.shape}")

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        final = attn.W_o(out)
        print(f"最终输出: {final.shape}")

    print(f"\n✅ Forward 过程追踪完毕！关键步骤就是 K/V 的 expand 操作")


# ============================================================
# 主函数：运行所有实验
# ============================================================

if __name__ == "__main__":
    print("🧠 Day 14: MQA & GQA — MHA / GQA / MQA 三合一实现与对比实验")
    print("=" * 70)
    print()

    # 实验1: KV Cache 内存对比
    compare_kv_cache_memory()

    # 实验2: 推理速度对比
    compare_inference_speed()

    # 实验3: 注意力模式可视化
    visualize_attention_patterns()

    # 实验4: Expand 操作详解
    demonstrate_expand()

    # 实验5: 真实模型配置分析
    analyze_real_models()

    # 实验6: GQA Forward 逐步追踪
    trace_gqa_forward()

    print("\n" + "=" * 70)
    print("✅ 所有实验完成！")
    print("=" * 70)
    print()
    print("🔑 核心要点回顾:")
    print("  1. GQA 通过 n_kv_heads 控制 K/V 的共享程度")
    print("  2. MHA (n_kv=n_heads) → GQA (1<n_kv<n_heads) → MQA (n_kv=1)")
    print("  3. KV Cache 缩减 n_heads/n_kv_heads 倍")
    print("  4. expand 操作零拷贝地将 KV '复制'给所有 Q head")
    print("  5. 现代 LLM 几乎全部采用 GQA (LLaMA-2/3, Mistral, Gemini)")
    print()
    print("下一节预告: Day 15 — Flash Attention，用'聪明地算'代替'算得快'")
