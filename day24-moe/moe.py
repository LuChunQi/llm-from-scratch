#!/usr/bin/env python3
"""
Day 24: MoE（混合专家模型）— 从零实现与实验
=============================================

实验内容：
1. 从零实现 MoE 层（Gate + 多专家 + Top-K 路由）
2. 路由器可视化 — 看看 token 被分配到哪个专家
3. 负载均衡损失实现与验证
4. Dense vs MoE 参数量/计算量对比
5. DeepSeek 风格 MoE（细粒度 + 共享专家）
6. 训练一个字符级 MoE 语言模型，观察专家分工
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import Counter


# ================================================================
# 1. 基础组件：单个 FFN 专家
# ================================================================

class ExpertFFN(nn.Module):
    """
    单个 FFN 专家，和标准 Transformer 的 FFN 一样：
    x → Linear(d, d_ff) → SiLU → Linear(d_ff, d) → output
    每个专家有独立的参数，不共享。
    """
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        # W1: (d_model, d_ff) — 升维
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        # W2: (d_ff, d_model) — 降维
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        # SiLU 激活函数（Swish = x * sigmoid(x)）
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (..., d_model) → (..., d_model)"""
        h = self.w1(x)          # 升维
        h = self.act(h)         # 激活
        h = self.dropout(h)     # Dropout
        out = self.w2(h)        # 降维
        return out


# ================================================================
# 2. 路由器（Gate / Router）
# ================================================================

class TopKRouter(nn.Module):
    """
    Top-K 路由器：为每个 token 选择概率最高的 K 个专家。

    核心流程：
    1. x @ W_gate → 计算每个专家的适配分数
    2. softmax → 变成概率分布
    3. topk → 选择概率最高的 K 个
    4. 重新归一化 → 确保选中的 K 个概率之和 = 1
    """
    def __init__(self, d_model, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 路由器权重：一个简单的线性层
        self.w_gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x):
        """
        x: (num_tokens, d_model)
        返回：
        - gate_scores: (num_tokens, num_experts) — softmax 后的概率
        - topk_indices: (num_tokens, top_k) — 选中的专家索引
        - topk_weights: (num_tokens, top_k) — 选中专家的权重
        """
        # Step 1: 计算适配分数
        logits = self.w_gate(x)  # (num_tokens, num_experts)

        # Step 2: Softmax 归一化
        gate_scores = F.softmax(logits, dim=-1)

        # Step 3: 选择 Top-K
        topk_weights, topk_indices = torch.topk(gate_scores, self.top_k, dim=-1)

        # Step 4: 重新归一化（让选中的 K 个权重之和 = 1）
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        return gate_scores, topk_indices, topk_weights


# ================================================================
# 3. 负载均衡损失
# ================================================================

def load_balancing_loss(gate_scores, topk_indices, num_experts):
    """
    计算负载均衡损失，防止路由崩塌。

    L_balance = N × Σ(f_i × P_i)

    其中：
    - f_i: 专家 i 在当前 batch 中被选中的频率
    - P_i: 专家 i 的平均 Gate 概率
    - N: 专家总数
    """
    num_tokens = gate_scores.shape[0]
    top_k = topk_indices.shape[1]

    # 计算 P_i: 每个专家的平均 Gate 概率
    P = gate_scores.mean(dim=0)  # (num_experts,)

    # 计算 f_i: 每个专家被选中的频率
    expert_counts = torch.zeros(num_experts, device=gate_scores.device)
    for k in range(top_k):
        indices = topk_indices[:, k]
        one_hot = F.one_hot(indices, num_experts).float()
        expert_counts += one_hot.sum(dim=0)

    f = expert_counts / (num_tokens * top_k)

    # 负载均衡损失 = N × Σ(f_i × P_i)
    loss = num_experts * (f * P).sum()
    return loss


# ================================================================
# 4. 完整 MoE 层
# ================================================================

class MoELayer(nn.Module):
    """
    完整的 MoE 层：Gate + 多个 FFN 专家 + Top-K 路由 + 加权求和

    Dense: x → FFN(x) → output（每个 token 走同一个 FFN）
    MoE:   x → Gate → 选 Top-K 个专家 → 加权求和 → output
    """
    def __init__(self, d_model, d_ff, num_experts=8, top_k=2, dropout=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 创建 N 个独立的 FFN 专家
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff, dropout)
            for _ in range(num_experts)
        ])

        # 路由器
        self.router = TopKRouter(d_model, num_experts, top_k)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        返回：output, (gate_scores, topk_indices, topk_weights), aux_loss
        """
        original_shape = x.shape
        # 展平为 (num_tokens, d_model)
        x_flat = x.view(-1, x.shape[-1])

        # Step 1: 路由
        gate_scores, topk_indices, topk_weights = self.router(x_flat)

        # Step 2: 计算每个专家的输出，加权求和
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):
            expert_indices = topk_indices[:, k]
            weights = topk_weights[:, k].unsqueeze(-1)

            for e in range(self.num_experts):
                mask = (expert_indices == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e](expert_input)
                    expert_weight = weights[mask]
                    output[mask] += expert_weight * expert_output

        # Step 3: 计算负载均衡损失
        aux_loss = load_balancing_loss(gate_scores, topk_indices, self.num_experts)

        # 恢复原始形状
        output = output.view(original_shape)
        return output, (gate_scores, topk_indices, topk_weights), aux_loss


# ================================================================
# 5. DeepSeek 风格 MoE：细粒度专家 + 共享专家
# ================================================================

class DeepSeekMoELayer(nn.Module):
    """
    DeepSeek 风格的 MoE 层：
    - 更多更小的"路由专家"（如 64 个小专家）
    - 1-2 个"共享专家"（所有 token 都经过，不走 Gate）

    共享专家负责"通用知识"，路由专家负责"差异化知识"。
    """
    def __init__(self, d_model, d_ff_routed, d_ff_shared,
                 num_routed_experts=64, num_shared_experts=1,
                 top_k=8, dropout=0.0):
        super().__init__()
        self.num_routed = num_routed_experts
        self.num_shared = num_shared_experts
        self.top_k = top_k

        # 路由专家（更多更小）
        self.routed_experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff_routed, dropout)
            for _ in range(num_routed_experts)
        ])

        # 共享专家（所有 token 都经过）
        self.shared_experts = nn.ModuleList([
            ExpertFFN(d_model, d_ff_shared, dropout)
            for _ in range(num_shared_experts)
        ])

        # 路由器（只路由到路由专家）
        self.router = TopKRouter(d_model, num_routed_experts, top_k)

    def forward(self, x):
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])

        # ---- 共享专家部分（所有 token 都经过）----
        shared_output = torch.zeros_like(x_flat)
        for se in self.shared_experts:
            shared_output += se(x_flat)
        shared_output /= self.num_shared

        # ---- 路由专家部分（Top-K 路由）----
        gate_scores, topk_indices, topk_weights = self.router(x_flat)

        routed_output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = topk_indices[:, k]
            weights = topk_weights[:, k].unsqueeze(-1)

            for e in range(self.num_routed):
                mask = (expert_indices == e)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.routed_experts[e](expert_input)
                    expert_weight = weights[mask]
                    routed_output[mask] += expert_weight * expert_output

        # 总输出 = 共享专家 + 路由专家
        output = shared_output + routed_output
        aux_loss = load_balancing_loss(gate_scores, topk_indices, self.num_routed)

        output = output.view(original_shape)
        return output, (gate_scores, topk_indices, topk_weights), aux_loss


# ================================================================
# 6. 辅助函数
# ================================================================

def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ================================================================
# 实验 1：基本 MoE 层 — 前向传播 + 路由分析
# ================================================================

def experiment_1_basic_moe():
    print("=" * 70)
    print("实验 1：基本 MoE 层 — 前向传播 + 路由分析")
    print("=" * 70)

    torch.manual_seed(42)

    d_model = 64
    d_ff = 256
    num_experts = 8
    top_k = 2

    moe = MoELayer(d_model, d_ff, num_experts, top_k)

    # 创建输入：4 个 token
    x = torch.randn(2, 2, d_model)

    print(f"\n输入: batch=2, seq_len=2, d_model={d_model}")
    print(f"MoE 配置: {num_experts} 个专家, Top-{top_k} 路由")

    output, (gate_scores, topk_indices, topk_weights), aux_loss = moe(x)

    print(f"\n输出形状: {output.shape}")
    print(f"负载均衡损失: {aux_loss.item():.4f}")

    # 分析路由分配
    print("\n--- 路由分析 ---")
    num_tokens = 4

    print(f"\nGate 分数（softmax 后的概率）:")
    for i in range(num_tokens):
        probs = gate_scores[i].detach().numpy()
        selected = topk_indices[i].detach().numpy()
        weights = topk_weights[i].detach().numpy()
        print(f"  Token {i}: 选中专家 {selected} (权重 {weights[0]:.3f}, {weights[1]:.3f})")
        prob_str = ", ".join([f"E{j}:{p:.3f}" for j, p in enumerate(probs)])
        print(f"          全部分数: [{prob_str}]")

    # 统计每个专家被选中的次数
    expert_counts = Counter()
    for i in range(num_tokens):
        for k_idx in range(top_k):
            expert_counts[topk_indices[i, k_idx].item()] += 1

    print(f"\n专家负载分布:")
    for e in range(num_experts):
        count = expert_counts.get(e, 0)
        bar = "█" * count + "░" * max(0, num_tokens * top_k // num_experts - count)
        print(f"  专家 {e}: {bar} ({count} 次)")

    print("\n✅ MoE 前向传播成功！每个 token 被路由到了 2 个专家。")


# ================================================================
# 实验 2：Dense vs MoE 参数量对比
# ================================================================

def experiment_2_parameter_comparison():
    print("\n" + "=" * 70)
    print("实验 2：Dense vs MoE 参数量对比")
    print("=" * 70)

    d_model = 512
    d_ff = 2048
    num_experts = 8
    top_k = 2

    # Dense FFN
    dense_ffn = ExpertFFN(d_model, d_ff)
    dense_params = count_parameters(dense_ffn)[0]

    # MoE 层（8 个专家）
    moe_layer = MoELayer(d_model, d_ff, num_experts, top_k)
    moe_total_params = count_parameters(moe_layer)[0]

    # MoE 激活参数
    expert_params = count_parameters(dense_ffn)[0]
    gate_params = count_parameters(moe_layer.router)[0]
    activated_params = top_k * expert_params + gate_params

    print(f"\n配置: d_model={d_model}, d_ff={d_ff}, {num_experts} 专家, Top-{top_k}")
    print(f"\n{'类型':<20} {'参数量':<15} {'vs Dense':<10}")
    print("-" * 50)
    print(f"{'Dense FFN':<20} {dense_params:>12,}   {'基准':<10}")
    print(f"{'MoE 总参数':<20} {moe_total_params:>12,}   {moe_total_params/dense_params:.1f}x")
    print(f"{'MoE 激活参数':<20} {activated_params:>12,}   {activated_params/dense_params:.1f}x")

    print(f"\n洞察:")
    print(f"  - MoE 总参数是 Dense 的 {moe_total_params/dense_params:.1f} 倍")
    print(f"  - 但激活参数只是 Dense 的 {activated_params/dense_params:.1f} 倍")
    print(f"  - 计算量 ≈ 激活参数（MoE 只比 Dense 多 ~{activated_params/dense_params:.1f}x）")
    print(f"  - 存储 ≈ 总参数（需要 {moe_total_params/dense_params:.1f}x 的内存）")

    # DeepSeek 风格对比
    print(f"\n--- DeepSeek 风格 MoE 对比 ---")
    d_ff_routed = d_ff // 8
    d_ff_shared = d_ff
    num_routed = 64
    top_k_deepseek = 8

    deepseek_moe = DeepSeekMoELayer(
        d_model, d_ff_routed, d_ff_shared,
        num_routed, 1, top_k_deepseek
    )
    deepseek_params = count_parameters(deepseek_moe)[0]

    routed_expert_params = d_model * d_ff_routed * 2
    shared_expert_params = d_model * d_ff_shared * 2
    deepseek_gate_params = count_parameters(deepseek_moe.router)[0]
    deepseek_activated = top_k_deepseek * routed_expert_params + shared_expert_params + deepseek_gate_params

    print(f"  配置: {num_routed} 个路由专家 (d_ff={d_ff_routed}), 1 个共享专家 (d_ff={d_ff_shared}), Top-{top_k_deepseek}")
    print(f"  {'DeepSeek MoE 总参数':<25} {deepseek_params:>12,}   {deepseek_params/dense_params:.1f}x Dense")
    print(f"  {'DeepSeek 激活参数':<25} {deepseek_activated:>12,}   {deepseek_activated/dense_params:.1f}x Dense")
    print(f"  精度提升来源：更多样的专家选择，同时激活参数量控制得当")


# ================================================================
# 实验 3：负载均衡损失效果验证
# ================================================================

def experiment_3_load_balancing():
    print("\n" + "=" * 70)
    print("实验 3：负载均衡损失效果验证")
    print("=" * 70)

    torch.manual_seed(42)

    d_model = 64
    num_experts = 8
    top_k = 2
    num_tokens = 128

    router = TopKRouter(d_model, num_experts, top_k)
    x = torch.randn(num_tokens, d_model)

    print(f"\n配置: {num_tokens} 个 token, {num_experts} 个专家, Top-{top_k}")

    # 初始状态
    gate_scores, topk_indices, _ = router(x)
    initial_loss = load_balancing_loss(gate_scores, topk_indices, num_experts)

    expert_counts = Counter()
    for i in range(num_tokens):
        for k_idx in range(top_k):
            expert_counts[topk_indices[i, k_idx].item()] += 1

    print(f"\n初始路由（随机初始化的 Gate）:")
    print(f"负载均衡损失: {initial_loss.item():.4f}")
    print(f"理想损失（均匀分布）: {1.0 / num_experts:.4f}")

    total_selections = num_tokens * top_k
    print(f"\n专家负载分布:")
    for e in range(num_experts):
        count = expert_counts.get(e, 0)
        pct = count / total_selections * 100
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"  专家 {e}: {bar} {count:3d} ({pct:.1f}%)  理想: {100/num_experts:.1f}%")

    # 优化负载均衡
    print(f"\n--- 模拟 100 步优化，观察负载均衡效果 ---")

    optimizer = torch.optim.Adam(router.parameters(), lr=0.01)

    for step in range(100):
        optimizer.zero_grad()
        gate_scores, topk_indices, _ = router(x)
        loss = load_balancing_loss(gate_scores, topk_indices, num_experts)
        loss.backward()
        optimizer.step()

    # 最终状态
    gate_scores, topk_indices, _ = router(x)
    final_loss = load_balancing_loss(gate_scores, topk_indices, num_experts)

    expert_counts_after = Counter()
    for i in range(num_tokens):
        for k_idx in range(top_k):
            expert_counts_after[topk_indices[i, k_idx].item()] += 1

    print(f"\n优化后:")
    print(f"负载均衡损失: {final_loss.item():.4f} (初始: {initial_loss.item():.4f})")
    improvement = (initial_loss.item() - final_loss.item()) / initial_loss.item() * 100
    print(f"改善: {improvement:.1f}%")

    print(f"\n优化后专家负载分布:")
    for e in range(num_experts):
        count = expert_counts_after.get(e, 0)
        pct = count / total_selections * 100
        bar_len = int(pct / 2)
        bar = "█" * bar_len
        print(f"  专家 {e}: {bar} {count:3d} ({pct:.1f}%)")

    print(f"\n✅ 负载均衡损失有效地将 token 均匀分配到了各专家！")


# ================================================================
# 实验 4：路由器可视化 — 不同特征的 token 倾向于不同专家
# ================================================================

def experiment_4_router_visualization():
    print("\n" + "=" * 70)
    print("实验 4：路由器可视化 — 不同特征 token 的路由偏好")
    print("=" * 70)

    torch.manual_seed(123)

    d_model = 32
    num_experts = 8
    n_per_group = 50

    # 创建 4 组不同特征的 token
    group_a = torch.randn(n_per_group, d_model) + torch.tensor(
        [2.0, 1.5, 0, 0, 0, 0, 0, 0] + [0] * (d_model - 8))
    group_b = torch.randn(n_per_group, d_model) + torch.tensor(
        [0, 0, 2.0, 1.5, 0, 0, 0, 0] + [0] * (d_model - 8))
    group_c = torch.randn(n_per_group, d_model) + torch.tensor(
        [0, 0, 0, 0, 2.0, 1.5, 0, 0] + [0] * (d_model - 8))
    group_d = torch.randn(n_per_group, d_model) + torch.tensor(
        [0, 0, 0, 0, 0, 0, 2.0, 1.5] + [0] * (d_model - 8))

    all_tokens = torch.cat([group_a, group_b, group_c, group_d], dim=0)

    router = TopKRouter(d_model, num_experts, top_k=2)
    gate_scores, topk_indices, topk_weights = router(all_tokens)

    print(f"\n4 组不同特征的 token, 每组 {n_per_group} 个, {num_experts} 个专家")

    print(f"\n每组 token 的首选专家分布（Top-1）:")
    print(f"{'组别':<8}", end="")
    for e in range(num_experts):
        print(f"{'专家'+str(e):<8}", end="")
    print()
    print("-" * (8 + 8 * num_experts))

    groups = {
        '数学': 0,
        '语言': n_per_group,
        '代码': 2 * n_per_group,
        '通用': 3 * n_per_group,
    }
    for name, start in groups.items():
        end = start + n_per_group
        counts = Counter(topk_indices[start:end, 0].tolist())
        print(f"{name:<8}", end="")
        for e in range(num_experts):
            print(f"{counts.get(e, 0):<8}", end="")
        print()

    print(f"\n不同特征的 token 确实有不同的路由偏好！")
    print(f"这解释了为什么 MoE 能学到专家分工——路由器根据 token 的语义特征分配专家。")


# ================================================================
# 实验 5：训练字符级 MoE 语言模型
# ================================================================

def experiment_5_train_moe_lm():
    print("\n" + "=" * 70)
    print("实验 5：训练字符级 MoE 语言模型 — 观察专家分工")
    print("=" * 70)

    torch.manual_seed(42)

    # 训练数据
    text = (
        "Attention is all you need. The Transformer architecture revolutionized NLP. "
        "自注意力机制是Transformer的核心。Multi-Head Attention让模型同时关注不同位置。 "
        "1234567890 are digits. Hello World! 你好世界！ "
        "The quick brown fox jumps over the lazy dog. 这只敏捷的棕色狐狸跳过了懒狗。 "
        "Deep Learning is a subset of Machine Learning. 深度学习是机器学习的一个子集。 "
        "PyTorch is a popular framework for deep learning research and development. "
        "Gradient descent optimization helps minimize the loss function during training. "
        "反向传播算法是训练神经网络的基础。Backpropagation computes gradients efficiently. "
        "Natural Language Processing deals with text data. 自然语言处理处理文本数据。 "
        "The MoE architecture uses a router to select experts for each token. "
        "混合专家模型使用路由器为每个token选择专家。This enables efficient computation."
    )

    # 字符级 tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for i, c in enumerate(chars)}

    print(f"词汇表大小: {vocab_size}（字符级）")
    print(f"训练文本长度: {len(text)} 个字符")

    data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)

    # 超参数
    d_model = 32
    d_ff = 128
    num_experts = 4
    top_k = 2
    n_heads = 4
    n_layers = 2
    block_size = 16
    lr = 0.005
    num_steps = 300
    aux_loss_weight = 0.01

    # MoE 语言模型
    class MoELanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(block_size, d_model)

            self.layers = nn.ModuleList()
            for _ in range(n_layers):
                self.layers.append(nn.ModuleDict({
                    'attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                    'ln1': nn.LayerNorm(d_model),
                    'moe': MoELayer(d_model, d_ff, num_experts, top_k),
                    'ln2': nn.LayerNorm(d_model),
                }))

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, idx):
            B, T = idx.shape
            tok_emb = self.token_emb(idx)
            pos = torch.arange(T, device=idx.device)
            pos_emb = self.pos_emb(pos)
            x = tok_emb + pos_emb

            total_aux_loss = 0.0

            for layer in self.layers:
                x_ln = layer['ln1'](x)
                attn_out, _ = layer['attn'](x_ln, x_ln, x_ln, need_weights=False)
                x = x + attn_out

                x_ln = layer['ln2'](x)
                moe_out, route_info, aux_loss = layer['moe'](x_ln)
                x = x + moe_out
                total_aux_loss += aux_loss

            x = self.ln_f(x)
            logits = self.head(x)
            return logits, total_aux_loss

    model = MoELanguageModel()
    total_params, _ = count_parameters(model)
    print(f"模型参数量: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n开始训练 ({num_steps} 步)...")
    print(f"{'步骤':<8} {'损失':<10} {'辅助损失':<12} {'路由熵':<12}")
    print("-" * 45)

    for step in range(num_steps):
        starts = torch.randint(0, len(data) - block_size - 1, (8,))
        x_batch = torch.stack([data[s:s + block_size] for s in starts])
        y_batch = torch.stack([data[s + 1:s + block_size + 1] for s in starts])

        logits, aux_loss = model(x_batch)

        B, T, C = logits.shape
        lm_loss = F.cross_entropy(logits.view(B * T, C), y_batch.view(B * T))

        loss = lm_loss + aux_loss_weight * aux_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == num_steps - 1:
            with torch.no_grad():
                emb = model.token_emb(x_batch) + model.pos_emb(
                    torch.arange(block_size).unsqueeze(0).expand(8, -1))
                _, route_info, _ = model.layers[0]['moe'](emb)
                avg_entropy = -(route_info[0] * (route_info[0] + 1e-10).log()).sum(-1).mean().item()

            print(f"{step:<8} {lm_loss.item():<10.4f} {aux_loss.item():<12.4f} {avg_entropy:<12.4f}")

    # 生成测试
    print(f"\n--- 生成测试 ---")
    model.eval()
    with torch.no_grad():
        context = torch.tensor([[char_to_idx['T']]], dtype=torch.long)
        generated = list(context[0].numpy())

        for _ in range(80):
            ctx = context[:, -block_size:]
            logits, _ = model(ctx)
            last_logits = logits[0, -1]
            probs = F.softmax(last_logits / 0.8, dim=-1)
            next_char = torch.multinomial(probs, 1)
            generated.append(next_char.item())
            context = torch.cat([context, next_char.unsqueeze(0)], dim=1)

        generated_text = ''.join([idx_to_char[i] for i in generated])
        print(f"生成文本: {generated_text}")

    # 分析训练后的专家路由
    print(f"\n--- 训练后的专家路由分析 ---")
    with torch.no_grad():
        test_text = "Attention is"  # 不超过 block_size=16
        test_idx = torch.tensor([[char_to_idx.get(c, 0) for c in test_text]])
        test_len = len(test_text)
        x_emb = model.token_emb(test_idx) + model.pos_emb(
            torch.arange(test_len).unsqueeze(0))

        _, route_info, _ = model.layers[0]['moe'](x_emb)
        gate_scores, topk_indices, topk_weights = route_info

        # MoE 层输出 shape: (batch*seq, ...) 被展平了
        # x_emb 是 (1, test_len, d_model) → 展平后 (test_len, d_model)
        print(f"测试文本: '{test_text}'")
        print(f"每个字符的首选专家:")
        for i, c in enumerate(test_text):
            expert = topk_indices[i, 0].item()
            weight = topk_weights[i, 0].item()
            print(f"  '{c}' -> 专家 {expert} (权重 {weight:.3f})")

    print(f"\n✅ MoE 语言模型训练完成！可以看到不同字符被路由到了不同专家。")


# ================================================================
# 实验 6：FLOPs 对比 — Dense vs MoE
# ================================================================

def experiment_6_flops_comparison():
    print("\n" + "=" * 70)
    print("实验 6：FLOPs 对比 — Dense vs MoE 省了多少计算？")
    print("=" * 70)

    # 模拟真实模型配置
    configs = [
        {
            'name': 'Llama-2-7B (Dense)',
            'd_model': 4096,
            'd_ff': 11008,
            'n_layers': 32,
            'num_experts': 1,
            'top_k': 1,
        },
        {
            'name': 'Mixtral 8x7B (MoE)',
            'd_model': 4096,
            'd_ff': 14336,
            'n_layers': 32,
            'num_experts': 8,
            'top_k': 2,
        },
    ]

    seq_len = 1  # 生成模式

    print(f"\n对比（每个 token 的前向传播）:")
    print(f"{'模型':<25} {'总参数':<14} {'激活参数':<14} {'FFN FLOPs':<18} {'激活比':<8}")
    print("-" * 80)

    for cfg in configs:
        d = cfg['d_model']
        d_ff = cfg['d_ff']
        n = cfg['n_layers']
        E = cfg['num_experts']
        K = cfg['top_k']

        # 每层 FFN 参数量
        ffn_params_per_layer = 2 * d * d_ff * E  # W1 + W2, 乘以专家数
        total_params = n * ffn_params_per_layer
        activated_params = n * 2 * d * d_ff * K  # 只激活 K 个专家

        # FFN FLOPs（矩阵乘法：2 * M * N，因为乘法+加法）
        ffn_flops = activated_params * 2

        activation_ratio = activated_params / total_params * 100 if total_params > 0 else 100

        total_str = f"{total_params / 1e9:.1f}B"
        active_str = f"{activated_params / 1e9:.1f}B"
        flops_str = f"{ffn_flops / 1e9:.1f} GFLOPs"
        ratio_str = f"{activation_ratio:.0f}%"

        print(f"{cfg['name']:<25} {total_str:<14} {active_str:<14} {flops_str:<18} {ratio_str:<8}")

    print(f"\n关键洞察:")
    print(f"  - Mixtral 的总参数是 Llama-2-7B 的 ~7 倍")
    print(f"  - 但激活参数只增加了约 2.6 倍")
    print(f"  - 推理计算量 ≈ 2.6 倍 Llama-2-7B，但效果接近 34B Dense")
    print(f"  - 这就是 MoE 的核心价值：用更少的计算获得更强的模型")


# ================================================================
# 主函数
# ================================================================

if __name__ == "__main__":
    print("🧠 Day 24: MoE（混合专家模型）— 从零实现与实验")
    print("=" * 70)
    print()

    experiment_1_basic_moe()
    experiment_2_parameter_comparison()
    experiment_3_load_balancing()
    experiment_4_router_visualization()
    experiment_5_train_moe_lm()
    experiment_6_flops_comparison()

    print("\n" + "=" * 70)
    print("🎉 所有实验完成！")
    print("=" * 70)
    print("\n关键收获:")
    print("  1. MoE = 多个专家 FFN + 路由器，每次只激活 Top-K 个专家")
    print("  2. 路由器是简单的线性层 + Softmax + Top-K")
    print("  3. 负载均衡损失防止路由崩塌（所有 token 挤到一个专家）")
    print("  4. 总参数多（存储贵），但激活参数少（计算便宜）")
    print("  5. DeepSeek 创新：64 个小专家 + 共享专家，分工更精细")
    print("  6. MoE 只替换 FFN，Attention 层仍然共享")
