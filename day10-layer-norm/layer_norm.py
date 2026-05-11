"""
Day 10: Layer Normalization + 残差连接 — Transformer 的"健康通道"
============================================================

实验内容：
1. 手写 LayerNorm（不依赖框架）并和 PyTorch 官方实现对比
2. 手写 RMSNorm 并和 LayerNorm 对比
3. 梯度消失实验：对比有/无 LayerNorm + 残差连接时梯度的健康程度
4. Pre-Norm vs Post-Norm 训练对比
5. 完整的 Transformer Block 组装

运行方式：python3 layer_norm.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================
# 1. 手写 LayerNorm — 从零实现，理解每一步
# ========================================================

class MyLayerNorm(nn.Module):
    """
    手写 Layer Normalization
    
    原理：对单个向量 x 的所有维度做标准化
    x_hat = (x - mean) / sqrt(var + eps)
    output = gamma * x_hat + beta
    
    gamma 和 beta 是可学习参数，初始值为 1 和 0
    eps 是防止除以 0 的小常数
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps                                  # 防止除零的小常数
        self.gamma = nn.Parameter(torch.ones(d_model))  # 可学习缩放参数，初始=1
        self.beta = nn.Parameter(torch.zeros(d_model))  # 可学习偏移参数，初始=0

    def forward(self, x):
        # x 的形状: [batch_size, seq_len, d_model]
        # 沿最后一个维度（d_model）计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)         # 均值，形状 [batch, seq, 1]
        var = x.var(dim=-1, unbiased=False, keepdim=True)  # 方差，形状 [batch, seq, 1]

        # 标准化：减均值，除以标准差
        x_hat = (x - mean) / torch.sqrt(var + self.eps)

        # 缩放和平移
        return self.gamma * x_hat + self.beta


class MyRMSNorm(nn.Module):
    """
    手写 RMSNorm（Root Mean Square Normalization）
    
    简化版 LayerNorm：
    - 不减均值（假设均值接近 0）
    - 不加 beta（只保留缩放）
    - 只除以 RMS 值
    
    output = gamma * x / sqrt(mean(x^2) + eps)
    
    被 LLaMA、Qwen、DeepSeek 等现代 LLM 采用
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))  # 只需要一个可学习参数

    def forward(self, x):
        # 计算 RMS（均方根）
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        # 缩放（不减均值、不加偏移）
        return self.gamma * (x / rms)


def test_layernorm_correctness():
    """验证手写 LayerNorm 和 PyTorch 官方实现一致"""
    print("=" * 60)
    print("实验 1: 手写 LayerNorm vs PyTorch 官方实现")
    print("=" * 60)
    
    torch.manual_seed(42)
    d_model = 64
    batch_size = 4
    seq_len = 10
    
    # 创建测试输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 手写版本
    my_ln = MyLayerNorm(d_model)
    # PyTorch 官方版本（用同样的 gamma 和 beta）
    pt_ln = nn.LayerNorm(d_model)
    pt_ln.weight.data = my_ln.gamma.data.clone()
    pt_ln.bias.data = my_ln.beta.data.clone()
    
    my_output = my_ln(x)
    pt_output = pt_ln(x)
    
    # 计算差异
    diff = (my_output - pt_output).abs().max().item()
    
    print(f"  输入形状: {x.shape}")
    print(f"  输入均值: {x.mean():.4f}, 标准差: {x.std():.4f}")
    print(f"  输出均值: {my_output.mean():.6f} (≈0)")
    print(f"  输出标准差: {my_output.std():.6f} (≈1)")
    print(f"  与 PyTorch 官方实现最大差异: {diff:.2e}")
    print(f"  ✅ 验证通过！" if diff < 1e-5 else "  ❌ 差异过大！")
    print()


def test_rmsnorm():
    """测试 RMSNorm 并和 LayerNorm 对比"""
    print("=" * 60)
    print("实验 2: RMSNorm vs LayerNorm 对比")
    print("=" * 60)
    
    torch.manual_seed(42)
    d_model = 64
    x = torch.randn(2, 5, d_model) * 3 + 1  # 均值偏移、大方差
    
    my_ln = MyLayerNorm(d_model)
    my_rms = MyRMSNorm(d_model)
    
    ln_out = my_ln(x)
    rms_out = my_rms(x)
    
    print(f"  输入 - 均值: {x.mean():.4f}, 标准差: {x.std():.4f}")
    print(f"  LayerNorm 输出 - 均值: {ln_out.mean():.6f}, 标准差: {ln_out.std():.6f}")
    print(f"  RMSNorm  输出 - 均值: {rms_out.mean():.6f}, 标准差: {rms_out.std():.6f}")
    print(f"  RMSNorm 参数量: {sum(p.numel() for p in my_rms.parameters())} (只有 gamma)")
    print(f"  LayerNorm 参数量: {sum(p.numel() for p in my_ln.parameters())} (gamma + beta)")
    print()
    print("  💡 RMSNorm 省掉了均值计算和 beta 参数，但输出分布也略有不同")
    print("  💡 实际使用中效果几乎一样，现代 LLM（LLaMA等）都选 RMSNorm")
    print()


# ========================================================
# 3. 梯度消失实验
# ========================================================

class DeepLinearNetwork(nn.Module):
    """深层线性网络（无归一化、无残差）"""
    def __init__(self, depth, d_model):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(d_model, d_model))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DeepLinearWithResidual(nn.Module):
    """深层线性网络（带残差连接，但无归一化）"""
    def __init__(self, depth, d_model):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(d_model, d_model))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x) + x   # 残差连接：输出 = 变换 + 原始输入
        return x


class DeepLinearWithLNAndResidual(nn.Module):
    """深层网络（带 LayerNorm + 残差连接 — Pre-Norm）"""
    def __init__(self, depth, d_model):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.Linear(d_model, d_model))
            self.norms.append(nn.LayerNorm(d_model))
    
    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = layer(norm(x)) + x   # Pre-Norm: 先归一化，再变换，再加残差
        return x


def gradient_health_check():
    """对比三种网络的梯度健康度"""
    print("=" * 60)
    print("实验 3: 梯度消失实验 — 50 层深层网络")
    print("=" * 60)
    
    torch.manual_seed(42)
    depth = 50
    d_model = 32
    batch_size = 4
    
    x = torch.randn(batch_size, d_model)
    target = torch.randn(batch_size, d_model)
    
    models = {
        "无归一化 无残差": DeepLinearNetwork(depth, d_model),
        "有残差 无归一化": DeepLinearWithResidual(depth, d_model),
        "LayerNorm + 残差 (Pre-Norm)": DeepLinearWithLNAndResidual(depth, d_model),
    }
    
    for name, model in models.items():
        # 前向传播
        output = model(x)
        loss = F.mse_loss(output, target)
        
        # 反向传播
        loss.backward()
        
        # 检查每层梯度的范数
        grad_norms = []
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'weight') and layer.weight.grad is not None:
                grad_norm = layer.weight.grad.norm().item()
                grad_norms.append((i, grad_norm))
        
        # 统计梯度健康度
        if grad_norms:
            first_grad = grad_norms[0][1]
            last_grad = grad_norms[-1][1]
            min_grad = min(g for _, g in grad_norms)
            max_grad = max(g for _, g in grad_norms)
            ratio = last_grad / (first_grad + 1e-10)
            
            print(f"\n  📊 {name}:")
            print(f"     第 1 层梯度范数:  {first_grad:.6f}")
            print(f"     第 50 层梯度范数: {last_grad:.6f}")
            print(f"     最小梯度范数:     {min_grad:.6f}")
            print(f"     最大梯度范数:     {max_grad:.6f}")
            print(f"     梯度衰减比(底/顶): {ratio:.6f}")
            
            if ratio < 1e-5:
                print(f"     ❌ 严重梯度消失！底层几乎不更新")
            elif ratio < 0.01:
                print(f"     ⚠️ 轻度梯度消失")
            else:
                print(f"     ✅ 梯度流动健康！")
        
        # 清零梯度
        model.zero_grad()
    
    print()
    print("  💡 结论：没有归一化和残差 → 梯度消失严重")
    print("  💡 只有残差 → 好一些，但仍然不稳定")
    print("  💡 LayerNorm + 残差 → 梯度传递健康，底层参数也能正常更新")
    print()


# ========================================================
# 4. Pre-Norm vs Post-Norm 训练对比
# ========================================================

class PreNormBlock(nn.Module):
    """Pre-Norm Transformer Block（LayerNorm 在子层之前）"""
    def __init__(self, d_model):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.Linear(d_model, d_model)  # 简化的"注意力"
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
    
    def forward(self, x):
        # Pre-Norm: 先归一化，再变换，再加残差
        x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        return x


class PostNormBlock(nn.Module):
    """Post-Norm Transformer Block（LayerNorm 在残差之后）"""
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # Post-Norm: 先变换，再加残差，最后归一化
        x = self.norm1(self.attn(x) + x)
        x = self.norm2(self.ffn(x) + x)
        return x


def train_compare_prenorm_postnorm():
    """训练对比 Pre-Norm 和 Post-Norm"""
    print("=" * 60)
    print("实验 4: Pre-Norm vs Post-Norm 训练对比（12 层网络）")
    print("=" * 60)
    
    torch.manual_seed(42)
    d_model = 32
    depth = 12
    lr = 1e-3
    steps = 200
    
    # 生成训练数据：简单的 y = sin(x)
    x_train = torch.randn(64, d_model)
    y_train = torch.sin(x_train)  # 目标是输入的正弦
    
    results = {}
    
    for name, BlockClass in [("Pre-Norm", PreNormBlock), ("Post-Norm", PostNormBlock)]:
        torch.manual_seed(42)  # 每次用相同的种子
        
        # 构建深层网络
        blocks = nn.Sequential(*[BlockClass(d_model) for _ in range(depth)])
        
        optimizer = torch.optim.Adam(blocks.parameters(), lr=lr)
        losses = []
        
        for step in range(steps):
            output = blocks(x_train)
            loss = F.mse_loss(output, y_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        results[name] = losses
        print(f"  {name}: 初始 Loss = {losses[0]:.4f} → 最终 Loss = {losses[-1]:.4f}")
    
    # 判断哪个更好
    pre_final = results["Pre-Norm"][-1]
    post_final = results["Post-Norm"][-1]
    winner = "Pre-Norm" if pre_final < post_final else "Post-Norm"
    print(f"\n  🏆 胜者: {winner} (Loss 更低)")
    print(f"  💡 Pre-Norm 的优势：梯度直接通过残差路径流回，不受 LayerNorm 干扰")
    print(f"  💡 现代大模型（GPT、LLaMA、Qwen）全部采用 Pre-Norm")
    print()


# ========================================================
# 5. 完整的 Transformer Block
# ========================================================

class SimpleSelfAttention(nn.Module):
    """简化版 Self-Attention（单头，无 mask）"""
    def __init__(self, d_model):
        super().__init__()
        self.d_k = d_model
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # 计算注意力分数
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ V
        
        return self.W_o(attn_output)


class TransformerBlock(nn.Module):
    """
    完整的 Transformer Decoder Block（Pre-Norm 版本）
    
    结构：
    x → LayerNorm → Self-Attention → (+x) → LayerNorm → FFN → (+x) → output
    
    这就是 GPT、LLaMA 等模型的核心单元！
    """
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        # 子层 1：Self-Attention
        self.norm1 = nn.LayerNorm(d_model)           # Pre-Norm
        self.attn = SimpleSelfAttention(d_model)      # 注意力
        self.dropout1 = nn.Dropout(dropout)
        
        # 子层 2：FFN
        self.norm2 = nn.LayerNorm(d_model)           # Pre-Norm
        self.ffn = nn.Sequential(                     # 前馈网络
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        # --- 子层 1: Self-Attention + 残差 ---
        # 先归一化，再做注意力，再加残差（Pre-Norm）
        attn_out = self.attn(self.norm1(x))
        x = x + self.dropout1(attn_out)              # 残差连接！
        
        # --- 子层 2: FFN + 残差 ---
        # 同样：先归一化，再做 FFN，再加残差
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out                              # 残差连接！
        
        return x


def test_transformer_block():
    """测试完整的 Transformer Block"""
    print("=" * 60)
    print("实验 5: 完整 Transformer Block（Pre-Norm）")
    print("=" * 60)
    
    torch.manual_seed(42)
    d_model = 64
    batch_size = 2
    seq_len = 8
    
    block = TransformerBlock(d_model, d_ff=256)
    
    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)
    
    # 前向传播
    output = block(x)
    
    print(f"  输入形状:  {x.shape}")
    print(f"  输出形状:  {output.shape}")
    print(f"  形状一致:  {'✅' if x.shape == output.shape else '❌'}")
    
    # 统计参数量
    total_params = sum(p.numel() for p in block.parameters())
    attn_params = sum(p.numel() for p in block.attn.parameters())
    ffn_params = sum(p.numel() for p in block.ffn.parameters())
    norm_params = sum(p.numel() for p in list(block.norm1.parameters()) + list(block.norm2.parameters()))
    
    print(f"\n  📊 参数量分布:")
    print(f"     总参数:       {total_params:,}")
    print(f"     Attention:    {attn_params:,} ({attn_params/total_params*100:.1f}%)")
    print(f"     FFN:          {ffn_params:,} ({ffn_params/total_params*100:.1f}%)")
    print(f"     LayerNorm:    {norm_params:,} ({norm_params/total_params*100:.1f}%)")
    
    # 堆叠多层，测试梯度
    print(f"\n  🔄 测试 24 层 Transformer Block 的梯度流动:")
    deep_model = nn.Sequential(*[TransformerBlock(d_model) for _ in range(24)])
    
    output = deep_model(x)
    loss = output.sum()
    loss.backward()
    
    # 检查各层梯度
    grad_health = []
    for i, block in enumerate(deep_model):
        grad = block.attn.W_q.weight.grad
        if grad is not None:
            grad_health.append((i, grad.norm().item()))
    
    if grad_health:
        first = grad_health[0][1]
        last = grad_health[-1][1]
        print(f"     第 1 层梯度:  {first:.6f}")
        print(f"     第 12 层梯度: {grad_health[11][1]:.6f}")
        print(f"     第 24 层梯度: {last:.6f}")
        ratio = last / (first + 1e-10)
        print(f"     梯度衰减比:   {ratio:.4f}")
        print(f"     梯度状态:     {'✅ 健康' if ratio > 0.01 else '⚠️ 注意'}")
    
    print(f"\n  💡 24 层堆叠后梯度仍然健康——这就是 LayerNorm + 残差连接的威力！")
    print()


# ========================================================
# 主函数
# ========================================================

if __name__ == "__main__":
    print("\n" + "🦞" * 30)
    print("  Day 10: Layer Normalization + 残差连接")
    print("  Transformer 的健康通道")
    print("🦞" * 30 + "\n")
    
    # 实验 1：手写 LayerNorm 验证
    test_layernorm_correctness()
    
    # 实验 2：RMSNorm 对比
    test_rmsnorm()
    
    # 实验 3：梯度消失实验
    gradient_health_check()
    
    # 实验 4：Pre-Norm vs Post-Norm
    train_compare_prenorm_postnorm()
    
    # 实验 5：完整 Transformer Block
    test_transformer_block()
    
    print("=" * 60)
    print("🎉 所有实验完成！")
    print("=" * 60)
    print()
    print("📝 今日总结：")
    print("  • LayerNorm 把每层输出归一化到稳定范围（均值0，方差1）")
    print("  • RMSNorm 是精简版，省掉均值计算和β参数，现代LLM标配")
    print("  • 残差连接保证原始信息无损传递，梯度永远不会消失到0")
    print("  • Pre-Norm 梯度流动更顺畅，是现代Transformer的标配")
    print("  • 完整 Block = LayerNorm → Attention → (+残差) → LayerNorm → FFN → (+残差)")
    print()
