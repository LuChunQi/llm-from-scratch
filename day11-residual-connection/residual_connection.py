"""
Day 11: 残差连接深入 — 为什么深层网络非它不可
================================================

实验内容：
1. 梯度传播可视化 — 对比有/无残差时每层梯度的大小，直观感受梯度消失
2. 信息保留实验 — 验证残差连接是否真的保留了原始输入信息
3. 不同深度的训练对比 — 从 4 层到 64 层，看残差连接如何影响训练效果
4. 残差流分析 — 可视化每一层在残差流中添加了多少新信息
5. 残差变体对比 — 标准残差 vs 门控残差 vs 缩放残差

运行方式：python3 residual_connection.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================================================
# 1. 梯度传播可视化 — 眼见为实
# ========================================================

class PlainBlock(nn.Module):
    """无残差的变换块：Linear → ReLU → Linear"""
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """带残差的变换块：(Linear → ReLU → Linear) + x"""
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, x):
        return self.net(x) + x    # 关键区别：加上原始输入 x


class ResidualBlockWithNorm(nn.Module):
    """完整的 Pre-Norm 残差块：x + (Linear → ReLU → Linear)(LayerNorm(x))"""
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, x):
        return x + self.net(self.norm(x))    # Pre-Norm + 残差


def gradient_propagation_visualization():
    """
    实验 1：梯度传播可视化
    
    构建 30 层深度网络，分别用无残差 / 有残差 / 有残差+归一化
    观察梯度在各层的分布
    """
    print("=" * 60)
    print("实验 1: 梯度传播可视化 — 30 层深度网络")
    print("=" * 60)
    
    torch.manual_seed(42)
    depth = 30
    d_model = 32
    batch_size = 4
    
    # 构建三种网络
    configs = {
        "无残差 (Plain)": nn.Sequential(*[PlainBlock(d_model) for _ in range(depth)]),
        "有残差 (Residual)": nn.Sequential(*[ResidualBlock(d_model) for _ in range(depth)]),
        "残差 + LayerNorm": nn.Sequential(*[ResidualBlockWithNorm(d_model) for _ in range(depth)]),
    }
    
    x = torch.randn(batch_size, d_model)
    target = torch.randn(batch_size, d_model)
    
    for name, model in configs.items():
        # 前向 + 反向
        output = model(x)
        loss = F.mse_loss(output, target)
        loss.backward()
        
        # 收集每层的梯度范数
        grad_norms = []
        for i, block in enumerate(model):
            # 每个 block 的 net[0]（第一个 Linear 层）有 weight
            weight = None
            for module in block.modules():
                if isinstance(module, nn.Linear) and weight is None:
                    weight = module
            if weight is not None and weight.weight.grad is not None:
                grad_norms.append((i, weight.weight.grad.norm().item()))
        
        # 打印梯度分布
        print(f"\n  📊 {name}:")
        
        # 显示采样点的梯度
        sample_layers = [0, 4, 9, 14, 19, 24, 29]
        for layer_idx in sample_layers:
            if layer_idx < len(grad_norms):
                _, g = grad_norms[layer_idx]
                bar = "█" * max(1, int(g * 10))   # 用方块条表示梯度大小
                print(f"     Layer {layer_idx+1:2d}: {g:8.5f}  {bar}")
        
        # 梯度衰减比
        if len(grad_norms) >= 2:
            first = grad_norms[0][1]
            last = grad_norms[-1][1]
            ratio = last / (first + 1e-10)
            if ratio < 1e-4:
                status = "❌ 严重梯度消失"
            elif ratio < 0.01:
                status = "⚠️ 轻度梯度消失"
            else:
                status = "✅ 梯度健康"
            print(f"     底层→顶层衰减比: {ratio:.6f}  {status}")
        
        model.zero_grad()
    
    print()
    print("  💡 结论：无残差网络在 30 层后梯度几乎消失")
    print("  💡 残差连接通过 (1 + ∂F/∂x) 的机制，保证梯度不会消失到 0")
    print("  💡 加上 LayerNorm 后梯度更加平稳，不会有突变")
    print()


# ========================================================
# 2. 信息保留实验 — 残差连接真的保留了原始信息吗？
# ========================================================

def information_preservation_experiment():
    """
    实验 2：信息保留实验
    
    验证：经过多层变换后，原始输入的信息是否还在？
    方法：测量最终输出和原始输入的余弦相似度
    """
    print("=" * 60)
    print("实验 2: 信息保留实验 — 输出还记得输入吗？")
    print("=" * 60)
    
    torch.manual_seed(42)
    d_model = 32
    
    x = torch.randn(1, d_model)     # 原始输入，形状 [1, d_model]
    x_norm = x / x.norm()           # 归一化版本
    
    depths = [4, 8, 16, 32, 64]
    
    for depth in depths:
        # 无残差网络
        plain_layers = nn.Sequential(*[PlainBlock(d_model) for _ in range(depth)])
        with torch.no_grad():
            plain_out = plain_layers(x)
            plain_out_norm = plain_out / (plain_out.norm() + 1e-8)
            # 余弦相似度：沿最后一个维度计算
            plain_sim = F.cosine_similarity(x_norm, plain_out_norm, dim=-1).mean().item()
        
        # 有残差网络
        res_layers = nn.Sequential(*[ResidualBlock(d_model) for _ in range(depth)])
        with torch.no_grad():
            res_out = res_layers(x)
            res_out_norm = res_out / (res_out.norm() + 1e-8)
            res_sim = F.cosine_similarity(x_norm, res_out_norm, dim=-1).mean().item()
        
        # 有残差 + LayerNorm
        res_ln_layers = nn.Sequential(*[ResidualBlockWithNorm(d_model) for _ in range(depth)])
        with torch.no_grad():
            res_ln_out = res_ln_layers(x)
            res_ln_norm = res_ln_out / (res_ln_out.norm() + 1e-8)
            res_ln_sim = F.cosine_similarity(x_norm, res_ln_norm, dim=-1).mean().item()
        
        print(f"  深度 {depth:2d} 层:")
        print(f"    无残差:        cos_sim = {plain_sim:+.4f}  {'❌ 信息丢失' if abs(plain_sim) < 0.1 else '⚠️ 部分保留' if abs(plain_sim) < 0.5 else '✅ 信息保留'}")
        print(f"    有残差:        cos_sim = {res_sim:+.4f}  {'❌ 信息丢失' if abs(res_sim) < 0.1 else '⚠️ 部分保留' if abs(res_sim) < 0.5 else '✅ 信息保留'}")
        print(f"    残差+LayerNorm: cos_sim = {res_ln_sim:+.4f}  {'❌ 信息丢失' if abs(res_ln_sim) < 0.1 else '⚠️ 部分保留' if abs(res_ln_sim) < 0.5 else '✅ 信息保留'}")
    
    print()
    print("  💡 无残差：深度增加后，输出和输入的相似度迅速降低（信息被覆盖）")
    print("  💡 有残差：无论多深，输出始终包含输入信息（x 直通路径）")
    print("  💡 残差 + LayerNorm：相似度最稳定，归一化让数值更可控")
    print()


# ========================================================
# 3. 不同深度的训练对比 — 残差连接让"越深越好"成为可能
# ========================================================

def depth_training_comparison():
    """
    实验 3：不同深度的训练对比
    
    用同样的数据、同样的训练步数，对比不同深度网络的训练 Loss
    无残差：深度增加 → Loss 不降反升（退化）
    有残差：深度增加 → Loss 持续降低（越深越强）
    """
    print("=" * 60)
    print("实验 3: 不同深度的训练对比 — 越深一定越好吗？")
    print("=" * 60)
    
    torch.manual_seed(42)
    d_model = 32
    lr = 1e-3
    steps = 300
    
    # 训练数据：x → 非线性目标
    x_train = torch.randn(32, d_model)
    y_train = torch.sin(x_train) + 0.5 * torch.cos(x_train * 2)
    
    depths = [4, 8, 16, 32]
    
    print(f"\n  训练 {steps} 步后的 Loss:")
    print(f"  {'深度':>6s}  {'无残差':>10s}  {'有残差':>10s}  {'残差+LN':>10s}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*10}")
    
    for depth in depths:
        results = {}
        for name, BlockClass in [
            ("无残差", PlainBlock),
            ("有残差", ResidualBlock),
            ("残差+LN", ResidualBlockWithNorm),
        ]:
            torch.manual_seed(42)
            model = nn.Sequential(*[BlockClass(d_model) for _ in range(depth)])
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            
            for _ in range(steps):
                output = model(x_train)
                loss = F.mse_loss(output, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            results[name] = F.mse_loss(model(x_train), y_train).item()
        
        print(f"  {depth:4d} 层  {results['无残差']:10.4f}  {results['有残差']:10.4f}  {results['残差+LN']:10.4f}")
    
    print()
    print("  💡 无残差：超过 16 层后 Loss 开始升高（网络退化，不是过拟合）")
    print("  💡 有残差：深度增加，Loss 持续下降（越深越强！）")
    print("  💡 残差+LN：训练最稳定，Loss 最低")
    print("  💡 这就是 ResNet 论文的核心发现：深层网络退化不是过拟合，是优化困难")
    print()


# ========================================================
# 4. 残差流分析 — 每一层往"河流"里注入了多少信息？
# ========================================================

class TransformerResidualBlock(nn.Module):
    """简化 Transformer Block，用于残差流分析"""
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
    
    def forward(self, x):
        # 保存输入（残差流"之前"的状态）
        residual = x
        # Pre-Norm + Attention + 残差
        normed = self.norm(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = residual + attn_out
        # Pre-Norm + FFN + 残差
        x = x + self.ffn(self.norm2(x))
        return x


def residual_stream_analysis():
    """
    实验 4：残差流分析
    
    观察：每一层的输出和输入之间差了多少？
    差值 = 该层注入残差流的新信息量
    """
    print("=" * 60)
    print("实验 4: 残差流分析 — 每层注入了多少新信息？")
    print("=" * 60)
    
    torch.manual_seed(42)
    d_model = 64
    n_layers = 12
    seq_len = 8
    batch_size = 2
    
    # 构建 12 层 Transformer
    layers = nn.ModuleList([TransformerResidualBlock(d_model) for _ in range(n_layers)])
    
    # 初始 embedding
    x = torch.randn(batch_size, seq_len, d_model)
    
    print(f"\n  初始 embedding 范数: {x.norm():.4f}")
    print()
    header = "  {:>4s}  {:>10s}  {:>10s}  {:>10s}  {:>8s}".format('层', '输出范数', '增量范数', '增量占比', '信息条')
    print(header)
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*20}")
    
    with torch.no_grad():
        current = x.clone()
        for i, layer in enumerate(layers):
            prev = current.clone()
            current = layer(current)
            
            # 增量 = 当前输出 - 上一层的输出 = 该层注入的新信息
            delta = current - prev
            delta_norm = delta.norm().item()
            output_norm = current.norm().item()
            delta_ratio = delta_norm / (output_norm + 1e-10) * 100
            
            # 用信息条可视化
            bar_len = max(1, int(delta_ratio / 2))
            bar = "▓" * bar_len + "░" * max(0, 20 - bar_len)
            
            print(f"  {i+1:3d}  {output_norm:10.4f}  {delta_norm:10.4f}  {delta_ratio:8.1f}%  {bar}")
    
    print()
    print("  💡 每层注入的增量（新信息）占输出的一定比例")
    print('  💡 增量比例适中（不过大也不过小），说明残差连接让每层做"微调"而非"重写"')
    print("  💡 输出范数随层数缓慢增长（因为每层都在加信息），但不会爆炸")
    print()


# ========================================================
# 5. 残差变体对比 — 标准残差 vs 门控残差 vs 缩放残差
# ========================================================

class GatedResidualBlock(nn.Module):
    """
    门控残差连接
    output = α · F(x) + (1 - α) · x
    α 由 sigmoid 生成，范围 [0, 1]
    """
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # 门控参数：决定保留多少原始信息 vs 新信息
        self.gate = nn.Linear(d_model * 2, d_model)
    
    def forward(self, x):
        f_x = self.net(x)
        # 拼接 F(x) 和 x，通过 sigmoid 生成门控权重
        alpha = torch.sigmoid(self.gate(torch.cat([f_x, x], dim=-1)))
        return alpha * f_x + (1 - alpha) * x


class ScaledResidualBlock(nn.Module):
    """
    缩放残差连接（DeepNorm 风格）
    output = F(x) / α + x
    α 随深度增长，用于超深网络
    """
    def __init__(self, d_model, depth, alpha_fn=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        # α = (2N)^0.25（DeepNorm 公式），N = 总层数
        if alpha_fn is None:
            self.alpha = (2 * depth) ** 0.25
        else:
            self.alpha = alpha_fn(depth)
    
    def forward(self, x):
        return self.net(x) / self.alpha + x


def residual_variant_comparison():
    """
    实验 5：对比三种残差变体的训练效果
    标准残差 / 门控残差 / 缩放残差
    """
    print("=" * 60)
    print("实验 5: 残差变体对比 — 标准 vs 门控 vs 缩放")
    print("=" * 60)
    
    torch.manual_seed(42)
    d_model = 32
    depth = 32
    lr = 1e-3
    steps = 300
    
    x_train = torch.randn(32, d_model)
    y_train = torch.sin(x_train) + 0.3 * x_train
    
    variants = {}
    
    # 标准残差
    torch.manual_seed(42)
    model_std = nn.Sequential(*[ResidualBlockWithNorm(d_model) for _ in range(depth)])
    opt = torch.optim.Adam(model_std.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.mse_loss(model_std(x_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    variants["标准残差 F(x)+x"] = F.mse_loss(model_std(x_train), y_train).item()
    
    # 门控残差
    torch.manual_seed(42)
    model_gate = nn.Sequential(*[GatedResidualBlock(d_model) for _ in range(depth)])
    opt = torch.optim.Adam(model_gate.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.mse_loss(model_gate(x_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    variants["门控残差 αF(x)+(1-α)x"] = F.mse_loss(model_gate(x_train), y_train).item()
    
    # 缩放残差
    torch.manual_seed(42)
    model_scaled = nn.Sequential(*[ScaledResidualBlock(d_model, depth) for _ in range(depth)])
    opt = torch.optim.Adam(model_scaled.parameters(), lr=lr)
    for _ in range(steps):
        loss = F.mse_loss(model_scaled(x_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    variants["缩放残差 F(x)/α+x"] = F.mse_loss(model_scaled(x_train), y_train).item()
    
    # 参数量对比
    print(f"\n  📊 {depth} 层网络，训练 {steps} 步:")
    print(f"  {'变体':>25s}  {'最终 Loss':>10s}  {'参数量':>10s}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}")
    
    models_dict = {
        "标准残差 F(x)+x": model_std,
        "门控残差 αF(x)+(1-α)x": model_gate,
        "缩放残差 F(x)/α+x": model_scaled,
    }
    
    for name, loss_val in variants.items():
        params = sum(p.numel() for p in models_dict[name].parameters())
        print(f"  {name:>25s}  {loss_val:10.4f}  {params:>10,}")
    
    print()
    print("  💡 标准残差：最简单、参数最少，通常效果就很好")
    print("  💡 门控残差：多了门控参数（参数量增加），但 32 层时优势不明显")
    print("  💡 缩放残差：通过缩小 F(x) 防止深层累积过大，适合 100+ 层的超深网络")
    print("  💡 实践建议：普通深度（<100层）用标准残差即可，超深网络考虑缩放残差")
    print()


# ========================================================
# 主函数
# ========================================================

if __name__ == "__main__":
    print("\n" + "🦞" * 30)
    print("  Day 11: 残差连接深入")
    print("  为什么深层网络非它不可")
    print("🦞" * 30 + "\n")
    
    # 实验 1：梯度传播可视化
    gradient_propagation_visualization()
    
    # 实验 2：信息保留实验
    information_preservation_experiment()
    
    # 实验 3：不同深度训练对比
    depth_training_comparison()
    
    # 实验 4：残差流分析
    residual_stream_analysis()
    
    # 实验 5：残差变体对比
    residual_variant_comparison()
    
    print("=" * 60)
    print("🎉 所有实验完成！")
    print("=" * 60)
    print()
    print("📝 今日总结：")
    print("  • 残差连接通过 (1 + ∂F/∂x) 保证梯度不消失，创造无数条并联信息路径")
    print("  • 零参数成本：只是一个加法操作，却解决了深度学习的核心难题")
    print("  • 在 Transformer 中，残差是'信息总线'——底层特征在高层仍然保留")
    print("  • 残差流视角：每层向信息流中注入增量，不覆盖旧信息")
    print("  • 标准加法残差在大多数场景下就是最优选择")
    print()
