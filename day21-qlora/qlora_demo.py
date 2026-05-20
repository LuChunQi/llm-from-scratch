"""
Day 21: QLoRA & LoRA 变体 — 量化与低秩的化学反应

演示内容：
1. NF4 量化：将 FP32 权重量化为 4-bit，再反量化回来
2. QLoRA 线性层：NF4 量化存储 + LoRA 可训练补丁
3. 显存对比：全量微调 vs LoRA vs QLoRA
4. LoRA 变体对比：AdaLoRA / DoRA / LoRA+ / PiSSA 的核心逻辑
5. 量化误差可视化：看看 4-bit 量化到底损失了多少精度
"""

import torch
import torch.nn as nn
import math
import time


# ============================================================
# 1. 简化版 NF4 量化器
# ============================================================

class SimpleNF4Quantizer:
    """简化版 NF4 量化器（教学用途）
    
    真正的 NF4 用正态分布分位数作为量化级别，
    这里用均匀量化 + 分块策略来演示原理。
    """
    
    def __init__(self, block_size=64):
        self.block_size = block_size
    
    def quantize(self, weight):
        """将 FP32 权重量化为 4-bit
        
        Args:
            weight: FP32 张量
        Returns:
            quantized: uint8 张量（值 0-15）
            scale: 每个 block 的缩放因子
            zero_point: 每个 block 的零点
        """
        flat = weight.flatten().clone()
        n = flat.shape[0]
        num_blocks = (n + self.block_size - 1) // self.block_size
        
        quantized = torch.zeros(n, dtype=torch.uint8)
        scale = torch.zeros(num_blocks)
        zero_point = torch.zeros(num_blocks)
        
        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, n)
            block = flat[start:end]
            
            block_min = block.min().item()
            block_max = block.max().item()
            
            # 避免除以零
            if block_max - block_min < 1e-8:
                scale[i] = 1.0
                zero_point[i] = block_min
                quantized[start:end] = 0
            else:
                scale[i] = (block_max - block_min) / 15.0
                zero_point[i] = block_min
                quantized[start:end] = (
                    ((block - block_min) / scale[i])
                    .round()
                    .clamp(0, 15)
                    .to(torch.uint8)
                )
        
        return quantized, scale, zero_point
    
    def dequantize(self, quantized, scale, zero_point, original_shape):
        """将 4-bit 权重反量化回 FP32"""
        n = quantized.shape[0]
        result = torch.zeros(n, dtype=torch.float32)
        
        num_blocks = scale.shape[0]
        for i in range(num_blocks):
            start = i * self.block_size
            end = min(start + self.block_size, n)
            result[start:end] = quantized[start:end].float() * scale[i] + zero_point[i]
        
        return result.view(original_shape)
    
    def measure_error(self, original, reconstructed):
        """测量量化误差"""
        diff = (original - reconstructed).abs()
        mae = diff.mean().item()  # 平均绝对误差
        max_err = diff.max().item()  # 最大误差
        # 相对误差（避免除以零）
        rel_err = (diff / (original.abs() + 1e-8)).mean().item()
        return mae, max_err, rel_err


# ============================================================
# 2. QLoRA 线性层
# ============================================================

class QLoRALinear(nn.Module):
    """NF4 量化存储 + LoRA 可训练补丁
    
    存储用 4-bit 省空间，计算时反量化到 FP32 保精度，
    LoRA 部分全程 FP32 训练。
    """
    
    def __init__(self, in_features, out_features, r=8, alpha=16, block_size=64):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r
        
        # --- 量化权重（冻结）---
        # 初始化为小随机数（模拟预训练权重）
        weight_fp32 = torch.randn(out_features, in_features) * 0.02
        self.quantizer = SimpleNF4Quantizer(block_size)
        quantized, scale, zp = self.quantizer.quantize(weight_fp32)
        
        # 用 register_buffer 保存（不参与梯度，但随模型保存/加载）
        self.register_buffer('weight_4bit', quantized)
        self.register_buffer('q_scale', scale)
        self.register_buffer('q_zero_point', zp)
        self._original_shape = (out_features, in_features)
        
        # --- LoRA 部分（可训练）---
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 初始化为 0：训练开始时 LoRA 贡献为零
    
    def _dequantize_weight(self):
        """反量化：4-bit → FP32"""
        return self.quantizer.dequantize(
            self.weight_4bit, self.q_scale, self.q_zero_point,
            self._original_shape
        )
    
    def forward(self, x):
        """前向传播：量化权重 + LoRA"""
        # 反量化计算
        weight_fp32 = self._dequantize_weight()
        base_output = x @ weight_fp32.t()
        
        # LoRA 补丁
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        return base_output + lora_output


# ============================================================
# 3. LoRA 变体实现
# ============================================================

class AdaLoRALinear(nn.Module):
    """AdaLoRA：自适应 Rank 的 LoRA
    
    核心：不同层可以有不同的 rank，
    训练中根据"重要性分数"动态调整每层的 rank。
    这里演示固定不同 rank 的版本。
    """
    
    def __init__(self, in_features, out_features, r=8, alpha=16, importance=1.0):
        super().__init__()
        # importance 越高 → 给这个层越多的 rank
        self.effective_r = max(2, int(r * importance))  # 至少 rank 2
        
        # 原始权重（冻结）
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02, requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        # LoRA，rank 根据 importance 调整
        actual_r = self.effective_r
        self.lora_A = nn.Parameter(torch.empty(actual_r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, actual_r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.scaling = alpha / actual_r
        self.importance = importance
    
    def forward(self, x):
        base = x @ self.weight.t() + self.bias
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


class DoRALinear(nn.Module):
    """DoRA：权重分解 LoRA
    
    把权重分解为方向（direction）和大小（magnitude）：
      W = m × (V / ||V||)
    LoRA 只调整方向，m 用单独标量调整大小。
    """
    
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        # 原始权重（冻结）
        weight = torch.randn(out_features, in_features) * 0.02
        
        # 分解：m（大小）和 V（方向）
        magnitude = weight.norm(dim=1, keepdim=True)  # 每行的范数
        direction = weight / (magnitude + 1e-8)
        
        self.register_buffer('magnitude', magnitude.detach())
        self.register_buffer('direction', direction.detach())
        
        # LoRA 只调整方向部分
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # 可学习的大小缩放因子
        self.magnitude_scale = nn.Parameter(torch.ones(out_features, 1))
        
        self.scaling = alpha / r
    
    def forward(self, x):
        # 方向 = 冻结方向 + LoRA 调整
        direction = self.direction + (self.lora_B @ self.lora_A) * self.scaling
        # 归一化方向
        direction = direction / (direction.norm(dim=1, keepdim=True) + 1e-8)
        # 大小 = 原始大小 × 可学习缩放
        effective_magnitude = self.magnitude * self.magnitude_scale
        # 重构权重
        weight = effective_magnitude * direction
        return x @ weight.t()


class LoRAPlusLinear(nn.Module):
    """LoRA+：不对称学习率
    
    A 和 B 使用不同的学习率，B 用更高的学习率。
    训练代码中体现（演示构造时标注推荐比例）。
    """
    
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.02, requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.scaling = alpha / r
        # LoRA+ 的关键：推荐 lr_B = 16 × lr_A
        self.lr_ratio = 16  # B 的学习率是 A 的 16 倍
    
    def forward(self, x):
        base = x @ self.weight.t() + self.bias
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora
    
    def get_param_groups(self, base_lr=1e-4):
        """返回不同学习率的参数组，用于优化器"""
        return [
            {'params': [self.lora_A], 'lr': base_lr},
            {'params': [self.lora_B], 'lr': base_lr * self.lr_ratio},
        ]


class PiSSALinear(nn.Module):
    """PiSSA：用 SVD 主成分初始化 LoRA
    
    不从零开始！用原始权重的前 r 个奇异值来初始化 LoRA。
    """
    
    def __init__(self, in_features, out_features, r=8, alpha=16):
        super().__init__()
        weight = torch.randn(out_features, in_features) * 0.02
        
        # SVD 分解
        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
        
        # 前 r 个主成分 → LoRA 初始化
        # A = sqrt(S[:r]) × Vh[:r]
        # B = U[:, :r] × sqrt(S[:r])
        sqrt_s = torch.sqrt(S[:r])
        self.lora_A = nn.Parameter(Vh[:r, :] * sqrt_s.unsqueeze(1))
        self.lora_B = nn.Parameter(U[:, :r] * sqrt_s.unsqueeze(0))
        
        # 剩余部分 → 冻结权重
        residual = weight - self.lora_B @ self.lora_A
        self.register_buffer('weight', residual.detach())
        
        self.scaling = alpha / r
    
    def forward(self, x):
        base = x @ self.weight.t()
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


# ============================================================
# 主程序：演示所有功能
# ============================================================

def main():
    print("=" * 70)
    print("Day 21: QLoRA & LoRA 变体 — 演示")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # ----------------------------------------------------------
    # 演示 1：NF4 量化与误差分析
    # ----------------------------------------------------------
    print("\n" + "─" * 70)
    print("📊 演示 1：NF4 量化误差分析")
    print("─" * 70)
    
    # 创建模拟权重（正态分布，模拟预训练模型权重）
    weight = torch.randn(256, 256) * 0.02
    
    quantizer = SimpleNF4Quantizer(block_size=64)
    quantized, scale, zp = quantizer.quantize(weight)
    reconstructed = quantizer.dequantize(quantized, scale, zp, weight.shape)
    mae, max_err, rel_err = quantizer.measure_error(weight, reconstructed)
    
    print(f"  原始权重形状: {weight.shape}")
    print(f"  原始大小 (FP32): {weight.numel() * 4 / 1024:.1f} KB")
    print(f"  量化后大小 (NF4): {weight.numel() * 0.5 / 1024:.1f} KB")
    print(f"  压缩比: {4 / 0.5:.0f}x")
    print(f"  ─────────────────────────────")
    print(f"  平均绝对误差 (MAE):  {mae:.6f}")
    print(f"  最大绝对误差:         {max_err:.6f}")
    print(f"  平均相对误差:         {rel_err:.4%}")
    
    # 不同 block_size 的对比
    print(f"\n  📏 不同 block_size 的量化误差对比：")
    for bs in [32, 64, 128, 256]:
        q = SimpleNF4Quantizer(block_size=bs)
        qnt, sc, z = q.quantize(weight)
        rec = q.dequantize(qnt, sc, z, weight.shape)
        m, mx, r = q.measure_error(weight, rec)
        print(f"    block_size={bs:4d}: MAE={m:.6f}, 最大误差={mx:.6f}")
    
    # ----------------------------------------------------------
    # 演示 2：QLoRA 线性层
    # ----------------------------------------------------------
    print("\n" + "─" * 70)
    print("🔬 演示 2：QLoRA 线性层前向传播")
    print("─" * 70)
    
    d = 256  # 用小尺寸演示
    qlora = QLoRALinear(d, d, r=8, alpha=16)
    x = torch.randn(4, d)  # batch=4
    
    output = qlora(x)
    print(f"  输入形状:  {x.shape}")
    print(f"  输出形状:  {output.shape}")
    
    # 检查 LoRA 初始贡献为零
    with torch.no_grad():
        weight_deq = qlora._dequantize_weight()
        base_out = x @ weight_deq.t()
        lora_contrib = (x @ qlora.lora_A.T @ qlora.lora_B.T) * qlora.scaling
        print(f"  ─────────────────────────────")
        print(f"  基础输出范数:      {base_out.norm():.4f}")
    print(f"  LoRA 初始贡献范数: {lora_contrib.norm():.6f}  ← B 初始化为 0，贡献≈0 ✅")
    
    # ----------------------------------------------------------
    # 演示 3：QLoRA 训练演示
    # ----------------------------------------------------------
    print("\n" + "─" * 70)
    print("🏋️ 演示 3：QLoRA 训练（5 步演示）")
    print("─" * 70)
    
    model = QLoRALinear(d, d, r=8, alpha=16)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3
    )
    
    for step in range(5):
        x = torch.randn(8, d)
        target = torch.randn(8, d)  # 模拟目标输出
        output = model(x)
        loss = nn.functional.mse_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            lora_norm = model.lora_B @ model.lora_A
        print(f"  Step {step+1}: loss={loss.item():.4f}, LoRA权重范数={lora_norm.norm():.4f}")
    
    print(f"  → LoRA 权重从 0 逐渐增长，正在学习！")
    
    # ----------------------------------------------------------
    # 演示 4：显存对比
    # ----------------------------------------------------------
    print("\n" + "─" * 70)
    print("💾 演示 4：7B 模型各方案的显存估算")
    print("─" * 70)
    
    # 简化估算：一个 Transformer 层的显存
    hidden = 4096
    ffn_dim = 11008
    num_layers = 32
    
    # 每层的线性层参数量
    attn_params = 4 * hidden * hidden  # Q, K, V, O
    ffn_params = 3 * hidden * ffn_dim  # W1, W2, W3 (LLaMA 用 SwiGLU)
    params_per_layer = attn_params + ffn_params
    total_params = params_per_layer * num_layers  # ≈ 6.7B
    
    def fmt_size(bytes_val):
        if bytes_val >= 1e9: return f"{bytes_val/1e9:.1f} GB"
        elif bytes_val >= 1e6: return f"{bytes_val/1e6:.0f} MB"
        else: return f"{bytes_val/1e3:.0f} KB"
    
    configs = [
        ("全量 FP16 微调", 2, 2, 8, 0),       # 权重, 梯度, 优化器, LoRA
        ("LoRA FP16 (r=8)", 2, 0, 0, 2),       # 权重FP16, LoRA有梯度+优化器
        ("QLoRA NF4 (r=8)", 0.5, 0, 0, 2),     # 权重NF4, LoRA FP16
        ("QLoRA + 双重量化", 0.375, 0, 0, 2),  # NF4 + 双重量化省元数据
    ]
    
    print(f"  {'方案':<22s} {'权重':>8s} {'梯度':>8s} {'优化器':>8s} {'LoRA':>8s} {'总计':>10s}")
    print(f"  {'─'*70}")
    
    for name, w_bytes, g_bytes, o_bytes, l_bytes in configs:
        weight_mem = total_params * w_bytes
        grad_mem = total_params * g_bytes
        
        if o_bytes > 0:  # 全量微调：优化器覆盖所有参数
            optim_mem = total_params * o_bytes
        else:  # LoRA/QLoRA：优化器只覆盖 LoRA 参数
            # LoRA 参数：每层 4 个线性层 × 2 矩阵 × 2 × rank × hidden
            lora_params = 4 * 2 * 8 * hidden * num_layers
            optim_mem = lora_params * 8  # Adam: FP32 的 m + v
        
        lora_mem = 0
        if l_bytes > 0:
            lora_params = 4 * 2 * 8 * hidden * num_layers
            lora_mem = lora_params * l_bytes
        
        total = weight_mem + grad_mem + optim_mem + lora_mem
        print(f"  {name:<22s} {fmt_size(weight_mem):>8s} {fmt_size(grad_mem):>8s} "
              f"{fmt_size(optim_mem):>8s} {fmt_size(lora_mem):>8s} {fmt_size(total):>10s}")
    
    # ----------------------------------------------------------
    # 演示 5：LoRA 变体对比
    # ----------------------------------------------------------
    print("\n" + "─" * 70)
    print("🧬 演示 5：LoRA 变体对比")
    print("─" * 70)
    
    d = 64
    x = torch.randn(2, d)
    
    # 标准 QLoRA
    qlora_layer = QLoRALinear(d, d, r=4, alpha=8)
    
    # AdaLoRA：不同层不同 rank
    adalora_high = AdaLoRALinear(d, d, r=8, importance=1.5)  # 重要层 rank=12
    adalora_low = AdaLoRALinear(d, d, r=8, importance=0.5)   # 不重要 rank=4
    
    # DoRA
    dora_layer = DoRALinear(d, d, r=4, alpha=8)
    
    # LoRA+
    loraplus_layer = LoRAPlusLinear(d, d, r=4, alpha=8)
    param_groups = loraplus_layer.get_param_groups(base_lr=1e-4)
    
    # PiSSA
    pissa_layer = PiSSALinear(d, d, r=4, alpha=8)
    
    print(f"  {'变体':<16s} {'有效 Rank':>10s} {'可训练参数':>12s} {'输出范数':>10s}")
    print(f"  {'─'*55}")
    
    with torch.no_grad():
        out_qlora = qlora_layer(x)
        q_params = sum(p.numel() for p in qlora_layer.parameters() if p.requires_grad)
        print(f"  {'QLoRA':<16s} {'4':>10s} "
              f"{q_params:>12d} "
              f"{out_qlora.norm():>10.4f}")
        
        ah_params = sum(p.numel() for p in adalora_high.parameters() if p.requires_grad)
        out_ada_h = adalora_high(x)
        print(f"  {'AdaLoRA(高)':<16s} {adalora_high.effective_r:>10d} "
              f"{ah_params:>12d} "
              f"{out_ada_h.norm():>10.4f}")
        
        al_params = sum(p.numel() for p in adalora_low.parameters() if p.requires_grad)
        out_ada_l = adalora_low(x)
        print(f"  {'AdaLoRA(低)':<16s} {adalora_low.effective_r:>10d} "
              f"{al_params:>12d} "
              f"{out_ada_l.norm():>10.4f}")
        
        do_params = sum(p.numel() for p in dora_layer.parameters() if p.requires_grad)
        out_dora = dora_layer(x)
        print(f"  {'DoRA':<16s} {'4':>10s} "
              f"{do_params:>12d} "
              f"{out_dora.norm():>10.4f}")
        
        lp_params = sum(p.numel() for p in loraplus_layer.parameters() if p.requires_grad)
        out_plus = loraplus_layer(x)
        print(f"  {'LoRA+':<16s} {'4':>10s} "
              f"{lp_params:>12d} "
              f"{out_plus.norm():>10.4f}")
        
        ps_params = sum(p.numel() for p in pissa_layer.parameters() if p.requires_grad)
        out_pissa = pissa_layer(x)
        print(f"  {'PiSSA':<16s} {'4':>10s} "
              f"{ps_params:>12d} "
              f"{out_pissa.norm():>10.4f}")
    
    # LoRA+ 学习率信息
    print(f"\n  📌 LoRA+ 参数组（不对称学习率）：")
    for i, pg in enumerate(param_groups):
        print(f"    组 {i+1}: lr={pg['lr']:.6f}, params={pg['params'][0].shape}")
    
    # ----------------------------------------------------------
    # 演示 6：PiSSA 的 SVD 初始化效果
    # ----------------------------------------------------------
    print("\n" + "─" * 70)
    print("🎯 演示 6：PiSSA vs 标准 LoRA 初始权重近似质量")
    print("─" * 70)
    
    # 用一个固定权重来对比
    weight_full = torch.randn(d, d) * 0.02
    r = 4
    
    # 标准 LoRA：A 随机初始化，B = 0 → 初始近似 = 原始权重（LoRA 没贡献）
    lora_A_std = torch.randn(r, d) * 0.01
    lora_B_std = torch.zeros(d, r)
    std_approx = weight_full  # LoRA 贡献为 0
    
    # PiSSA：用 SVD 初始化 → 初始近似更接近原始权重
    U, S, Vh = torch.linalg.svd(weight_full, full_matrices=False)
    pissa_A = Vh[:r, :] * torch.sqrt(S[:r]).unsqueeze(1)
    pissa_B = U[:, :r] * torch.sqrt(S[:r]).unsqueeze(0)
    pissa_residual = weight_full - pissa_B @ pissa_A
    pissa_approx = pissa_residual + pissa_B @ pissa_A  # = weight_full
    
    std_residual = weight_full - 0  # 标准 LoRA 的 "residual" 就是完整权重
    
    print(f"  原始权重范数:              {weight_full.norm():.4f}")
    print(f"  ─────────────────────────────")
    print(f"  标准 LoRA 残差范数:        {std_residual.norm():.4f}  (≈ 原始权重，LoRA 还没开始学)")
    print(f"  PiSSA 残差范数:            {pissa_residual.norm():.4f}  (已经拿走主成分！)")
    print(f"  PiSSA 捕捉的能量占比:      {1 - (pissa_residual.norm()/weight_full.norm())**2:.2%}")
    
    # ----------------------------------------------------------
    # 总结
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("✅ 总结")
    print("=" * 70)
    print("""
  QLoRA 的三大创新：
    1. NF4 量化 — 按正态分布分位数设计 4-bit 级别，精度更高
    2. 双重量化 — 对量化的元数据再做一次量化，再省 ~0.5GB
    3. 分页优化器 — GPU/CPU 自动换页，永不 OOM

  效果：7B 模型微调从 86GB → ~6GB，消费级显卡就能跑！

  LoRA 变体全家福：
    • AdaLoRA — 自适应 rank，参数分配更聪明
    • DoRA     — 分解方向+大小，更接近全量微调
    • LoRA+   — A/B 不对称学习率，收敛更快
    • rsLoRA  — √r 缩放，高 rank 更稳定
    • PiSSA   — SVD 初始化，起步就领先

  实践建议：不确定选哪个 → QLoRA + r=8 + α=16，最成熟最稳
    """)
    print("GitHub: https://github.com/LuChunQi/llm-from-scratch")


if __name__ == "__main__":
    main()
