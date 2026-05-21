#!/usr/bin/env python3
"""
Day 22: 量化基础 — 从 FP16 到 INT4 的数学之旅
================================================

手写线性量化器，包含：
1. 对称量化 & 非对称量化
2. 逐张量 & 逐通道量化
3. 不同精度（INT8/INT4/INT2）的量化误差分析
4. 模拟简单模型的量化前后对比
5. 可视化量化误差分布

运行方式：python3 quantization_basics.py
"""

import torch
import torch.nn as nn
import numpy as np
import math

# ============================================================
# 第一部分：基础量化器实现
# ============================================================


class SymmetricQuantizer:
    """
    对称量化器（zero_point = 0）
    
    原理：将浮点范围 [-max_abs, max_abs] 映射到 [-qmax, qmax]
    适合权重（通常关于 0 对称分布）
    """
    
    def __init__(self, n_bits=8):
        """
        Args:
            n_bits: 量化位数（4=INT4, 8=INT8）
        """
        self.n_bits = n_bits
        # 对称量化范围：[-2^(n_bits-1)+1, 2^(n_bits-1)-1]
        # 例如 INT8: [-127, 127]，不用 [-128, 127] 以保持对称
        self.qmax = 2 ** (n_bits - 1) - 1
        self.qmin = -self.qmax
        self.scale = None  # 缩放因子
    
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        量化：FP32/FP16 → INT
        
        公式：Q = clamp(round(W / scale), qmin, qmax)
        其中 scale = max(|W|) / qmax
        """
        # 计算缩放因子：让最大绝对值正好映射到 qmax
        max_abs = tensor.abs().max()
        # 避免除零
        self.scale = max_abs / self.qmax if max_abs > 0 else torch.tensor(1.0)
        
        # 量化：缩放 + 取整 + 截断
        quantized = torch.clamp(
            torch.round(tensor / self.scale),
            self.qmin, self.qmax
        ).to(torch.int32)
        
        return quantized
    
    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        反量化：INT → FP32
        
        公式：W_hat = Q × scale
        对称量化不需要 zero_point，直接乘 scale
        """
        return quantized.float() * self.scale


class AsymmetricQuantizer:
    """
    非对称量化器（zero_point 不为 0）
    
    原理：将浮点范围 [W_min, W_max] 映射到 [qmin, qmax]
    适合激活值（可能偏向一侧，如 ReLU 后全为正）
    """
    
    def __init__(self, n_bits=8):
        self.n_bits = n_bits
        # 非对称量化范围：[0, 2^n_bits - 1] 或 [-2^(n_bits-1), 2^(n_bits-1)-1]
        self.qmin = 0
        self.qmax = 2 ** n_bits - 1
        self.scale = None
        self.zero_point = None
    
    def quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        量化：FP → INT
        
        公式：
            scale = (W_max - W_min) / (qmax - qmin)
            zero_point = qmin - round(W_min / scale)
            Q = clamp(round(W / scale) + zero_point, qmin, qmax)
        """
        t_min = tensor.min()
        t_max = tensor.max()
        
        # 计算缩放因子
        self.scale = (t_max - t_min) / (self.qmax - self.qmin) if t_max > t_min else torch.tensor(1.0)
        
        # 计算零点：让 W_min 正好映射到 qmin
        self.zero_point = torch.clamp(
            torch.round(self.qmin - t_min / self.scale),
            self.qmin, self.qmax
        ).to(torch.int32)
        
        # 量化
        quantized = torch.clamp(
            torch.round(tensor / self.scale) + self.zero_point.float(),
            self.qmin, self.qmax
        ).to(torch.int32)
        
        return quantized
    
    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        反量化：INT → FP
        
        公式：W_hat = (Q - zero_point) × scale
        """
        return (quantized.float() - self.zero_point.float()) * self.scale


# ============================================================
# 第二部分：逐通道量化
# ============================================================


class PerChannelQuantizer:
    """
    逐通道对称量化器
    
    对权重的每个输出通道（每行）独立量化
    每个 channel 有自己的 scale，精度更高
    """
    
    def __init__(self, n_bits=8, axis=0):
        """
        Args:
            n_bits: 量化位数
            axis: 沿哪个轴量化（0=按行/输出通道）
        """
        self.n_bits = n_bits
        self.qmax = 2 ** (n_bits - 1) - 1
        self.qmin = -self.qmax
        self.axis = axis
        self.scales = None
    
    def quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """
        逐通道量化
        
        对 weight 的每一行（axis=0）独立计算 scale
        """
        # 沿 axis=1（每行内部）找最大绝对值 → 每行一个 scale
        max_abs = weight.abs().max(dim=self.axis, keepdim=True)[0]
        self.scales = max_abs / self.qmax
        # 避免除零
        self.scales = torch.where(self.scales > 0, self.scales, torch.ones_like(self.scales))
        
        # 量化：每行除以自己的 scale
        quantized = torch.clamp(
            torch.round(weight / self.scales),
            self.qmin, self.qmax
        ).to(torch.int32)
        
        return quantized
    
    def dequantize(self, quantized: torch.Tensor) -> torch.Tensor:
        """
        逐通道反量化：每行乘以自己的 scale
        """
        return quantized.float() * self.scales


# ============================================================
# 第三部分：量化误差分析
# ============================================================


def analyze_quantization_error(original: torch.Tensor, reconstructed: torch.Tensor, label: str):
    """
    分析量化误差的统计指标
    
    Args:
        original: 原始浮点张量
        reconstructed: 量化后反量化的张量
        label: 标签（用于打印）
    """
    error = (original - reconstructed).abs()
    
    print(f"\n{'='*60}")
    print(f"  📊 {label}")
    print(f"{'='*60}")
    print(f"  平均绝对误差 (MAE):       {error.mean():.6f}")
    print(f"  最大绝对误差 (Max AE):    {error.max():.6f}")
    print(f"  均方误差 (MSE):           {(error**2).mean():.8f}")
    print(f"  相对误差 (平均):          {(error / (original.abs() + 1e-8)).mean():.4f}")
    print(f"  余弦相似度:               {torch.nn.functional.cosine_similarity(original.flatten().unsqueeze(0), reconstructed.flatten().unsqueeze(0)).item():.6f}")
    
    # 误差分布直方图（用文本表示）
    error_flat = error.flatten()
    print(f"\n  误差分布:")
    bins = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, float('inf')]
    for i in range(len(bins) - 1):
        count = ((error_flat >= bins[i]) & (error_flat < bins[i+1])).sum().item()
        pct = count / error_flat.numel() * 100
        bar = '█' * int(pct / 2)
        upper = f"{bins[i+1]}" if bins[i+1] != float('inf') else "∞"
        print(f"    [{bins[i]:.3f}, {upper:>5s})  {pct:5.1f}%  {bar}")


def demo_symmetric_vs_asymmetric():
    """
    演示：对称量化 vs 非对称量化在不同分布上的表现
    """
    print("\n" + "="*70)
    print("  🔬 实验 1：对称量化 vs 非对称量化")
    print("="*70)
    
    torch.manual_seed(42)
    
    # 场景 A：正态分布权重（关于 0 对称）
    weights_symmetric = torch.randn(256) * 0.3
    
    print("\n  📌 场景 A：正态分布权重 N(0, 0.3)")
    print(f"     范围: [{weights_symmetric.min():.4f}, {weights_symmetric.max():.4f}]")
    
    # 对称量化
    sq = SymmetricQuantizer(n_bits=8)
    q_sym = sq.quantize(weights_symmetric)
    r_sym = sq.dequantize(q_sym)
    analyze_quantization_error(weights_symmetric, r_sym, "对称量化 (INT8)")
    
    # 非对称量化
    aq = AsymmetricQuantizer(n_bits=8)
    q_asym = aq.quantize(weights_symmetric)
    r_asym = aq.dequantize(q_asym)
    analyze_quantization_error(weights_symmetric, r_asym, "非对称量化 (INT8)")
    
    # 场景 B：偏移分布（模拟 ReLU 后的激活值，全为正）
    weights_shifted = torch.rand(256) * 2.0 + 0.5  # 范围 [0.5, 2.5]
    
    print(f"\n  📌 场景 B：偏移分布（全为正）[0.5, 2.5]")
    print(f"     范围: [{weights_shifted.min():.4f}, {weights_shifted.max():.4f}]")
    
    sq2 = SymmetricQuantizer(n_bits=8)
    q_sym2 = sq2.quantize(weights_shifted)
    r_sym2 = sq2.dequantize(q_sym2)
    analyze_quantization_error(weights_shifted, r_sym2, "对称量化 (INT8)")
    
    aq2 = AsymmetricQuantizer(n_bits=8)
    q_asym2 = aq2.quantize(weights_shifted)
    r_asym2 = aq2.dequantize(q_asym2)
    analyze_quantization_error(weights_shifted, r_asym2, "非对称量化 (INT8)")
    
    print("\n  💡 结论：对于对称分布，两种方法差不多；")
    print("     对于偏移分布，非对称量化精度更高（充分利用了量化级别）")


def demo_different_precisions():
    """
    演示：不同精度（INT8/INT4/INT2）的量化误差
    """
    print("\n" + "="*70)
    print("  🔬 实验 2：不同精度等级的量化误差对比")
    print("="*70)
    
    torch.manual_seed(42)
    
    # 模拟真实权重的正态分布
    weights = torch.randn(4096) * 0.3  # 典型权重分布
    
    print(f"\n  模拟权重：4096 个参数，正态分布 N(0, 0.3)")
    print(f"  范围: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  标准差: {weights.std():.4f}")
    
    precisions = [
        ("INT8 (256 级)", 8),
        ("INT4 (16 级)", 4),
        ("INT2 (4 级)", 2),
    ]
    
    for name, bits in precisions:
        q = SymmetricQuantizer(n_bits=bits)
        quantized = q.quantize(weights)
        reconstructed = q.dequantize(quantized)
        
        # 存储大小
        original_size = weights.numel() * 4  # FP32 = 4 bytes
        quantized_size = weights.numel() * bits / 8  # 量化后大小
        compression_ratio = original_size / quantized_size
        
        analyze_quantization_error(weights, reconstructed, f"{name} 对称量化")
        print(f"  存储压缩比: {compression_ratio:.1f}x（{original_size/1024:.1f}KB → {quantized_size/1024:.1f}KB）")
        
        # 显示量化级别使用情况
        unique_levels = torch.unique(quantized).numel()
        total_levels = 2 ** bits
        print(f"  量化级别利用率: {unique_levels}/{total_levels} ({unique_levels/total_levels*100:.0f}%)")
    
    print("\n  💡 结论：")
    print("     INT8 → 几乎无损，压缩 4x")
    print("     INT4 → 微小误差，压缩 8x，大多数场景可接受")
    print("     INT2 → 误差较大，压缩 16x，通常不可用")


def demo_per_tensor_vs_per_channel():
    """
    演示：逐张量量化 vs 逐通道量化
    """
    print("\n" + "="*70)
    print("  🔬 实验 3：逐张量 vs 逐通道量化精度对比")
    print("="*70)
    
    torch.manual_seed(42)
    
    # 模拟权重矩阵：不同通道的方差差异很大
    n_out, n_in = 128, 256
    weight = torch.randn(n_out, n_in) * 0.3
    
    # 人为制造几行范围特别大的通道
    weight[10] = weight[10] * 15  # 第 10 行范围很大
    weight[50] = weight[50] * 20  # 第 50 行范围很大
    weight[90] = weight[90] * 10  # 第 90 行范围很大
    
    print(f"\n  模拟权重矩阵: {n_out}×{n_in}")
    print(f"  大部分通道范围: [{(weight.abs().mean(dim=1).min()):.4f}, {(weight.abs().mean(dim=1).median()):.4f}]")
    print(f"  异常通道范围: 第10行 max={weight[10].abs().max():.2f}, 第50行 max={weight[50].abs().max():.2f}")
    
    # 逐张量量化
    pt_q = SymmetricQuantizer(n_bits=8)
    pt_quantized = pt_q.quantize(weight)
    pt_reconstructed = pt_q.dequantize(pt_quantized)
    analyze_quantization_error(weight, pt_reconstructed, "逐张量对称量化 (INT8)")
    
    # 逐通道量化
    pc_q = PerChannelQuantizer(n_bits=8, axis=0)
    pc_quantized = pc_q.quantize(weight)
    pc_reconstructed = pc_q.dequantize(pc_quantized)
    analyze_quantization_error(weight, pc_reconstructed, "逐通道对称量化 (INT8)")
    
    # 对比异常通道的误差
    print(f"\n  📌 异常通道（第 10 行）的量化误差对比:")
    for ch_idx in [10, 50, 90]:
        err_pt = (weight[ch_idx] - pt_reconstructed[ch_idx]).abs().mean()
        err_pc = (weight[ch_idx] - pc_reconstructed[ch_idx]).abs().mean()
        improvement = (err_pt - err_pc) / err_pt * 100
        print(f"    通道 {ch_idx:3d}: 逐张量 MAE={err_pt:.6f}, 逐通道 MAE={err_pc:.6f} (改善 {improvement:.1f}%)")
    
    print(f"\n  💡 逐通道量化为每个通道量身定制 scale，")
    print("     特别在通道间方差差异大时，优势明显")


# ============================================================
# 第四部分：模拟模型量化实验
# ============================================================


class SimpleNet(nn.Module):
    """
    简单的两层 MLP，用于演示模型级量化
    """
    def __init__(self, dim=64, num_classes=5):
        super().__init__()
        self.fc1 = nn.Linear(dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def quantize_model_weights(model: nn.Module, n_bits: int) -> dict:
    """
    对模型的所有 Linear 层进行对称量化
    
    Returns:
        dict: 每层的量化误差统计
    """
    results = {}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight = module.weight.data
            
            # 逐通道量化（业界标准做法）
            quantizer = PerChannelQuantizer(n_bits=n_bits, axis=0)
            quantized = quantizer.quantize(weight)
            reconstructed = quantizer.dequantize(quantized)
            
            # 计算误差
            error = (weight - reconstructed).abs()
            results[name] = {
                'mae': error.mean().item(),
                'max_error': error.max().item(),
                'cosine_sim': torch.nn.functional.cosine_similarity(
                    weight.flatten().unsqueeze(0),
                    reconstructed.flatten().unsqueeze(0)
                ).item(),
                'original_size_kb': weight.numel() * 4 / 1024,
                'quantized_size_kb': weight.numel() * n_bits / 8 / 1024,
            }
    
    return results


def demo_model_quantization():
    """
    演示：完整模型的量化实验
    """
    print("\n" + "="*70)
    print("  🔬 实验 4：模型级量化实验（两层 MLP）")
    print("="*70)
    
    torch.manual_seed(42)
    
    dim = 64
    num_classes = 5
    model = SimpleNet(dim=dim, num_classes=num_classes)
    
    # 用随机数据测试
    x_test = torch.randn(20, dim)
    
    # FP32 原始预测
    with torch.no_grad():
        original_output = model(x_test)
        original_pred = original_output.argmax(dim=1)
    
    print(f"\n  模型结构: Linear({dim},128) → ReLU → Linear(128,64) → ReLU → Linear(64,{num_classes})")
    print(f"  测试样本: {x_test.shape[0]} 个")
    
    for n_bits, label in [(8, "INT8"), (4, "INT4"), (2, "INT2")]:
        results = quantize_model_weights(model, n_bits)
        
        # 用量化权重替换原始权重，跑推理
        model_quantized = SimpleNet(dim=dim, num_classes=num_classes)
        model_quantized.load_state_dict(model.state_dict())
        
        for name, module in model_quantized.named_modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                quantizer = PerChannelQuantizer(n_bits=n_bits, axis=0)
                quantized = quantizer.quantize(weight)
                module.weight.data = quantizer.dequantize(quantized)
        
        with torch.no_grad():
            quantized_output = model_quantized(x_test)
            quantized_pred = quantized_output.argmax(dim=1)
        
        # 对比预测结果
        pred_match = (original_pred == quantized_pred).sum().item()
        output_diff = (original_output - quantized_output).abs().mean().item()
        
        total_orig_kb = sum(r['original_size_kb'] for r in results.values())
        total_quant_kb = sum(r['quantized_size_kb'] for r in results.values())
        
        print(f"\n  {'─'*50}")
        print(f"  📌 {label} 逐通道对称量化")
        print(f"  {'─'*50}")
        print(f"  权重存储: {total_orig_kb:.1f} KB → {total_quant_kb:.1f} KB ({total_orig_kb/total_quant_kb:.1f}x 压缩)")
        
        for layer_name, r in results.items():
            print(f"    {layer_name:10s}: MAE={r['mae']:.6f}, MaxErr={r['max_error']:.6f}, CosSim={r['cosine_sim']:.6f}")
        
        print(f"  输出平均差异: {output_diff:.6f}")
        print(f"  预测一致率: {pred_match}/{x_test.shape[0]} ({pred_match/x_test.shape[0]*100:.0f}%)")
    
    print(f"\n  💡 INT8 对预测几乎没有影响，INT4 开始有微小变化，INT2 误差显著")


# ============================================================
# 第五部分：伪量化（QAT 的核心操作）
# ============================================================


def fake_quantize(tensor: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    伪量化：量化后立即反量化
    
    用于 QAT（量化感知训练）中模拟量化噪声
    前向传播时带噪声，反向传播时梯度直通（STE）
    """
    qmax = 2 ** (n_bits - 1) - 1
    scale = tensor.abs().max() / qmax if tensor.abs().max() > 0 else torch.tensor(1.0)
    
    # 量化 + 反量化
    quantized = torch.clamp(torch.round(tensor / scale), -qmax, qmax)
    return quantized * scale


def demo_fake_quantization():
    """
    演示：伪量化引入的噪声
    """
    print("\n" + "="*70)
    print("  🔬 实验 5：伪量化（Fake Quantization）噪声可视化")
    print("="*70)
    
    torch.manual_seed(42)
    
    weight = torch.randn(64) * 0.3
    
    print(f"\n  原始权重（前 10 个值）:")
    for i in range(10):
        print(f"    w[{i}] = {weight[i]:+.6f}")
    
    for n_bits in [8, 4, 2]:
        fq_weight = fake_quantize(weight, n_bits)
        noise = (weight - fq_weight).abs()
        
        print(f"\n  {'─'*50}")
        print(f"  📌 INT{n_bits} 伪量化噪声（前 10 个值）:")
        print(f"  {'─'*50}")
        for i in range(10):
            print(f"    原始: {weight[i]:+.6f}  →  伪量化: {fq_weight[i]:+.6f}  →  噪声: {noise[i]:.6f}")
        
        print(f"  平均噪声: {noise.mean():.6f}")
        print(f"  最大噪声: {noise.max():.6f}")
    
    print(f"\n  💡 伪量化给每个权重加了微小的随机噪声")
    print("     QAT 训练时模型会'习惯'这些噪声，真正量化时就不会措手不及")


# ============================================================
# 第六部分：量化后模型大小估算
# ============================================================


def estimate_model_size():
    """
    估算不同量化方案下 LLM 的模型大小
    """
    print("\n" + "="*70)
    print("  🔬 实验 6：LLM 模型大小估算")
    print("="*70)
    
    models = {
        "LLaMA-2-7B": 7_000_000_000,
        "LLaMA-2-13B": 13_000_000_000,
        "Qwen-72B": 72_000_000_000,
    }
    
    formats = [
        ("FP32", 4),      # 4 bytes per param
        ("FP16/BF16", 2), # 2 bytes per param
        ("INT8", 1),      # 1 byte per param
        ("INT4", 0.5),    # 0.5 bytes per param
        ("INT3", 0.375),  # 0.375 bytes per param
        ("INT2", 0.25),   # 0.25 bytes per param
    ]
    
    print(f"\n  {'模型':<18} {'参数量':>12}", end="")
    for fmt_name, _ in formats:
        print(f" {fmt_name:>10}", end="")
    print()
    print(f"  {'─'*90}")
    
    for model_name, n_params in models.items():
        print(f"  {model_name:<18} {n_params/1e9:>10.1f}B", end="")
        for fmt_name, bytes_per_param in formats:
            size_gb = n_params * bytes_per_param / (1024 ** 3)
            print(f" {size_gb:>9.1f}GB", end="")
        print()
    
    print(f"\n  💡 关键数据点:")
    print(f"     LLaMA-2-7B INT4 = {7e9 * 0.5 / (1024**3):.1f}GB → 一张 RTX 3060 12GB 就能跑推理")
    print(f"     Qwen-72B  INT4 = {72e9 * 0.5 / (1024**3):.1f}GB → 需要至少 2× A100 80GB 或 4× RTX 4090")


# ============================================================
# 主函数：运行所有实验
# ============================================================


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Day 22: 量化基础 — 从 FP16 到 INT4 的数学之旅                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # 实验 1：对称 vs 非对称量化
    demo_symmetric_vs_asymmetric()
    
    # 实验 2：不同精度等级
    demo_different_precisions()
    
    # 实验 3：逐张量 vs 逐通道
    demo_per_tensor_vs_per_channel()
    
    # 实验 4：模型级量化
    demo_model_quantization()
    
    # 实验 5：伪量化噪声
    demo_fake_quantization()
    
    # 实验 6：模型大小估算
    estimate_model_size()
    
    print("\n" + "="*70)
    print("  ✅ 所有实验完成！")
    print("="*70)
    print("""
  📝 今日要点回顾：
  
  1. 量化的本质是线性映射：scale 和 zero_point 是关键参数
  2. 对称量化适合权重，非对称量化适合激活值
  3. 逐通道量化比逐张量量化精度更高
  4. INT8 几乎无损，INT4 微小损失，INT2 通常不可用
  5. PTQ 不需要训练，QAT 精度更高
  6. 伪量化（量化后立即反量化）是 QAT 的核心技巧
  
  明天我们将学习更高级的量化方法：GPTQ、AWQ、GGUF
  它们会"聪明地"保护重要权重，进一步提升 INT4 的精度。
    """)


if __name__ == "__main__":
    main()
