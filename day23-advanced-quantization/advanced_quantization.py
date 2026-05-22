#!/usr/bin/env python3
"""
Day 23: 量化进阶 — AWQ、GGUF、BitsAndBytes，聪明量化的艺术
================================================================

今天我们动手实现：
1. AWQ 核心算法：激活感知 → 找重要权重 → 缩放保护 → 量化
2. 对比实验：线性量化 vs AWQ 风格量化 vs 逐通道量化
3. 模拟混合精度量化：不同层用不同精度
4. GGUF K-quant 模型大小估算
5. NF4（正态分布优化量化）模拟

运行方式：python3 advanced_quantization.py
"""

import torch
import torch.nn as nn
import numpy as np
import math

# ============================================================
# 第一部分：基础量化工具（复用 Day 22 的核心逻辑）
# ============================================================


def symmetric_quantize(tensor: torch.Tensor, n_bits: int = 4):
    """
    对称量化（逐张量版本）
    
    Args:
        tensor: 待量化的浮点张量
        n_bits: 量化位数
    
    Returns:
        quantized: 量化后的整数张量
        scale: 缩放因子
    """
    qmax = 2 ** (n_bits - 1) - 1
    max_abs = tensor.abs().max()
    scale = max_abs / qmax if max_abs > 0 else torch.tensor(1.0)
    
    quantized = torch.clamp(
        torch.round(tensor / scale),
        -qmax, qmax
    ).to(torch.int32)
    
    return quantized, scale


def symmetric_dequantize(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    对称反量化：Q × scale
    """
    return quantized.float() * scale


def per_channel_quantize(weight: torch.Tensor, n_bits: int = 4):
    """
    逐通道对称量化（每行独立的 scale）
    """
    qmax = 2 ** (n_bits - 1) - 1
    # 沿 axis=1 找每行的最大绝对值
    max_abs = weight.abs().max(dim=1, keepdim=True)[0]
    scales = max_abs / qmax
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    
    quantized = torch.clamp(
        torch.round(weight / scales),
        -qmax, qmax
    ).to(torch.int32)
    
    return quantized, scales


def per_channel_dequantize(quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """
    逐通道反量化
    """
    return quantized.float() * scales


# ============================================================
# 第二部分：AWQ 核心算法复现
# ============================================================


def compute_activation_importance(weight: torch.Tensor, activation: torch.Tensor) -> torch.Tensor:
    """
    计算 AWQ 的"权重重要性"：|W| × |Activation|
    
    核心洞察：一个权重重不重要，不只看它自身的值，
    还要看它对应的激活值——两者都大的权重才是"关键权重"
    
    Args:
        weight: 权重矩阵 [out_features, in_features]
        activation: 对应的输入激活 [batch_size, in_features]
    
    Returns:
        importance: 每个通道的重要性分数 [out_features]
    """
    # 激活值的绝对值均值（沿 batch 维度）
    # 表示每个输入通道"通常有多大"
    act_importance = activation.abs().mean(dim=0)  # [in_features]
    
    # 权重的绝对值均值（沿输出通道维度，即每列）
    # 表示每个输入通道对应的权重"通常有多大"
    w_importance = weight.abs().mean(dim=0)  # [in_features]
    
    # AWQ 的重要性 = 权重大小 × 激活大小
    # 这是逐"输入通道"（列方向）的重要性
    channel_importance = w_importance * act_importance  # [in_features]
    
    return channel_importance


def awq_search_scale(weight: torch.Tensor, importance: torch.Tensor, n_bits: int = 4, 
                      alpha: float = 0.5, n_grid: int = 20, max_scale: float = 2.0):
    """
    AWQ 核心：搜索每个通道的最优缩放因子
    
    原理：
    - 对重要通道乘以 s > 1（放大），使量化后的相对误差减小
    - 对不重要通道不缩放或缩小，避免浪费量化级别
    - 在 s 的候选值中搜索使量化误差最小的组合
    
    Args:
        weight: 权重矩阵 [out_features, in_features]
        importance: 每个通道的重要性分数 [in_features]
        n_bits: 量化位数
        alpha: 重要通道的筛选比例（top alpha%）
        n_grid: 搜索网格大小
        max_scale: 最大缩放因子
    
    Returns:
        best_scales: 最优缩放因子 [in_features]
    """
    qmax = 2 ** (n_bits - 1) - 1
    n_channels = weight.shape[1]
    
    # 找到重要通道的阈值
    threshold = torch.quantile(importance, 1 - alpha)
    important_mask = importance >= threshold  # [in_features]，哪些通道是"重要的"
    
    # 初始化缩放因子为 1（不缩放）
    best_scales = torch.ones(n_channels)
    
    # 分组处理：每组 group_size 个通道共享一个缩放因子
    group_size = 64
    n_groups = (n_channels + group_size - 1) // group_size
    
    for g in range(n_groups):
        start = g * group_size
        end = min(start + group_size, n_channels)
        
        # 这组的权重切片
        w_group = weight[:, start:end]
        
        # 这组中是否有重要通道
        group_important = important_mask[start:end]
        
        if not group_important.any():
            # 没有重要通道，保持 scale = 1
            continue
        
        # 在网格中搜索最优缩放因子
        best_error = float('inf')
        best_s = 1.0
        
        for s_val in torch.linspace(0.5, max_scale, n_grid):
            # 构造缩放向量：重要通道乘以 s，不重要通道保持 1
            s_vector = torch.ones(end - start)
            s_vector[important_mask[start:end]] = s_val
            
            # 缩放后量化
            w_scaled = w_group * s_vector.unsqueeze(0)
            scale = w_scaled.abs().max() / qmax if w_scaled.abs().max() > 0 else 1.0
            w_quantized = torch.clamp(torch.round(w_scaled / scale), -qmax, qmax)
            w_dequantized = w_quantized * scale
            
            # 反缩放回来
            w_reconstructed = w_dequantized / s_vector.unsqueeze(0)
            
            # 计算误差（加权：重要通道的误差权重更大）
            error = (w_group - w_reconstructed).abs()
            weight_importance = importance[start:end].unsqueeze(0)
            weighted_error = (error * weight_importance).sum()
            
            if weighted_error < best_error:
                best_error = weighted_error
                best_s = s_val.item()
        
        # 应用最优缩放
        best_scales[start:end][important_mask[start:end]] = best_s
    
    return best_scales


def awq_quantize(weight: torch.Tensor, activation: torch.Tensor, n_bits: int = 4):
    """
    完整的 AWQ 量化流程
    
    1. 计算通道重要性（基于激活值）
    2. 搜索最优缩放因子
    3. 缩放 → 量化 → 反量化 → 反缩放
    
    Args:
        weight: 权重矩阵 [out_features, in_features]
        activation: 校准激活值 [batch_size, in_features]
        n_bits: 量化位数
    
    Returns:
        reconstructed: 量化后反量化的权重
        scales: AWQ 缩放因子
    """
    # 第 1 步：计算重要性
    importance = compute_activation_importance(weight, activation)
    
    # 第 2 步：搜索最优缩放
    awq_scales = awq_search_scale(weight, importance, n_bits=n_bits)
    
    # 第 3 步：缩放权重
    w_scaled = weight * awq_scales.unsqueeze(0)
    
    # 第 4 步：逐通道量化（缩放后的权重）
    quantized, quant_scales = per_channel_quantize(w_scaled, n_bits=n_bits)
    
    # 第 5 步：反量化
    dequantized = per_channel_dequantize(quantized, quant_scales)
    
    # 第 6 步：反缩放（恢复原始尺度）
    reconstructed = dequantized / awq_scales.unsqueeze(0)
    
    return reconstructed, awq_scales


# ============================================================
# 第三部分：NF4（NormalFloat4）模拟
# ============================================================


def compute_nf4_quantiles(n_levels: int = 16):
    """
    计算 NF4 的量化级别
    
    NF4 的核心思想：量化级别不是均匀分布的，
    而是按照标准正态分布的分位数来分布。
    
    这样可以更好地表示集中在 0 附近的权重值。
    
    类比：
    均匀量化 = 一把刻度等距的尺子
    NF4      = 一把在 0 附近刻度更密的尺子（因为权重大多在 0 附近）
    """
    # 在 [-1, 1] 范围内，按正态分布的分位数取值
    # 这样在 0 附近更密集，在两端更稀疏
    quantiles = torch.linspace(0, 1, n_levels + 1)[:-1]  # 不包含 1.0
    
    # 使用正态分布的逆 CDF（ppf）来计算分位数值
    # 因为权重近似正态分布，所以用正态分位数
    from scipy.stats import norm
    nf4_levels = torch.tensor([norm.ppf(q * 0.5 + 0.5) for q in quantiles.numpy()], dtype=torch.float32)
    
    # 归一化到 [-1, 1]
    nf4_levels = nf4_levels / nf4_levels.abs().max()
    
    return nf4_levels


def nf4_quantize(tensor: torch.Tensor, n_levels: int = 16):
    """
    NF4 量化：将张量量化到 NF4 级别
    
    与均匀量化的区别：
    - 均匀量化：量化级别等间距 → 在权重密集区域精度浪费
    - NF4：量化级别按正态分位数 → 在权重密集区域（0 附近）精度更高
    """
    # 归一化到 [-1, 1]
    max_abs = tensor.abs().max()
    normalized = tensor / max_abs if max_abs > 0 else tensor
    
    # 获取 NF4 级别
    try:
        levels = compute_nf4_quantiles(n_levels)
    except ImportError:
        # 如果没有 scipy，退回到均匀量化
        levels = torch.linspace(-1, 1, n_levels)
    
    # 找每个元素最近的量化级别
    # 展开 normalized 为 [n_elements, 1]，levels 为 [1, n_levels]
    distances = (normalized.unsqueeze(-1) - levels.unsqueeze(0)).abs()
    indices = distances.argmin(dim=-1)
    
    # 量化后的值
    quantized = levels[indices]
    
    # 反归一化
    reconstructed = quantized * max_abs
    
    return reconstructed, indices


# ============================================================
# 第四部分：实验函数
# ============================================================


def demo_awq_core():
    """
    实验 1：AWQ 核心思想演示
    对比：线性量化 vs 逐通道量化 vs AWQ 风格量化
    """
    print("\n" + "=" * 70)
    print("  🔬 实验 1：AWQ 核心算法 — 激活感知量化")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 模拟一个权重矩阵和对应的激活值
    out_features, in_features = 128, 256
    weight = torch.randn(out_features, in_features) * 0.3
    
    # 模拟激活值：大部分通道的激活很小，但有几个通道的激活特别大
    activation = torch.randn(32, in_features) * 0.1
    # 人为制造几个"高激活"通道
    activation[:, 50] = activation[:, 50] * 20   # 通道 50 激活很大
    activation[:, 120] = activation[:, 120] * 15  # 通道 120 激活很大
    activation[:, 200] = activation[:, 200] * 25  # 通道 200 激活最大
    
    # 对应地，给这些通道设置较大的权重
    weight[:, 50] = weight[:, 50] * 3
    weight[:, 120] = weight[:, 120] * 2
    weight[:, 200] = weight[:, 200] * 4
    
    print(f"\n  权重矩阵: {weight.shape}")
    print(f"  激活矩阵: {activation.shape}")
    
    # 计算重要性
    importance = compute_activation_importance(weight, activation)
    top_channels = importance.topk(5)
    print(f"\n  📌 最重要通道 (importance = |W| × |Act|):")
    for idx, val in zip(top_channels.indices, top_channels.values):
        print(f"    通道 {idx.item():3d}: 重要性 = {val.item():.4f}")
    
    n_bits = 4
    
    # 方案 A：朴素对称量化
    q_a, scale_a = symmetric_quantize(weight, n_bits)
    r_a = symmetric_dequantize(q_a, scale_a)
    error_a = (weight - r_a).abs().mean()
    
    # 方案 B：逐通道量化
    q_b, scales_b = per_channel_quantize(weight, n_bits)
    r_b = per_channel_dequantize(q_b, scales_b)
    error_b = (weight - r_b).abs().mean()
    
    # 方案 C：AWQ 风格量化
    r_c, awq_scales = awq_quantize(weight, activation, n_bits)
    error_c = (weight - r_c).abs().mean()
    
    print(f"\n  {'─'*50}")
    print(f"  📊 量化误差对比 (INT4):")
    print(f"  {'─'*50}")
    print(f"  A. 朴素对称量化 MAE:    {error_a:.6f}")
    print(f"  B. 逐通道量化   MAE:    {error_b:.6f}  (改善 {(error_a - error_b)/error_a*100:.1f}%)")
    print(f"  C. AWQ 风格量化 MAE:    {error_c:.6f}  (改善 {(error_a - error_c)/error_c*100:.1f}%)")
    
    # 检查重要通道上的误差
    print(f"\n  📌 重要通道（通道 50/120/200）的误差对比:")
    for ch in [50, 120, 200]:
        e_naive = (weight[:, ch] - r_a[:, ch]).abs().mean()
        e_perch = (weight[:, ch] - r_b[:, ch]).abs().mean()
        e_awq = (weight[:, ch] - r_c[:, ch]).abs().mean()
        print(f"    通道 {ch:3d}: 朴素={e_naive:.6f}, 逐通道={e_perch:.6f}, AWQ={e_awq:.6f}")
    
    # AWQ 缩放因子
    scaled_channels = (awq_scales != 1.0).sum().item()
    print(f"\n  AWQ 缩放因子统计:")
    print(f"    被缩放的通道数: {scaled_channels}/{in_features}")
    print(f"    最大缩放因子: {awq_scales.max():.3f}")
    print(f"    最小缩放因子: {awq_scales.min():.3f}")
    
    print(f"\n  💡 AWQ 通过识别重要通道并缩放保护，在关键位置精度更好")


def demo_nf4_vs_uniform():
    """
    实验 2：NF4（正态分位数量化）vs 均匀量化
    """
    print("\n" + "=" * 70)
    print("  🔬 实验 2：NF4 vs 均匀量化 — 正态分布权重上的对比")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 模拟正态分布的权重
    weight = torch.randn(2048) * 0.3  # N(0, 0.3)
    
    print(f"\n  权重分布: N(0, 0.3), {weight.numel()} 个参数")
    print(f"  范围: [{weight.min():.4f}, {weight.max():.4f}]")
    
    # 均匀 INT4 量化
    q_uniform, scale_u = symmetric_quantize(weight, n_bits=4)
    r_uniform = symmetric_dequantize(q_uniform, scale_u)
    error_uniform = (weight - r_uniform).abs()
    
    # NF4 量化
    r_nf4, indices_nf4 = nf4_quantize(weight, n_levels=16)
    error_nf4 = (weight - r_nf4).abs()
    
    print(f"\n  {'─'*50}")
    print(f"  📊 量化误差对比:")
    print(f"  {'─'*50}")
    print(f"  均匀 INT4 量化:")
    print(f"    平均误差 (MAE): {error_uniform.mean():.6f}")
    print(f"    最大误差:       {error_uniform.max():.6f}")
    print(f"    中位误差:       {error_uniform.median():.6f}")
    
    print(f"\n  NF4 量化（正态分位数）:")
    print(f"    平均误差 (MAE): {error_nf4.mean():.6f}")
    print(f"    最大误差:       {error_nf4.max():.6f}")
    print(f"    中位误差:       {error_nf4.median():.6f}")
    
    improvement = (error_uniform.mean() - error_nf4.mean()) / error_uniform.mean() * 100
    print(f"\n  NF4 相比均匀量化 MAE 改善: {improvement:.1f}%")
    
    # 量化级别分布对比
    print(f"\n  📌 量化级别分布对比:")
    uniform_levels = torch.linspace(-scale_u * 7, scale_u * 7, 15)
    print(f"  均匀量化级别（间距 {scale_u.item():.4f}）:")
    print(f"    {uniform_levels[:8].tolist()}")
    print(f"    {uniform_levels[8:].tolist()}")
    
    try:
        nf4_levels = compute_nf4_quantiles(16) * weight.abs().max()
        print(f"  NF4 量化级别（间距不均匀，中间密两头稀）:")
        print(f"    {[f'{v:.4f}' for v in nf4_levels[:8].tolist()]}")
        print(f"    {[f'{v:.4f}' for v in nf4_levels[8:].tolist()]}")
    except:
        print(f"  (NF4 级别需要 scipy)")
    
    # 小值区间的精度对比（这是 NF4 的优势区域）
    small_mask = weight.abs() < 0.1
    if small_mask.sum() > 0:
        err_uniform_small = error_uniform[small_mask].mean()
        err_nf4_small = error_nf4[small_mask].mean()
        print(f"\n  📌 小权重区间 (|w| < 0.1, 共 {small_mask.sum().item()} 个):")
        print(f"    均匀量化 MAE: {err_uniform_small:.6f}")
        print(f"    NF4 MAE:      {err_nf4_small:.6f} (改善 {(err_uniform_small-err_nf4_small)/err_uniform_small*100:.1f}%)")
    
    print(f"\n  💡 NF4 在正态分布权重上的优势：在 0 附近（密集区）精度更高")


def demo_mixed_precision():
    """
    实验 3：模拟混合精度量化
    不同层使用不同精度，观察综合效果
    """
    print("\n" + "=" * 70)
    print("  🔬 实验 3：混合精度量化 — 模拟 Transformer 各层")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 模拟 Transformer 各层的权重
    layers = {
        "embedding":    torch.randn(1000, 512) * 0.1,      # Embedding 层
        "attn_q_proj":  torch.randn(512, 512) * 0.08,      # Q 投影
        "attn_k_proj":  torch.randn(512, 512) * 0.08,      # K 投影
        "attn_v_proj":  torch.randn(512, 512) * 0.12,      # V 投影
        "attn_out":     torch.randn(512, 512) * 0.10,      # 输出投影
        "ffn_gate":     torch.randn(512, 2048) * 0.15,     # FFN gate（参数最多）
        "ffn_up":       torch.randn(512, 2048) * 0.15,     # FFN up
        "ffn_down":     torch.randn(2048, 512) * 0.13,     # FFN down
        "output_head":  torch.randn(512, 1000) * 0.09,     # 输出头
    }
    
    # 模拟激活值（用于 AWQ 风格的敏感度分析）
    activations = {}
    for name, w in layers.items():
        batch = 16
        in_dim = w.shape[1]
        activations[name] = torch.randn(batch, in_dim) * 0.2
    
    print(f"\n  模拟 Transformer 层结构:")
    for name, w in layers.items():
        print(f"    {name:15s}: {w.shape} = {w.numel():,} 参数")
    
    total_params = sum(w.numel() for w in layers.values())
    
    # 策略 1：统一 INT4
    print(f"\n  {'─'*60}")
    print(f"  📌 策略 1：统一 INT4 量化")
    print(f"  {'─'*60}")
    
    total_error_int4 = 0
    total_size_int4 = 0
    for name, w in layers.items():
        q, s = symmetric_quantize(w, n_bits=4)
        r = symmetric_dequantize(q, s)
        mae = (w - r).abs().mean().item()
        size = w.numel() * 4 / 8  # 4 bit per param
        total_error_int4 += mae * w.numel()
        total_size_int4 += size
        print(f"    {name:15s}: MAE={mae:.6f}, 大小={size/1024:.1f}KB")
    
    avg_error_int4 = total_error_int4 / total_params
    
    # 策略 2：混合精度（基于规则的分层策略）
    print(f"\n  {'─'*60}")
    print(f"  📌 策略 2：混合精度量化")
    print(f"  {'─'*60}")
    
    precision_plan = {
        "embedding":    8,   # Embedding 用 INT8（保护精度）
        "attn_q_proj":  8,   # Q/K 用 INT8（注意力敏感）
        "attn_k_proj":  8,
        "attn_v_proj":  6,   # V/O 可以稍低
        "attn_out":     6,
        "ffn_gate":     4,   # FFN 用 INT4（冗余大，不怕压缩）
        "ffn_up":       4,
        "ffn_down":     4,
        "output_head":  6,   # 输出头折中
    }
    
    total_error_mixed = 0
    total_size_mixed = 0
    for name, w in layers.items():
        bits = precision_plan[name]
        q, s = symmetric_quantize(w, n_bits=bits)
        r = symmetric_dequantize(q, s)
        mae = (w - r).abs().mean().item()
        size = w.numel() * bits / 8
        total_error_mixed += mae * w.numel()
        total_size_mixed += size
        print(f"    {name:15s}: INT{bits} → MAE={mae:.6f}, 大小={size/1024:.1f}KB")
    
    avg_error_mixed = total_error_mixed / total_params
    
    # 对比
    print(f"\n  {'─'*60}")
    print(f"  📊 策略对比:")
    print(f"  {'─'*60}")
    print(f"  统一 INT4:")
    print(f"    总大小: {total_size_int4/1024/1024:.2f} MB")
    print(f"    加权平均误差: {avg_error_int4:.6f}")
    print(f"  混合精度:")
    print(f"    总大小: {total_size_mixed/1024/1024:.2f} MB")
    print(f"    加权平均误差: {avg_error_mixed:.6f}")
    print(f"    精度改善: {(avg_error_int4 - avg_error_mixed)/avg_error_int4*100:.1f}%")
    print(f"    体积增加: {(total_size_mixed - total_size_int4)/total_size_int4*100:.1f}%")
    
    print(f"\n  💡 混合精度只比纯 INT4 大一点，但精度显著提升")
    print(f"     这就是 GGUF K-quant（如 Q4_K_M）的设计思想")


def demo_gguf_size_estimation():
    """
    实验 4：GGUF K-quant 模型大小估算
    """
    print("\n" + "=" * 70)
    print("  🔬 实验 4：GGUF K-quant 模型大小估算")
    print("=" * 70)
    
    models = {
        "LLaMA-2-7B":  7_000_000_000,
        "Qwen2.5-14B": 14_000_000_000,
        "LLaMA-3-70B": 70_000_000_000,
    }
    
    quant_types = [
        ("FP16",      2.0),
        ("Q8_0",      1.0),
        ("Q6_K",      0.75),
        ("Q5_K_M",    0.69),
        ("Q4_K_M",    0.58),
        ("Q4_0",      0.50),
        ("Q3_K_M",    0.43),
        ("Q2_K",      0.33),
    ]
    
    print(f"\n  {'模型':<16} {'参数量':>10}", end="")
    for name, _ in quant_types:
        print(f" {name:>8}", end="")
    print()
    print(f"  {'─'*82}")
    
    for model_name, n_params in models.items():
        print(f"  {model_name:<16} {n_params/1e9:>9.1f}B", end="")
        for qt_name, bytes_per_param in quant_types:
            size_gb = n_params * bytes_per_param / (1024 ** 3)
            print(f" {size_gb:>7.1f}GB", end="")
        print()
    
    print(f"\n  📌 关键参考:")
    print(f"    RTX 3060 12GB → 能跑 Q4_K_M 的 7B 模型（4.0GB）✅")
    print(f"    RTX 4090 24GB → 能跑 Q4_K_M 的 14B 模型（7.8GB）✅")
    print(f"    MacBook 16GB  → 能跑 Q4_K_M 的 7B 模型（CPU + mmap）✅")
    print(f"    MacBook 8GB   → 勉强跑 Q3_K_M 的 7B 模型（3.0GB）⚠️")
    print(f"    树莓派 8GB    → Q2_K 的 7B 模型（2.3GB），速度很慢 🐌")
    
    print(f"\n  💡 选择建议:")
    print(f"    Q5_K_M = 精度/体积最佳平衡（推荐）")
    print(f"    Q4_K_M = 最受欢迎的折中方案")
    print(f"    Q3_K_M = 内存紧张时的选择")
    print(f"    Q2_K   = 极限压缩，但精度明显下降")


def demo_bnb_int8_simulation():
    """
    实验 5：模拟 BitsAndBytes LLM.int8() 的混合精度分解
    """
    print("\n" + "=" * 70)
    print("  🔬 实验 5：LLM.int8() 混合精度分解模拟")
    print("=" * 70)
    
    torch.manual_seed(42)
    
    # 模拟一个权重矩阵和激活向量
    weight = torch.randn(512, 512) * 0.3
    activation = torch.randn(1, 512) * 0.5
    
    # 人为在激活中添加几个极端值
    activation[0, 10] = 8.0
    activation[0, 100] = -6.5
    activation[0, 250] = 7.2
    activation[0, 400] = -9.0
    
    print(f"\n  权重矩阵: {weight.shape}")
    print(f"  激活向量: {activation.shape}")
    print(f"  激活值范围: [{activation.min():.2f}, {activation.max():.2f}]")
    
    # FP16 基准（精确结果）
    output_fp16 = activation @ weight.T
    
    # 纯 INT8 量化（无混合精度）
    # 量化权重
    w_max = weight.abs().max()
    w_scale = w_max / 127
    w_int8 = torch.clamp(torch.round(weight / w_scale), -128, 127).to(torch.int32)
    
    # 量化激活
    a_max = activation.abs().max()
    a_scale = a_max / 127
    a_int8 = torch.clamp(torch.round(activation / a_scale), -128, 127).to(torch.int32)
    
    # INT8 矩阵乘法 → 反量化
    output_int8_pure = (a_int8 @ w_int8.T).float() * a_scale * w_scale
    
    # LLM.int8() 混合精度分解
    threshold = 6.0  # 极端值阈值（通常用 6.0σ）
    
    # 找到激活中的极端值
    outlier_mask = activation.abs() > threshold
    normal_mask = ~outlier_mask
    
    n_outliers = outlier_mask.sum().item()
    n_normal = normal_mask.sum().item()
    
    print(f"\n  极端值阈值: {threshold}")
    print(f"  极端值数量: {n_outliers} ({n_outliers/activation.numel()*100:.1f}%)")
    print(f"  正常值数量: {n_normal} ({n_normal/activation.numel()*100:.1f}%)")
    
    # 正常值部分 → INT8 计算
    a_normal = activation.clone()
    a_normal[outlier_mask] = 0  # 把极端值清零
    a_normal_scale = a_normal.abs().max() / 127 if a_normal.abs().max() > 0 else 1.0
    a_normal_int8 = torch.clamp(torch.round(a_normal / a_normal_scale), -128, 127).to(torch.int32)
    
    output_normal = (a_normal_int8 @ w_int8.T).float() * a_normal_scale * w_scale
    
    # 极端值部分 → FP16 精确计算
    a_outlier = activation.clone()
    a_outlier[normal_mask] = 0  # 把正常值清零
    output_outlier = a_outlier @ weight.T  # FP16 精确计算
    
    # 合并结果
    output_mixed = output_normal + output_outlier
    
    # 对比误差
    error_int8 = (output_fp16 - output_int8_pure).abs().mean()
    error_mixed = (output_fp16 - output_mixed).abs().mean()
    
    print(f"\n  {'─'*50}")
    print(f"  📊 输出误差对比:")
    print(f"  {'─'*50}")
    print(f"  纯 INT8 量化误差:          {error_int8:.6f}")
    print(f"  LLM.int8() 混合精度误差:   {error_mixed:.6f}")
    print(f"  精度改善: {(error_int8 - error_mixed)/error_int8*100:.1f}%")
    
    print(f"\n  💡 LLM.int8() 的核心：只对极端值（{n_outliers}个）用高精度")
    print(f"     其余 {n_normal} 个值用 INT8 → 几乎不损失精度，但省了内存")


# ============================================================
# 主函数
# ============================================================


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  Day 23: 量化进阶 — AWQ、GGUF、BitsAndBytes                   ║")
    print("║           聪明量化的艺术                                       ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # 实验 1：AWQ 核心
    demo_awq_core()
    
    # 实验 2：NF4 vs 均匀量化
    demo_nf4_vs_uniform()
    
    # 实验 3：混合精度
    demo_mixed_precision()
    
    # 实验 4：GGUF 大小估算
    demo_gguf_size_estimation()
    
    # 实验 5：LLM.int8() 模拟
    demo_bnb_int8_simulation()
    
    print("\n" + "=" * 70)
    print("  ✅ 所有实验完成！")
    print("=" * 70)
    print("""
  📝 今日要点回顾：
  
  1. AWQ 用"激活感知"找到重要权重，通过缩放保护它们
  2. GPTQ 用 Hessian 信息做误差补偿（"边裁边补"）
  3. GGUF 的 K-quant 是混合精度：不同层用不同 bit 数
  4. NF4 按正态分位数分布量化级别，在 0 附近更精密
  5. BitsAndBytes 的 LLM.int8() 拆分极端值和正常值
  6. 选择量化方案要看场景：GPU/CPU、推理/训练、内存大小
  
  明天我们将学习 MoE（混合专家模型）——另一种"省资源"的架构。
    """)


if __name__ == "__main__":
    main()
