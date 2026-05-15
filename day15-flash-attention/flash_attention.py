#!/usr/bin/env python3
"""
Day 15: Flash Attention — 分块注意力计算的核心思想

本代码用纯 Python + PyTorch 模拟 Flash Attention 的分块计算流程：
1. 标准注意力实现（对照基线）
2. 分块注意力实现（模拟 Flash Attention 的核心算法）
3. 数值验证：确认两者输出完全一致
4. IO 量估算：对比标准 vs 分块的 HBM 读写
5. 显存占用对比
6. 速度对比：PyTorch SDPA（内置 Flash Attention）vs 手动实现

不依赖任何特殊硬件，纯 Python 就能理解核心思想。
"""

import torch
import torch.nn.functional as F
import math
import time
import sys

# ============================================================
# 1. 标准注意力实现 —— 教科书版本，作为对照
# ============================================================

def standard_attention(Q, K, V, mask=None):
    """
    标准的 Scaled Dot-Product Attention 实现。
    
    参数:
        Q: (B, n_heads, T_q, d_head) — Query 矩阵
        K: (B, n_heads, T_k, d_head) — Key 矩阵
        V: (B, n_heads, T_k, d_head) — Value 矩阵
        mask: 可选的注意力掩码
    返回:
        O: (B, n_heads, T_q, d_head) — 输出
    """
    d_head = Q.shape[-1]
    
    # 步骤 1: Q × K^T → 得到注意力分数矩阵 S (T_q × T_k)
    # 这里会产生 N×N 的中间矩阵，需要写入 HBM
    S = torch.matmul(Q, K.transpose(-2, -1))  # (B, n_heads, T_q, T_k)
    
    # 缩放
    S = S / math.sqrt(d_head)
    
    # 可选：应用掩码（如 causal mask）
    if mask is not None:
        S = S.masked_fill(mask == 0, float('-inf'))
    
    # 步骤 2: Softmax → 得到注意力权重 P (T_q × T_k)
    # 又一个 N×N 的中间矩阵，需要写入 HBM
    P = F.softmax(S, dim=-1)  # (B, n_heads, T_q, T_k)
    
    # 步骤 3: P × V → 得到输出 O
    O = torch.matmul(P, V)  # (B, n_heads, T_q, d_head)
    
    return O


# ============================================================
# 2. 分块注意力实现 —— 模拟 Flash Attention 的核心算法
# ============================================================

def flash_attention_simulated(Q, K, V, mask=None, block_size=64):
    """
    模拟 Flash Attention 的分块计算。
    
    核心思想：
    - 把 Q、K、V 切成小块（block）
    - 在每个小块上计算注意力，使用"在线 Softmax"逐块更新结果
    - 不需要存储完整的 N×N 中间矩阵 S 和 P
    
    数学上和 standard_attention 完全等价——输出逐元素相等。
    
    参数:
        Q: (B, n_heads, T_q, d_head)
        K: (B, n_heads, T_k, d_head)
        V: (B, n_heads, T_k, d_head)
        mask: 可选掩码（暂不支持分块掩码，仅演示无掩码情况）
        block_size: 分块大小（模拟 SRAM 的大小限制）
    返回:
        O: (B, n_heads, T_q, d_head) — 和 standard_attention 完全相同
    """
    B, n_heads, T_q, d_head = Q.shape
    T_k = K.shape[2]
    scale = 1.0 / math.sqrt(d_head)
    
    # 初始化输出和统计量
    # O: 累积输出（相当于最终结果的"半成品"）
    # l: 每行的 exp 之和（softmax 的分母）
    # m: 每行的最大值（用于数值稳定）
    O = torch.zeros(B, n_heads, T_q, d_head, dtype=Q.dtype, device=Q.device)
    l = torch.zeros(B, n_heads, T_q, 1, dtype=Q.dtype, device=Q.device)      # (B, H, T_q, 1)
    m = torch.full((B, n_heads, T_q, 1), float('-inf'), dtype=Q.dtype, device=Q.device)  # (B, H, T_q, 1)
    
    # Flash Attention 的核心：O 存储的是「未归一化的加权累积」
    # 即 O = Σ_j exp(S_j - m) * V_j（还没有除以 softmax 分母）
    # l 存储的是 Σ_j exp(S_j - m)（即 softmax 的分母部分）
    # 最终 O / l 才是真正的 softmax 加权输出
    
    # 把 Q 按行分块（外层循环 Q 的块）
    for q_start in range(0, T_q, block_size):
        q_end = min(q_start + block_size, T_q)
        Q_block = Q[:, :, q_start:q_end, :]  # (B, H, block_q, d_head)
        
        # 取出当前 Q 块对应的 O（未归一化累积）、l、m
        Oi = O[:, :, q_start:q_end, :]    # (B, H, block_q, d_head) — 未归一化累积
        li = l[:, :, q_start:q_end, :]    # (B, H, block_q, 1) — exp 之和
        mi = m[:, :, q_start:q_end, :]    # (B, H, block_q, 1) — 当前行最大值
        
        # 内层循环：遍历 K、V 的块
        for kv_start in range(0, T_k, block_size):
            kv_end = min(kv_start + block_size, T_k)
            K_block = K[:, :, kv_start:kv_end, :]  # (B, H, block_kv, d_head)
            V_block = V[:, :, kv_start:kv_end, :]  # (B, H, block_kv, d_head)
            
            # 步骤 1: 计算当前块的注意力分数 S_ij
            # 只有小块，放得进 SRAM！
            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) * scale
            # S_block: (B, H, block_q, block_kv)
            
            # 步骤 2: 计算当前块的行最大值 m_ij
            m_ij = S_block.max(dim=-1, keepdim=True).values  # (B, H, block_q, 1)
            
            # 步骤 3: 计算新的全局最大值
            m_new = torch.maximum(mi, m_ij)  # (B, H, block_q, 1)
            
            # 步骤 4: 计算当前块的 exp(P_ij)，注意用的是 m_ij 而不是 m_new
            # P_ij = exp(S_ij - m_ij) 是用当前块自己的 max 做的数值稳定
            P_block = torch.exp(S_block - m_ij)  # (B, H, block_q, block_kv)
            
            # 步骤 5: 在线 Softmax 更新（最关键的部分！）
            # 当 m 变化时，之前的累积值需要乘以校正因子
            # 校正因子 = exp(m_old - m_new)
            # 这是因为 Oi 里存储的是 exp(S - m_old) * V 的累积
            # 现在要变成 exp(S - m_new) * V，所以要乘 exp(m_old - m_new)
            alpha = torch.exp(mi - m_new)       # 旧累积的校正因子
            beta = torch.exp(m_ij - m_new)      # 新块的校正因子
            
            # 更新未归一化的输出累积：Oi = α * Oi + β * (P @ V)
            Oi = alpha * Oi + beta * torch.matmul(P_block, V_block)
            
            # 更新 exp 之和：li = α * li + β * Σ(P)
            li = alpha * li + beta * P_block.sum(dim=-1, keepdim=True)
            
            # 更新行最大值
            mi = m_new
        
        # 写回当前 Q 块的结果
        O[:, :, q_start:q_end, :] = Oi
        l[:, :, q_start:q_end, :] = li
        m[:, :, q_start:q_end, :] = mi
    
    # 最终归一化：O / l
    # 此时 l 是 softmax 分母的完整值
    O = O / l
    
    return O


# ============================================================
# 3. IO 量估算
# ============================================================

def estimate_io(T, d, n_heads, block_size=64):
    """
    估算标准注意力和 Flash Attention 的 HBM 读写量。
    
    参数:
        T: 序列长度
        d: d_head
        n_heads: 头数
        block_size: Flash Attention 的块大小
    """
    bytes_per_elem = 2  # float16
    
    print(f"\n{'='*60}")
    print(f"📊 IO 量估算 (T={T}, d_head={d}, n_heads={n_heads})")
    print(f"{'='*60}")
    
    # 标准注意力的 HBM 读写（每个头）
    io_std_read_Q = T * d                     # 读 Q
    io_std_read_K = T * d                     # 读 K
    io_std_write_S = T * T                    # 写 S（QK^T 结果）
    io_std_read_S = T * T                     # 读 S（做 softmax）
    io_std_write_P = T * T                    # 写 P（softmax 结果）
    io_std_read_P = T * T                     # 读 P（乘 V）
    io_std_read_V = T * d                     # 读 V
    io_std_write_O = T * d                    # 写 O
    
    io_std_total = (io_std_read_Q + io_std_read_K + io_std_write_S + 
                    io_std_read_S + io_std_write_P + io_std_read_P + 
                    io_std_read_V + io_std_write_O)
    
    print(f"\n标准注意力（每头）:")
    print(f"  读 Q:       {T*d:>10,} 元素")
    print(f"  读 K:       {T*d:>10,} 元素")
    print(f"  写 S (QK^T):{T*T:>10,} 元素 ← 巨大的中间矩阵")
    print(f"  读 S:       {T*T:>10,} 元素")
    print(f"  写 P (softmax):{T*T:>10,} 元素 ← 又一个巨大矩阵")
    print(f"  读 P:       {T*T:>10,} 元素")
    print(f"  读 V:       {T*d:>10,} 元素")
    print(f"  写 O:       {T*d:>10,} 元素")
    print(f"  ─────────────────────")
    print(f"  总 IO:      {io_std_total:>10,} 元素 × {n_heads} 头 = {io_std_total * n_heads:,}")
    
    # Flash Attention 的 HBM 读写（每个头）
    # 分块计算中，S 和 P 不写入 HBM
    n_blocks_q = math.ceil(T / block_size)
    n_blocks_kv = math.ceil(T / block_size)
    
    # 读写 Q、K、V 各一次，写 O 一次
    # Q 被分块读：T × d（总量一样，但分块读）
    # K 被分块读：n_blocks_q × T × d（每处理一个 Q 块，读一遍完整的 K）
    #   但实际上 K 的每个块只被读 n_blocks_q 次
    #   总量：n_blocks_q × T × d（和标准一样，只是读的顺序不同）
    # 简化估算：每个头读 Q + K + V + 写 O
    io_flash_total = 4 * T * d  # 读 Q + 读 K + 读 V + 写 O
    
    print(f"\nFlash Attention（每头，分块大小={block_size}）:")
    print(f"  读 Q:       {T*d:>10,} 元素（分块读）")
    print(f"  读 K:       {T*d:>10,} 元素（分块读）")
    print(f"  读 V:       {T*d:>10,} 元素（分块读）")
    print(f"  S, P:       {'不写入 HBM!':>10}")
    print(f"  写 O:       {T*d:>10,} 元素")
    print(f"  ─────────────────────")
    print(f"  总 IO:      {io_flash_total:>10,} 元素 × {n_heads} 头 = {io_flash_total * n_heads:,}")
    
    ratio = io_std_total / io_flash_total
    print(f"\n⚡ Flash Attention 的 IO 量是标准实现的 1/{ratio:.1f}")
    print(f"   （即 Flash Attention 的 IO 减少了 {ratio:.1f} 倍）")


# ============================================================
# 4. 显存占用对比
# ============================================================

def estimate_memory(T, d, n_heads, layers=32):
    """
    估算标准注意力和 Flash Attention 的显存占用。
    """
    bytes_per_elem = 2  # float16
    
    print(f"\n{'='*60}")
    print(f"💾 显存占用估算 (T={T}, d_head={d}, n_heads={n_heads}, layers={layers})")
    print(f"{'='*60}")
    
    # 标准注意力的中间矩阵（训练时需要保存用于反向传播）
    mem_S = T * T * n_heads * bytes_per_elem * layers  # S 矩阵
    mem_P = T * T * n_heads * bytes_per_elem * layers  # P 矩阵
    mem_std_total = mem_S + mem_P
    
    print(f"\n标准注意力（训练时）:")
    print(f"  S 矩阵 (QK^T): {mem_S / 1e9:.2f} GB")
    print(f"  P 矩阵 (softmax): {mem_P / 1e9:.2f} GB")
    print(f"  总计: {mem_std_total / 1e9:.2f} GB")
    
    # Flash Attention：不存中间矩阵
    mem_Q = T * d * n_heads * bytes_per_elem * layers
    mem_K = T * d * n_heads * bytes_per_elem * layers
    mem_V = T * d * n_heads * bytes_per_elem * layers
    mem_O = T * d * n_heads * bytes_per_elem * layers
    mem_flash_total = mem_Q + mem_K + mem_V + mem_O
    
    print(f"\nFlash Attention（训练时）:")
    print(f"  Q: {mem_Q / 1e9:.3f} GB")
    print(f"  K: {mem_K / 1e9:.3f} GB")
    print(f"  V: {mem_V / 1e9:.3f} GB")
    print(f"  O: {mem_O / 1e9:.3f} GB")
    print(f"  总计: {mem_flash_total / 1e9:.3f} GB")
    print(f"  （S 和 P 不存储，反向时重计算）")
    
    ratio = mem_std_total / mem_flash_total
    print(f"\n⚡ Flash Attention 的注意力显存占用是标准实现的 1/{ratio:.1f}")
    print(f"   （省了 {(mem_std_total - mem_flash_total) / 1e9:.2f} GB）")


# ============================================================
# 5. 在线 Softmax 演示 — 逐步展示修正过程
# ============================================================

def demonstrate_online_softmax():
    """
    用一个简单的数值例子演示在线 Softmax 的逐步计算过程。
    这帮助理解 Flash Attention 中"逐块更新"的核心机制。
    """
    print(f"\n{'='*60}")
    print(f"🧮 在线 Softmax 逐步演示")
    print(f"{'='*60}")
    
    # 假设一行注意力分数（Q 和所有 K 的点积，已经缩放）
    x = torch.tensor([2.0, 1.0, 3.0, 0.5, 2.5])
    n = len(x)
    
    print(f"\n输入分数 x = {x.tolist()}")
    
    # 标准 softmax（一次性计算）
    x_max = x.max()
    exp_x = torch.exp(x - x_max)
    softmax_result = exp_x / exp_x.sum()
    print(f"\n标准 softmax 结果: {[f'{v:.4f}' for v in softmax_result.tolist()]}")
    print(f"  max = {x_max.item():.1f}, sum(exp) = {exp_x.sum().item():.4f}")
    
    # 在线 softmax（分块计算，模拟 Flash Attention 的过程）
    print(f"\n--- 在线 Softmax（分 3 块处理）---")
    
    # 模拟分块：块大小 B_c = 2
    blocks = [x[0:2], x[2:4], x[4:5]]
    block_names = ["[2.0, 1.0]", "[3.0, 0.5]", "[2.5]"]
    
    # 初始化
    running_max = float('-inf')
    running_sum = 0.0
    running_output = torch.zeros(n)  # 用于最终验证
    
    # 存储每个元素的 exp 值（用于最终验证）
    all_exp_values = []
    
    for i, (block, name) in enumerate(zip(blocks, block_names)):
        print(f"\n  处理第 {i+1} 块: {name}")
        
        # 当前块的最大值
        block_max = block.max().item()
        print(f"    当前块 max = {block_max:.1f}, 之前全局 max = {running_max:.1f}")
        
        # 新的全局 max
        new_max = max(running_max, block_max)
        
        # 校正旧的 sum
        if running_max != float('-inf'):
            correction = math.exp(running_max - new_max)
            print(f"    校正因子 exp({running_max:.1f} - {new_max:.1f}) = {correction:.4f}")
            running_sum *= correction
            # 校正之前的 exp 值
            all_exp_values = [v * correction for v in all_exp_values]
        
        # 新块的 exp
        block_exp = torch.exp(block - new_max)
        print(f"    新块 exp(x - {new_max:.1f}) = {[f'{v:.4f}' for v in block_exp.tolist()]}")
        
        # 更新
        running_max = new_max
        running_sum += block_exp.sum().item()
        all_exp_values.extend(block_exp.tolist())
        
        print(f"    更新后: max = {running_max:.1f}, sum = {running_sum:.4f}")
        
        # 当前（部分）softmax
        partial_softmax = [v / running_sum for v in all_exp_values]
        print(f"    当前 softmax = {[f'{v:.4f}' for v in partial_softmax]}")
    
    # 最终结果
    final_softmax = torch.tensor(all_exp_values) / running_sum
    print(f"\n  最终在线 softmax: {[f'{v:.4f}' for v in final_softmax.tolist()]}")
    print(f"  标准 softmax:     {[f'{v:.4f}' for v in softmax_result.tolist()]}")
    print(f"  ✅ 完全一致！最大误差: {(final_softmax - softmax_result).abs().max().item():.2e}")


# ============================================================
# 6. 速度对比：PyTorch SDPA（内置 Flash Attention）vs 手动实现
# ============================================================

def benchmark_attention(B=2, n_heads=8, T=512, d_head=64, num_warmup=3, num_iters=10):
    """
    对比三种注意力实现的速度：
    1. 标准手动实现
    2. 分块模拟实现
    3. PyTorch SDPA（可能自动使用 Flash Attention）
    """
    print(f"\n{'='*60}")
    print(f"⏱️  速度对比 (B={B}, n_heads={n_heads}, T={T}, d_head={d_head})")
    print(f"{'='*60}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  设备: {device}")
    
    # 生成随机数据
    torch.manual_seed(42)
    Q = torch.randn(B, n_heads, T, d_head, device=device)
    K = torch.randn(B, n_heads, T, d_head, device=device)
    V = torch.randn(B, n_heads, T, d_head, device=device)
    
    results = {}
    
    # 测试标准实现
    if device.type == 'cpu':
        # CPU 上直接测试
        times_std = []
        for _ in range(num_warmup):
            _ = standard_attention(Q, K, V)
        for _ in range(num_iters):
            start = time.perf_counter()
            O_std = standard_attention(Q, K, V)
            times_std.append(time.perf_counter() - start)
        results['标准注意力'] = sum(times_std) / len(times_std)
        print(f"\n  标准注意力: {results['标准注意力']*1000:.2f} ms")
    else:
        # GPU 上用 CUDA events 计时
        torch.cuda.synchronize()
        for _ in range(num_warmup):
            _ = standard_attention(Q, K, V)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iters):
            O_std = standard_attention(Q, K, V)
        end_event.record()
        torch.cuda.synchronize()
        results['标准注意力'] = start_event.elapsed_time(end_event) / num_iters
        print(f"\n  标准注意力: {results['标准注意力']:.2f} ms")
    
    # 测试分块模拟实现
    if device.type == 'cpu':
        times_flash = []
        for _ in range(num_warmup):
            _ = flash_attention_simulated(Q, K, V, block_size=64)
        for _ in range(num_iters):
            start = time.perf_counter()
            O_flash = flash_attention_simulated(Q, K, V, block_size=64)
            times_flash.append(time.perf_counter() - start)
        results['分块模拟'] = sum(times_flash) / len(times_flash)
        print(f"  分块模拟:   {results['分块模拟']*1000:.2f} ms")
    else:
        torch.cuda.synchronize()
        for _ in range(num_warmup):
            _ = flash_attention_simulated(Q, K, V, block_size=64)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iters):
            O_flash = flash_attention_simulated(Q, K, V, block_size=64)
        end_event.record()
        torch.cuda.synchronize()
        results['分块模拟'] = start_event.elapsed_time(end_event) / num_iters
        print(f"  分块模拟:   {results['分块模拟']:.2f} ms")
    
    # 测试 PyTorch SDPA
    if device.type == 'cpu':
        times_sdpa = []
        for _ in range(num_warmup):
            _ = F.scaled_dot_product_attention(Q, K, V)
        for _ in range(num_iters):
            start = time.perf_counter()
            O_sdpa = F.scaled_dot_product_attention(Q, K, V)
            times_sdpa.append(time.perf_counter() - start)
        results['SDPA'] = sum(times_sdpa) / len(times_sdpa)
        print(f"  SDPA (PyTorch): {results['SDPA']*1000:.2f} ms")
    else:
        torch.cuda.synchronize()
        for _ in range(num_warmup):
            _ = F.scaled_dot_product_attention(Q, K, V)
        torch.cuda.synchronize()
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_iters):
            O_sdpa = F.scaled_dot_product_attention(Q, K, V)
        end_event.record()
        torch.cuda.synchronize()
        results['SDPA'] = start_event.elapsed_time(end_event) / num_iters
        print(f"  SDPA (PyTorch): {results['SDPA']:.2f} ms")
    
    # 注意：分块模拟是 Python 实现的，比 C++/CUDA 的标准实现慢是正常的
    # 真正的 Flash Attention 是用 CUDA 写的，比标准实现快 2-4x
    print(f"\n  💡 注意：分块模拟是 Python 实现的，主要用于理解算法。")
    print(f"     真正的 Flash Attention 是 CUDA 实现，比标准注意力快 2-4x。")
    
    return results


# ============================================================
# 主函数：运行所有实验
# ============================================================

def main():
    print("=" * 60)
    print("🧠 Day 15: Flash Attention — 分块注意力计算的核心思想")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # 实验 1: 数值验证 — 确认分块实现和标准实现完全一致
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"✅ 实验 1: 数值验证（标准 vs 分块）")
    print(f"{'='*60}")
    
    torch.manual_seed(42)
    B, n_heads, T, d_head = 2, 4, 128, 32
    
    Q = torch.randn(B, n_heads, T, d_head)
    K = torch.randn(B, n_heads, T, d_head)
    V = torch.randn(B, n_heads, T, d_head)
    
    print(f"\n输入形状: Q={Q.shape}, K={K.shape}, V={V.shape}")
    
    # 标准实现
    O_std = standard_attention(Q, K, V)
    print(f"标准注意力输出形状: {O_std.shape}")
    
    # 分块实现（不同块大小）
    for block_size in [32, 64, 128]:
        O_flash = flash_attention_simulated(Q, K, V, block_size=block_size)
        max_diff = (O_std - O_flash).abs().max().item()
        print(f"  分块实现 (block_size={block_size:>3}): 最大误差 = {max_diff:.2e}  "
              f"{'✅ 完全一致' if max_diff < 1e-5 else '❌ 有差异'}")
    
    # ----------------------------------------------------------
    # 实验 2: 在线 Softmax 演示
    # ----------------------------------------------------------
    demonstrate_online_softmax()
    
    # ----------------------------------------------------------
    # 实验 3: IO 量估算
    # ----------------------------------------------------------
    estimate_io(T=2048, d=128, n_heads=32, block_size=64)
    estimate_io(T=8192, d=128, n_heads=32, block_size=64)
    
    # ----------------------------------------------------------
    # 实验 4: 显存占用对比
    # ----------------------------------------------------------
    estimate_memory(T=2048, d=128, n_heads=32, layers=32)
    estimate_memory(T=8192, d=128, n_heads=32, layers=32)
    
    # ----------------------------------------------------------
    # 实验 5: 速度对比
    # ----------------------------------------------------------
    benchmark_attention(B=2, n_heads=8, T=512, d_head=64)
    
    # ----------------------------------------------------------
    # 总结
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"📋 总结")
    print(f"{'='*60}")
    print(f"""
Flash Attention 的三个核心思想：

1. 📦 分块计算（Tiling）
   - 把 Q、K、V 切成小块，每次只在 SRAM 中处理一小块
   - 避免 N×N 的巨大中间矩阵溢出到 HBM

2. 🔄 在线 Softmax（Online Softmax）
   - 逐块更新 softmax，通过"修正因子"保持数值等价
   - 不需要先看完所有数据就能开始计算

3. 🔁 重计算优于缓存（Recomputation > Caching）
   - 不存中间矩阵，反向传播时重计算
   - 在 GPU 上，计算快、IO 慢 → 重计算比读缓存更快

结果：
- ⚡ 速度提升 2-4x（减少 HBM 读写）
- 💾 显存从 O(N²) 降到 O(N)
- ✅ 数学结果完全一致（不是近似！）
    """)


if __name__ == "__main__":
    main()
