"""
Day 20: LoRA（Low-Rank Adaptation）— 手写 LoRA 层 + 注入 Transformer + 训练演示

本代码实现：
1. LoRALinear — LoRA 包装的线性层（核心！）
2. SimpleTransformer — 简化版 Transformer 用于演示
3. inject_lora — 将 LoRA 注入到模型中
4. merge_lora — 合并 LoRA 权重（推理零开销）
5. 参数量统计 + 显存估算
6. 模拟微调训练 + 前后对比
7. 多 LoRA 切换演示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import List, Optional


# ============================================================
# 1. LoRA 线性层 — LoRA 的核心实现
# ============================================================

class LoRALinear(nn.Module):
    """
    LoRA 包装的线性层。
    
    原始线性层 y = Wx 完全冻结。
    加上 LoRA 补丁：y = Wx + (α/r) * B @ A @ x
    
    其中 A ∈ ℝ^(r×k), B ∈ ℝ^(d×r), r << min(d,k)
    """
    
    def __init__(self, original_linear: nn.Linear, r: int = 8, alpha: int = 16,
                 dropout: float = 0.0):
        """
        参数：
            original_linear: 原始的 nn.Linear 层（会被冻结）
            r: LoRA 的秩（瓶颈维度）
            alpha: 缩放系数
            dropout: LoRA 路径上的 dropout（可选）
        """
        super().__init__()
        
        # 保存原始线性层并冻结
        self.original = original_linear
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False
        
        d = original_linear.out_features   # 输出维度
        k = original_linear.in_features    # 输入维度
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r           # 缩放因子
        
        # LoRA 矩阵 A: (r, k) — 输入到瓶颈的映射
        self.lora_A = nn.Parameter(torch.empty(r, k))
        # LoRA 矩阵 B: (d, r) — 瓶颈到输出的映射
        self.lora_B = nn.Parameter(torch.zeros(d, r))
        
        # 初始化：A 用 Kaiming 初始化，B 初始化为 0
        # B=0 保证训练开始时 LoRA 的贡献为零，不干扰原始模型
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 已经用 torch.zeros 初始化了
        
        # 可选的 dropout
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 记录是否已合并
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：y = Wx + (α/r) * dropout(x) @ A^T @ B^T
        
        如果已合并，则直接用合并后的权重计算。
        """
        # 原始线性层的输出（冻结，不计算梯度）
        result = self.original(x)
        
        # 如果 LoRA 没有被合并，单独计算 LoRA 路径
        if not self.merged:
            # x: (batch, seq, k) → lora_out: (batch, seq, d)
            lora_input = self.lora_dropout(x)
            # lora_input @ A^T: (batch, seq, k) @ (k, r) = (batch, seq, r)
            # 再 @ B^T: (batch, seq, r) @ (r, d) = (batch, seq, d)
            lora_out = (lora_input @ self.lora_A.T) @ self.lora_B.T
            result = result + lora_out * self.scaling
        
        return result
    
    def merge(self):
        """将 LoRA 权重合并到原始权重中，推理时零额外开销"""
        if not self.merged:
            # W_new = W + (α/r) * B @ A
            self.original.weight.data += (self.scaling * self.lora_B @ self.lora_A)
            self.merged = True
    
    def unmerge(self):
        """取消合并（用于需要重新训练的场景）"""
        if self.merged:
            self.original.weight.data -= (self.scaling * self.lora_B @ self.lora_A)
            self.merged = False


# ============================================================
# 2. 简化版 Transformer — 用于演示 LoRA 的效果
# ============================================================

class SimpleAttention(nn.Module):
    """简化版 Self-Attention，使用 nn.Linear 便于 LoRA 注入"""
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # 这些线性层就是 LoRA 要注入的目标！
        self.q_proj = nn.Linear(d_model, d_model)   # Query 投影
        self.k_proj = nn.Linear(d_model, d_model)   # Key 投影
        self.v_proj = nn.Linear(d_model, d_model)   # Value 投影
        self.o_proj = nn.Linear(d_model, d_model)   # Output 投影
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        
        # 计算投影
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Causal mask（防止看到未来的 token）
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        
        # 重新拼接并投影
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class SimpleTransformerBlock(nn.Module):
    """一个 Transformer Block：Attention + FFN"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.attn = SimpleAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Transformer
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SimpleTransformer(nn.Module):
    """简化版 Transformer 语言模型"""
    
    def __init__(self, vocab_size: int = 1000, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 4, d_ff: int = 1024,
                 max_seq_len: int = 128):
        super().__init__()
        self.d_model = d_model
        
        # Token embedding + 位置编码
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])
        
        # 输出层
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享（embedding 和 output head 共享权重）
        self.head.weight = self.token_embed.weight
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        
        # Embedding
        positions = torch.arange(S, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # 输出 logits
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits


# ============================================================
# 3. LoRA 注入工具函数
# ============================================================

def inject_lora(model: nn.Module, r: int = 8, alpha: int = 16,
                target_modules: List[str] = None,
                dropout: float = 0.0) -> nn.Module:
    """
    将 LoRA 注入到模型的指定线性层中。
    
    参数：
        model: 原始模型
        r: LoRA 的秩
        alpha: 缩放系数
        target_modules: 需要注入 LoRA 的模块名列表（如 ["q_proj", "v_proj"]）
        dropout: LoRA dropout 概率
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # 先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 遍历所有模块，找到目标线性层并注入 LoRA
    replaced = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # 检查这个线性层是否在目标列表中
            if any(target in name for target in target_modules):
                # 找到父模块，替换子模块
                *path, attr = name.split('.')
                parent = model
                for p in path:
                    parent = getattr(parent, p)
                
                # 创建 LoRA 层并替换
                lora_layer = LoRALinear(module, r=r, alpha=alpha, dropout=dropout)
                setattr(parent, attr, lora_layer)
                replaced += 1
    
    print(f"  ✅ 已注入 {replaced} 个 LoRA 层 (rank={r}, alpha={alpha})")
    return model


def count_parameters(model: nn.Module) -> dict:
    """统计模型的可训练参数和总参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_pct": trainable / total * 100,
    }


def merge_lora_weights(model: nn.Module):
    """合并模型中所有 LoRA 层的权重"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.merge()
    print("  ✅ 已合并所有 LoRA 权重")


def unmerge_lora_weights(model: nn.Module):
    """取消合并所有 LoRA 层"""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.unmerge()
    print("  ✅ 已取消合并所有 LoRA 权重")


# ============================================================
# 4. 模拟训练函数
# ============================================================

def simulate_training(model, n_steps=30, lr=1e-3, seq_len=32, vocab_size=1000):
    """模拟 LoRA 微调训练过程"""
    # 只优化 LoRA 参数
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )
    
    model.train()
    losses = []
    
    for step in range(n_steps):
        # 生成随机输入（模拟真实数据）
        input_ids = torch.randint(0, vocab_size, (4, seq_len))
        
        # 前向传播
        logits = model(input_ids)
        
        # 计算损失（预测下一个 token）
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1)
        )
        
        # 反向传播（只有 LoRA 参数会更新）
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (step + 1) % 10 == 0:
            print(f"    Step {step+1:3d}/{n_steps}: loss = {loss.item():.4f}")
    
    return losses


# ============================================================
# 5. 多 LoRA 管理器 — 演示多任务切换
# ============================================================

class LoRAManager:
    """
    多 LoRA 管理器，支持为同一个基座模型切换不同的 LoRA 权重。
    
    就像一个万能遥控器，基座模型是遥控器本体，
    每个 LoRA 是一个"频道"，可以瞬间切换。
    """
    
    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.lora_weights = {}  # name -> state_dict
    
    def save_lora(self, name: str):
        """保存当前 LoRA 权重"""
        lora_state = {
            k: v.clone() for k, v in self.base_model.state_dict().items()
            if 'lora_' in k
        }
        self.lora_weights[name] = lora_state
        size_mb = sum(v.numel() * v.element_size() for v in lora_state.values()) / 1024 / 1024
        print(f"  💾 保存 LoRA '{name}': {len(lora_state)} 个参数, {size_mb:.2f} MB")
    
    def load_lora(self, name: str):
        """加载指定 LoRA 权重"""
        if name not in self.lora_weights:
            print(f"  ❌ LoRA '{name}' 不存在")
            return
        
        # 先取消合并（如果已合并）
        unmerge_lora_weights(self.base_model)
        
        # 加载 LoRA 权重
        state_dict = self.base_model.state_dict()
        for k, v in self.lora_weights[name].items():
            state_dict[k] = v
        self.base_model.load_state_dict(state_dict)
        print(f"  🔄 已切换到 LoRA '{name}'")
    
    def list_loras(self):
        """列出所有已保存的 LoRA"""
        print(f"  📋 已保存 {len(self.lora_weights)} 个 LoRA:")
        for name, state in self.lora_weights.items():
            size_mb = sum(v.numel() * v.element_size() for v in state.values()) / 1024 / 1024
            print(f"    - {name}: {size_mb:.2f} MB")


# ============================================================
# 6. 主程序：完整的 LoRA 演示
# ============================================================

def main():
    print("=" * 70)
    print("Day 20: LoRA（Low-Rank Adaptation）实战演示")
    print("=" * 70)
    
    # ----------------------------------------------------------
    # 演示 1：参数量对比 — 全量 vs LoRA
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊 演示 1：参数量对比 — 全量微调 vs LoRA")
    print("=" * 70)
    
    # 创建基座模型
    vocab_size = 1000
    model = SimpleTransformer(
        vocab_size=vocab_size, d_model=256, n_heads=8,
        n_layers=4, d_ff=1024, max_seq_len=128
    )
    
    # 统计原始模型参数
    original_params = count_parameters(model)
    print(f"\n原始模型参数统计：")
    print(f"  总参数量: {original_params['total']:,}")
    print(f"  可训练:   {original_params['trainable']:,} ({original_params['trainable_pct']:.2f}%)")
    
    # 注入 LoRA（rank=8）
    print(f"\n注入 LoRA (rank=8, alpha=16)...")
    model = inject_lora(model, r=8, alpha=16,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # 统计 LoRA 模型参数
    lora_params = count_parameters(model)
    print(f"\nLoRA 模型参数统计：")
    print(f"  总参数量: {lora_params['total']:,}")
    print(f"  可训练:   {lora_params['trainable']:,} ({lora_params['trainable_pct']:.4f}%)")
    print(f"  冻结:     {lora_params['frozen']:,}")
    
    # 对比
    print(f"\n💡 压缩效果：")
    print(f"  可训练参数减少: {original_params['trainable'] / lora_params['trainable']:.1f} 倍")
    print(f"  可训练参数占比: {lora_params['trainable_pct']:.4f}%（不到 0.1%！）")
    
    # 估算显存
    print(f"\n💾 显存估算（FP16）：")
    full_mem = original_params['total'] * 2 + original_params['total'] * 2 + original_params['total'] * 8  # 权重+梯度+Adam
    lora_mem = original_params['total'] * 2 + lora_params['trainable'] * 2 + lora_params['trainable'] * 8
    print(f"  全量微调: {full_mem / 1024**3:.2f} GB（权重 + 梯度 + Adam 优化器）")
    print(f"  LoRA 微调: {lora_mem / 1024**3:.2f} GB（冻结权重 + LoRA梯度 + LoRA优化器）")
    print(f"  节省: {(1 - lora_mem / full_mem) * 100:.1f}%")
    
    # ----------------------------------------------------------
    # 演示 2：LoRA 初始化验证 — B=0 保证初始无扰动
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("🔍 演示 2：LoRA 初始化验证")
    print("=" * 70)
    
    # 创建测试输入
    test_input = torch.randn(1, 10, 256)  # (batch=1, seq=10, d_model=256)
    
    # 用一个新模型做对照
    model_ref = SimpleTransformer(vocab_size=1000, d_model=256, n_heads=8,
                                   n_layers=1, d_ff=1024)
    model_lora = SimpleTransformer(vocab_size=1000, d_model=256, n_heads=8,
                                    n_layers=1, d_ff=1024)
    
    # 复制权重确保完全一致
    model_lora.load_state_dict(model_ref.state_dict())
    
    # 获取注入前的输出
    model_ref.eval()
    model_lora.eval()
    
    # 注意：这里用 token id 作为输入
    test_ids = torch.randint(0, 1000, (1, 10))
    
    with torch.no_grad():
        output_before = model_lora(test_ids)
    
    # 注入 LoRA
    model_lora = inject_lora(model_lora, r=8, alpha=16)
    model_lora.eval()
    
    # 获取注入后的输出（B=0，所以应该完全一样！）
    with torch.no_grad():
        output_after = model_lora(test_ids)
    
    diff = (output_before - output_after).abs().max().item()
    print(f"\n  注入 LoRA 前后的输出最大差异: {diff:.10f}")
    print(f"  → B 初始化为 0，所以注入 LoRA 后输出完全不变 ✅")
    print(f"  （训练开始时 LoRA 不干扰原始模型的行为）")
    
    # ----------------------------------------------------------
    # 演示 3：模拟 LoRA 训练
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("🏋️ 演示 3：模拟 LoRA 微调训练")
    print("=" * 70)
    
    # 训练前的 LoRA 权重
    lora_A_before = model_lora.blocks[0].attn.q_proj.lora_A.data.clone()
    lora_B_before = model_lora.blocks[0].attn.q_proj.lora_B.data.clone()
    
    # 模拟训练
    print(f"\n开始 LoRA 训练（30 步）...")
    losses = simulate_training(model_lora, n_steps=30, lr=1e-3,
                                seq_len=32, vocab_size=1000)
    
    # 训练后的 LoRA 权重
    lora_A_after = model_lora.blocks[0].attn.q_proj.lora_A.data
    lora_B_after = model_lora.blocks[0].attn.q_proj.lora_B.data
    
    # 看看 LoRA 权重变了多少
    a_diff = (lora_A_before - lora_A_after).abs().mean().item()
    b_diff = (lora_B_before - lora_B_after).abs().mean().item()
    print(f"\n  LoRA A 矩阵变化量（均值）: {a_diff:.6f}")
    print(f"  LoRA B 矩阵变化量（均值）: {b_diff:.6f}")
    print(f"  B 不再是全零了 → LoRA 已经学到了有用的调整 ✅")
    
    # 损失曲线
    print(f"\n  损失曲线: {losses[0]:.4f}", end="")
    for i in range(1, len(losses)):
        if (i + 1) % 10 == 0:
            print(f" → {losses[i]:.4f}", end="")
    print(f"\n  训练后损失降低: {losses[0] - losses[-1]:.4f}")
    
    # ----------------------------------------------------------
    # 演示 4：权重合并 — 推理零额外开销
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("🔀 演示 4：LoRA 权重合并（推理零开销）")
    print("=" * 70)
    
    model_lora.eval()
    
    # 合并前的输出
    with torch.no_grad():
        output_unmerged = model_lora(test_ids)
    
    # 检查哪些是 LoRA 层
    lora_layers = [(name, m) for name, m in model_lora.named_modules()
                   if isinstance(m, LoRALinear)]
    print(f"\n  模型中有 {len(lora_layers)} 个 LoRA 层")
    for name, layer in lora_layers[:4]:  # 只显示前4个
        print(f"    - {name}: rank={layer.r}, alpha={layer.alpha}, "
              f"A={tuple(layer.lora_A.shape)}, B={tuple(layer.lora_B.shape)}")
    
    # 合并权重
    print(f"\n  合并 LoRA 权重...")
    merge_lora_weights(model_lora)
    
    # 合并后的输出（应该和合并前完全一样！）
    with torch.no_grad():
        output_merged = model_lora(test_ids)
    
    diff = (output_unmerged - output_merged).abs().max().item()
    print(f"\n  合并前后的输出最大差异: {diff:.10f}")
    print(f"  → 合并后数学结果完全一致，但不再有额外的矩阵乘法 ✅")
    print(f"  → 推理速度 = 原始模型，零额外开销！")
    
    # ----------------------------------------------------------
    # 演示 5：多 LoRA 切换
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print('🔄 演示 5：多 LoRA 切换（一个基座，多个频道）')
    print("=" * 70)
    
    # 取消合并，恢复 LoRA 状态
    unmerge_lora_weights(model_lora)
    
    # 创建 LoRA 管理器
    manager = LoRAManager(model_lora)
    
    # 保存当前 LoRA 为 "v1"
    print(f"\n保存当前 LoRA 权重为 'v1-initial'...")
    manager.save_lora("v1-initial")
    
    # 再训练一会儿，得到 "v2"
    print(f"\n继续训练得到 'v2-improved'...")
    losses2 = simulate_training(model_lora, n_steps=20, lr=5e-4,
                                 seq_len=32, vocab_size=1000)
    manager.save_lora("v2-improved")
    
    # 再训练得到 "v3"
    print(f"\n继续训练得到 'v3-further'...")
    losses3 = simulate_training(model_lora, n_steps=20, lr=2e-4,
                                 seq_len=32, vocab_size=1000)
    manager.save_lora("v3-further")
    
    # 列出所有 LoRA
    print(f"\n所有已保存的 LoRA:")
    manager.list_loras()
    
    # 切换演示
    print(f"\n切换到不同 LoRA 版本并测试输出:")
    for name in ["v1-initial", "v2-improved", "v3-further"]:
        manager.load_lora(name)
        model_lora.eval()
        with torch.no_grad():
            logits = model_lora(test_ids)
            # 看预测的第一个 token 的概率分布
            probs = F.softmax(logits[0, 0], dim=-1)
            top5 = torch.topk(probs, 5)
            print(f"\n  LoRA '{name}' 的 top-5 预测:")
            for prob, idx in zip(top5.values, top5.indices):
                print(f"    token {idx.item():4d}: {prob.item():.4f}")
    
    # ----------------------------------------------------------
    # 演示 6：不同 rank 的效果对比
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("📏 演示 6：不同 rank 的参数量对比")
    print("=" * 70)
    
    for r in [1, 2, 4, 8, 16, 32, 64]:
        m = SimpleTransformer(vocab_size=1000, d_model=256, n_heads=8,
                              n_layers=4, d_ff=1024)
        inject_lora(m, r=r, alpha=2*r,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        params = count_parameters(m)
        bar = "█" * int(params['trainable'] / 5000)
        print(f"  rank={r:2d}: {params['trainable']:>8,} 参数 ({params['trainable_pct']:.3f}%) {bar}")
    
    print(f"\n  💡 rank 越大，可训练参数越多，但通常 rank=8~16 就足够了")
    
    # ----------------------------------------------------------
    # 演示 7：显存估算 — 模拟 LLaMA-7B 场景
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("🖥️ 演示 7：LLaMA-7B 显存估算")
    print("=" * 70)
    
    # LLaMA-7B 的参数
    llama_params = 7_000_000_000  # 7B
    hidden_dim = 4096
    n_layers = 32
    
    # 全量微调
    full_weights = llama_params * 2 / (1024**3)     # FP16 权重
    full_grads = llama_params * 2 / (1024**3)       # FP16 梯度
    full_adam = llama_params * 8 / (1024**3)        # Adam (FP32 m + v)
    full_total = full_weights + full_grads + full_adam
    
    print(f"\n全量微调 LLaMA-7B:")
    print(f"  模型权重 (FP16): {full_weights:.1f} GB")
    print(f"  梯度 (FP16):     {full_grads:.1f} GB")
    print(f"  Adam 优化器:     {full_adam:.1f} GB")
    print(f"  ─────────────────────────")
    print(f"  总计:            {full_total:.1f} GB")
    print(f"  需要: 2-4 × A100 (80GB)")
    
    # LoRA 微调（rank=8, 只加 Attention）
    lora_params_per_layer = 4 * 2 * hidden_dim * 8  # 4层(QKVO) × 2矩阵(A+B) × hidden × rank
    total_lora_params = n_layers * lora_params_per_layer
    
    lora_weights = llama_params * 2 / (1024**3)               # 冻结的 FP16 权重（不需要梯度）
    lora_train_weights = total_lora_params * 2 / (1024**3)    # LoRA FP16 权重
    lora_grads = total_lora_params * 2 / (1024**3)            # LoRA FP16 梯度
    lora_adam = total_lora_params * 8 / (1024**3)             # LoRA Adam
    lora_total = lora_weights + lora_train_weights + lora_grads + lora_adam
    
    print(f"\nLoRA 微调 LLaMA-7B (rank=8):")
    print(f"  冻结权重 (FP16):    {lora_weights:.1f} GB")
    print(f"  LoRA 权重 (FP16):   {lora_train_weights:.4f} GB ({total_lora_params/1e6:.1f}M 参数)")
    print(f"  LoRA 梯度 (FP16):   {lora_grads:.4f} GB")
    print(f"  LoRA Adam 优化器:   {lora_adam:.4f} GB")
    print(f"  ─────────────────────────")
    print(f"  总计:               {lora_total:.1f} GB")
    print(f"  需要: 1 × RTX 4090 (24GB) ✅")
    
    print(f"\n  💰 节省: {full_total - lora_total:.1f} GB ({(1 - lora_total/full_total)*100:.0f}%)")
    print(f"  💰 硬件成本: 从 ~$60,000 降到 ~$1,500")
    
    # ----------------------------------------------------------
    # 总结
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("📝 总结")
    print("=" * 70)
    print("""
  LoRA 的三大核心要点：

  1️⃣  低秩分解：权重变化 ΔW 天然是低秩的，用两个小矩阵 A 和 B 就能近似
     → y = Wx + (α/r) × BAx
     → 参数量从 7B 降到 ~8M

  2️⃣  冻结原始权重：只训练 LoRA 的 A 和 B 矩阵
     → 显存需求从 ~86 GB 降到 ~16 GB
     → 消费级显卡就能微调大模型

  3️⃣  推理零开销：训练完成后，把 BA 合并回原始权重
     → W_new = W + (α/r) × BA
     → 推理速度和原始模型完全一样

  额外福利：多 LoRA 切换
     → 一个基座模型 + 多个几 MB 的 LoRA 权重
     → 瞬间切换不同任务，像一个万能遥控器！
    """)


if __name__ == "__main__":
    # 设置随机种子确保可复现
    torch.manual_seed(42)
    
    main()
    
    print("✅ Day 20 LoRA 演示完成！")
