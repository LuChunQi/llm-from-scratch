#!/usr/bin/env python3
"""
Day 19: DPO（Direct Preference Optimization）— 绕过强化学习，直接优化偏好

本代码从零实现 DPO 的完整流程：
1. 构造模拟偏好数据（指令 + 好回答 + 差回答）
2. 实现 DPO 损失函数
3. 训练策略模型 π_θ（参考模型 π_ref 冻结）
4. 对比 SFT 模型 vs DPO 模型的输出
5. 可视化隐式奖励在训练中的变化

运行: python3 dpo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

# ============================================================
# 1. 定义超小 Transformer LM（和前几天的结构一致）
# ============================================================

class SimpleTokenizer:
    """字符级分词器（简化版，足够教学用）"""

    def __init__(self, text: str):
        # 收集所有唯一字符
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        # 字符 → ID 的映射
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        # ID → 字符 的映射
        self.id_to_char = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> list[int]:
        """文本 → ID 列表"""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """ID 列表 → 文本"""
        return "".join(self.id_to_char[i] for i in ids)


class MiniTransformerLM(nn.Module):
    """
    一个超小的 Transformer 语言模型，用于教学演示。
    结构：Token Embedding + 位置编码 + Transformer 层 + 输出头
    """

    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Token 嵌入层：将 token ID 映射到 d_model 维向量
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码：用可学习参数（简单起见不用 sinusoidal）
        self.pos_embedding = nn.Embedding(256, d_model)  # 最多支持 256 个位置

        # Transformer 编码器层（简化版，2 层）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,  # 教学演示关闭 dropout
            batch_first=True,  # 输入形状 (batch, seq, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 输出头：d_model → vocab_size 的线性层
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播：给定 input_ids，返回每个位置的 logits
        input_ids: (batch, seq_len)
        返回: (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # 生成位置 ID：0, 1, 2, ..., seq_len-1
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        # Token embedding + Position embedding
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        # 因果 mask：每个位置只能看到自己及之前的位置（自回归）
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1
        ).bool()

        # 通过 Transformer
        x = self.transformer(x, mask=causal_mask)

        # 输出 logits
        logits = self.output_head(x)
        return logits

    def get_log_probs(self, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        计算给定 input_ids 和 labels 的对数概率（每个 token 的平均）
        input_ids: (batch, seq_len) — 输入 token 序列
        labels: (batch, seq_len) — 目标 token 序列（通常 input_ids 和 labels 是同一个序列）
        返回: (batch,) — 每个样本的平均对数概率
        """
        logits = self.forward(input_ids)  # (batch, seq_len, vocab_size)

        # 对每个位置，用 log_softmax 计算所有 token 的对数概率
        log_probs_all = F.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab_size)

        # 取出目标 token 位置的对数概率
        # gather 操作：在每个位置挑出 labels 指定的那个 token 的概率
        token_log_probs = log_probs_all.gather(
            dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)  # (batch, seq_len)

        # 返回每个样本的平均对数概率
        return token_log_probs.mean(dim=-1)  # (batch,)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50) -> torch.Tensor:
        """
        自回归生成文本
        input_ids: (1, seq_len) — 输入的 prompt
        返回: (1, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # 取最后 256 个 token（位置编码限制）
            context = input_ids[:, -256:]
            # 前向传播获取 logits
            logits = self.forward(context)
            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :]  # (1, vocab_size)
            # 取概率最大的 token（贪心解码）
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
            # 拼接到输入后面
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# ============================================================
# 2. 构造模拟数据
# ============================================================

def create_preference_dataset(tokenizer: SimpleTokenizer):
    """
    创建模拟偏好数据集。
    每条数据是一个三元组：(指令, 好回答, 差回答)
    
    真实场景中，这些偏好数据来自人类标注员对不同回答的排序。
    这里我们手工构造，确保"好回答"确实比"差回答"好。
    """
    preference_data = [
        # (指令+回答 的完整文本, 类别)
        # 每一对是同一个指令的好回答和差回答
        {
            "prompt": "问：什么是光合作用？答：",
            "chosen": "问：什么是光合作用？答：光合作用是植物利用阳光将二氧化碳和水转化为有机物和氧气的过程。",
            "rejected": "问：什么是光合作用？答：光合作用就是植物吃阳光。",
        },
        {
            "prompt": "问：地球为什么是圆的？答：",
            "chosen": "问：地球为什么是圆的？答：因为万有引力使物质向质心聚集，在足够大的天体上形成近似球形的形状。",
            "rejected": "问：地球为什么是圆的？答：因为圆的东西好看。",
        },
        {
            "prompt": "问：1加1等于几？答：",
            "chosen": "问：1加1等于几？答：1加1等于2，这是基本的数学运算。",
            "rejected": "问：1加1等于几？答：大约是3吧。",
        },
        {
            "prompt": "问：太阳是什么颜色的？答：",
            "chosen": "问：太阳是什么颜色的？答：太阳的光是白色的，包含了所有可见光的颜色，但在地球上看起来偏黄。",
            "rejected": "问：太阳是什么颜色的？答：太阳是红色的。",
        },
        {
            "prompt": "问：水为什么会沸腾？答：",
            "chosen": "问：水为什么会沸腾？答：当水温达到沸点时，水分子获得足够动能从液态转变为气态，形成气泡上升。",
            "rejected": "问：水为什么会沸腾？答：因为水很生气。",
        },
        {
            "prompt": "问：月亮为什么会有阴晴圆缺？答：",
            "chosen": "问：月亮为什么会有阴晴圆缺？答：因为月球绕地球公转，太阳照亮月球的角度不同，我们从地球上看到的被照亮面积就不同。",
            "rejected": "问：月亮为什么会有阴晴圆缺？答：因为月亮被天狗吃了。",
        },
        {
            "prompt": "问：风是怎么产生的？答：",
            "chosen": "问：风是怎么产生的？答：风是由大气压差引起的空气流动，太阳对地球表面加热不均匀导致不同区域气压不同。",
            "rejected": "问：风是怎么产生的？答：风是吹出来的。",
        },
        {
            "prompt": "问：为什么天空是蓝色的？答：",
            "chosen": "问：为什么天空是蓝色的？答：因为大气中的气体分子对阳光产生瑞利散射，蓝光波长较短更容易被散射到各个方向。",
            "rejected": "问：为什么天空是蓝色的？答：因为有人把天空涂成了蓝色。",
        },
    ]
    return preference_data


# ============================================================
# 3. DPO 损失函数
# ============================================================

def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    计算 DPO 损失。
    
    参数：
        policy_chosen_logps:  策略模型对"好回答"的对数概率 (batch,)
        policy_rejected_logps: 策略模型对"差回答"的对数概率 (batch,)
        ref_chosen_logps:     参考模型对"好回答"的对数概率 (batch,)
        ref_rejected_logps:   参考模型对"差回答"的对数概率 (batch,)
        beta:                 温度系数，控制偏好优化的力度
    
    返回：
        loss: 标量损失
        info: 包含各种统计信息的字典
    """
    # 计算 log ratio：当前策略 vs 参考策略
    # 好回答的 log ratio
    chosen_logratios = policy_chosen_logps - ref_chosen_logps    # (batch,)
    # 差回答的 log ratio
    rejected_logratios = policy_rejected_logps - ref_rejected_logps  # (batch,)

    # DPO 核心：好回答的 log ratio 应该大于差回答的 log ratio
    logits = beta * (chosen_logratios - rejected_logratios)  # (batch,)

    # Bradley-Terry 偏好模型的负对数似然
    # sigmoid(logits) 越接近 1 越好 → -log(sigmoid(logits)) 越小越好
    losses = -F.logsigmoid(logits)  # (batch,)

    # 平均损失
    loss = losses.mean()

    # 隐式奖励（用于可视化，等于 log ratio × β）
    chosen_rewards = beta * chosen_logratios    # 隐式奖励_好回答
    rejected_rewards = beta * rejected_logratios  # 隐式奖励_差回答

    # 收集统计信息
    info = {
        "loss": loss.item(),
        "chosen_rewards_mean": chosen_rewards.mean().item(),
        "rejected_rewards_mean": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),  # 好差回答的奖励差
        "chosen_logratios_mean": chosen_logratios.mean().item(),
        "rejected_logratios_mean": rejected_logratios.mean().item(),
    }

    return loss, info


# ============================================================
# 4. DPO 训练
# ============================================================

def train_dpo(
    policy_model: MiniTransformerLM,
    ref_model: MiniTransformerLM,
    tokenizer: SimpleTokenizer,
    preference_data: list[dict],
    num_epochs: int = 80,
    learning_rate: float = 3e-4,
    beta: float = 0.1,
):
    """
    DPO 训练循环。
    
    流程：
    1. 对每条偏好数据，计算策略模型和参考模型的 log probs
    2. 计算 DPO 损失
    3. 反向传播更新策略模型
    """
    print("=" * 60)
    print("🚀 DPO 训练开始")
    print("=" * 60)
    print(f"  偏好数据量: {len(preference_data)} 条")
    print(f"  训练轮数: {num_epochs}")
    print(f"  学习率: {learning_rate}")
    print(f"  β (温度系数): {beta}")
    print()

    # 参考模型冻结（不需要梯度）
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # 优化器只更新策略模型
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=learning_rate)

    # 记录训练统计
    history = {
        "loss": [],
        "chosen_rewards": [],
        "rejected_rewards": [],
        "reward_margin": [],
    }

    # --- 准备数据：把文本编码为 tensor ---
    chosen_ids_list = []
    rejected_ids_list = []
    for item in preference_data:
        chosen_ids_list.append(torch.tensor(tokenizer.encode(item["chosen"]), dtype=torch.long))
        rejected_ids_list.append(torch.tensor(tokenizer.encode(item["rejected"]), dtype=torch.long))

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_info = {"chosen_rewards_mean": 0, "rejected_rewards_mean": 0, "reward_margin": 0}

        for i in range(len(preference_data)):
            chosen_ids = chosen_ids_list[i].unsqueeze(0)    # (1, seq_len)
            rejected_ids = rejected_ids_list[i].unsqueeze(0)  # (1, seq_len)

            # --- 步骤 1：计算四个 log prob ---
            # 策略模型的 log prob
            policy_chosen_logps = policy_model.get_log_probs(chosen_ids[:, :-1], chosen_ids[:, 1:])
            policy_rejected_logps = policy_model.get_log_probs(rejected_ids[:, :-1], rejected_ids[:, 1:])

            # 参考模型的 log prob（不需要梯度）
            with torch.no_grad():
                ref_chosen_logps = ref_model.get_log_probs(chosen_ids[:, :-1], chosen_ids[:, 1:])
                ref_rejected_logps = ref_model.get_log_probs(rejected_ids[:, :-1], rejected_ids[:, 1:])

            # --- 步骤 2：计算 DPO 损失 ---
            loss, info = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                beta=beta,
            )

            # --- 步骤 3：反向传播 ---
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += info["loss"]
            epoch_info["chosen_rewards_mean"] += info["chosen_rewards_mean"]
            epoch_info["rejected_rewards_mean"] += info["rejected_rewards_mean"]
            epoch_info["reward_margin"] += info["reward_margin"]

        # 计算平均值
        n = len(preference_data)
        avg_loss = epoch_loss / n
        avg_chosen_reward = epoch_info["chosen_rewards_mean"] / n
        avg_rejected_reward = epoch_info["rejected_rewards_mean"] / n
        avg_margin = epoch_info["reward_margin"] / n

        history["loss"].append(avg_loss)
        history["chosen_rewards"].append(avg_chosen_reward)
        history["rejected_rewards"].append(avg_rejected_reward)
        history["reward_margin"].append(avg_margin)

        # 每 10 轮打印一次
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"好回答奖励: {avg_chosen_reward:+.4f} | "
                  f"差回答奖励: {avg_rejected_reward:+.4f} | "
                  f"奖励差: {avg_margin:+.4f}")

    print()
    print("✅ DPO 训练完成！")
    return history


# ============================================================
# 5. 对比生成：SFT（参考模型） vs DPO（策略模型）
# ============================================================

def compare_generation(
    ref_model: MiniTransformerLM,
    dpo_model: MiniTransformerLM,
    tokenizer: SimpleTokenizer,
    prompts: list[str],
    max_new_tokens: int = 40,
):
    """
    对比参考模型（SFT）和 DPO 模型对同一 prompt 的生成结果。
    """
    print("=" * 60)
    print("📊 生成对比：SFT 模型 vs DPO 模型")
    print("=" * 60)

    ref_model.eval()
    dpo_model.eval()

    for prompt in prompts:
        print(f"\n📌 Prompt: {prompt}")
        print("-" * 50)

        # 编码 prompt
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

        # 参考模型（SFT）生成
        ref_output_ids = ref_model.generate(input_ids, max_new_tokens=max_new_tokens)
        ref_output = tokenizer.decode(ref_output_ids[0].tolist())

        # DPO 模型生成
        dpo_output_ids = dpo_model.generate(input_ids, max_new_tokens=max_new_tokens)
        dpo_output = tokenizer.decode(dpo_output_ids[0].tolist())

        print(f"  🔵 SFT 模型: {ref_output[len(prompt):]}")
        print(f"  🟢 DPO 模型: {dpo_output[len(prompt):]}")


# ============================================================
# 6. 可视化隐式奖励
# ============================================================

def print_reward_analysis(history: dict):
    """打印训练过程中隐式奖励的变化趋势"""
    print("=" * 60)
    print("📈 隐式奖励变化趋势")
    print("=" * 60)
    print()

    total_epochs = len(history["loss"])
    # 取关键时间点
    checkpoints = [0, total_epochs // 4, total_epochs // 2, 3 * total_epochs // 4, total_epochs - 1]
    checkpoints = sorted(set(checkpoints))

    print(f"  {'Epoch':>8s} | {'Loss':>8s} | {'好回答奖励':>10s} | {'差回答奖励':>10s} | {'奖励差':>10s}")
    print(f"  {'-----':>8s}-+-{'-----':>8s}-+-{'---------':>10s}-+-{'---------':>10s}-+-{'---------':>10s}")

    for idx in checkpoints:
        print(f"  {idx+1:>8d} | {history['loss'][idx]:>8.4f} | "
              f"{history['chosen_rewards'][idx]:>+10.4f} | "
              f"{history['rejected_rewards'][idx]:>+10.4f} | "
              f"{history['reward_margin'][idx]:>+10.4f}")

    print()
    # 判断训练是否成功
    final_margin = history["reward_margin"][-1]
    if final_margin > 0.01:
        print(f"  ✅ 训练成功！奖励差 = {final_margin:+.4f} > 0")
        print(f"     模型已经学会区分好回答和差回答。")
        print(f"     好回答的隐式奖励 ↑↑ ({history['chosen_rewards'][-1]:+.4f})")
        print(f"     差回答的隐式奖励 ↓↓ ({history['rejected_rewards'][-1]:+.4f})")
    else:
        print(f"  ⚠️ 奖励差 = {final_margin:+.4f}，模型可能需要更多训练轮次。")


# ============================================================
# 7. DPO 损失函数直觉演示
# ============================================================

def demo_dpo_intuition():
    """用数值演示 DPO 损失的行为"""
    print("=" * 60)
    print("🔢 DPO 损失函数直觉演示")
    print("=" * 60)
    print()
    print("  当 β=0.1 时，DPO 损失如何随 Δ（好差回答的 log ratio 差）变化：")
    print()

    beta = 0.1
    print(f"  {'Δ (log ratio差)':>18s} | {'β×Δ':>10s} | {'σ(β×Δ)':>10s} | {'Loss':>10s}")
    print(f"  {'----------------':>18s}-+-{'---------':>10s}-+-{'---------':>10s}-+-{'---------':>10s}")

    for delta in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        scaled = beta * delta
        sig = torch.sigmoid(torch.tensor(scaled)).item()
        loss = -math.log(sig) if sig > 1e-10 else float("inf")
        print(f"  {delta:>+18.1f} | {scaled:>+10.4f} | {sig:>10.4f} | {loss:>10.4f}")

    print()
    print("  解释：")
    print("  - Δ < 0：模型更偏好差回答 → 损失大，会被惩罚")
    print("  - Δ = 0：模型分不清好差 → 损失 = -log(0.5) = 0.693")
    print("  - Δ > 0：模型正确偏好好回答 → 损失小，被鼓励")
    print("  - Δ 越大：模型对偏好的把握越确信 → 损失越小")


# ============================================================
# 8. 主流程
# ============================================================

def main():
    print("🧠 Day 19: DPO（Direct Preference Optimization）")
    print("   绕过强化学习，直接优化偏好")
    print()

    # 设定随机种子（保证可复现）
    torch.manual_seed(42)

    # --- 步骤 1：准备数据和分词器 ---
    print("📁 步骤 1：准备数据和分词器...")

    # 把所有文本（偏好数据里的所有文本）拼在一起建词表
    preference_data = create_preference_dataset(None)  # 先获取数据
    all_text = ""
    for item in preference_data:
        all_text += item["chosen"] + item["rejected"]

    tokenizer = SimpleTokenizer(all_text)
    print(f"  词表大小: {tokenizer.vocab_size} 个字符")
    print(f"  词表内容: {''.join(tokenizer.chars[:50])}{'...' if len(tokenizer.chars) > 50 else ''}")
    print()

    # --- 步骤 2：创建模型 ---
    print("🏗️ 步骤 2：创建模型...")
    print("  - 参考模型 π_ref（冻结，代表 SFT 后的模型）")
    print("  - 策略模型 π_θ（要训练的，初始 = π_ref）")
    print()

    # 创建参考模型
    ref_model = MiniTransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )

    # 用 SFT 数据做简单预训练（模拟 SFT 过程）
    print("  📚 先做 SFT 预训练（让模型学会基本语言能力）...")
    sft_optimizer = torch.optim.Adam(ref_model.parameters(), lr=1e-3)

    for item in preference_data:
        text_ids = torch.tensor([tokenizer.encode(item["chosen"])], dtype=torch.long)
        for _ in range(60):
            logits = ref_model(text_ids[:, :-1])
            targets = text_ids[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, tokenizer.vocab_size), targets.reshape(-1))
            sft_optimizer.zero_grad()
            loss.backward()
            sft_optimizer.step()

    print(f"  SFT 完成！最后一条数据的损失: {loss.item():.4f}")
    print()

    # 策略模型 = 参考模型的深拷贝（初始参数完全一样）
    policy_model = copy.deepcopy(ref_model)

    # --- 步骤 3：DPO 损失直觉演示 ---
    demo_dpo_intuition()
    print()

    # --- 步骤 4：DPO 训练 ---
    history = train_dpo(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        preference_data=preference_data,
        num_epochs=80,
        learning_rate=3e-4,
        beta=0.1,
    )
    print()

    # --- 步骤 5：隐式奖励分析 ---
    print_reward_analysis(history)
    print()

    # --- 步骤 6：对比生成 ---
    test_prompts = [
        "问：什么是光合作用？答：",
        "问：地球为什么是圆的？答：",
        "问：风是怎么产生的？答：",
    ]
    compare_generation(ref_model, policy_model, tokenizer, test_prompts, max_new_tokens=30)
    print()

    # --- 步骤 7：模型参数差异分析 ---
    print("=" * 60)
    print("📊 π_θ 和 π_ref 的参数差异")
    print("=" * 60)
    total_params = 0
    changed_params = 0
    max_diff = 0.0
    for (name_p, param_p), (name_r, param_r) in zip(
        policy_model.named_parameters(), ref_model.named_parameters()
    ):
        diff = (param_p - param_r).abs()
        total_params += diff.numel()
        changed_params += (diff > 1e-6).sum().item()
        max_diff = max(max_diff, diff.max().item())

    print(f"  总参数量: {total_params:,}")
    print(f"  变化的参数: {changed_params:,} ({changed_params/total_params*100:.1f}%)")
    print(f"  最大参数差异: {max_diff:.6f}")
    print()
    print("  💡 DPO 只微调了策略模型的参数，参考模型完全不变。")
    print("     变化虽小，但足以让模型学会偏好排序。")
    print()

    # --- 总结 ---
    print("=" * 60)
    print("🎉 Day 19 总结")
    print("=" * 60)
    print()
    print("  今天我们学到了：")
    print("  1. RLHF 有四大痛点：额外 RM、PPO 不稳定、显存爆炸、效率低")
    print("  2. DPO 通过数学推导'消掉'了 Reward Model")
    print("  3. DPO 损失 = 让'好回答'概率↑ + '差回答'概率↓")
    print("  4. DPO 只需要 2 个模型（策略+参考），训练和 SFT 一样简单")
    print("  5. 隐式奖励 = β × log ratio，直观衡量偏好差异")
    print()
    print("  ⭐ DPO 已成为工业界主流对齐方法（LLaMA、Mistral、Qwen 都在用）")
    print()
    print("  下节课预告: Day 20 — LoRA，用 0.1% 的参数微调大模型！")


if __name__ == "__main__":
<<<<<<< HEAD
    main()
=======
    main()
>>>>>>> de53a65 (📝 Day 19: DPO（直接偏好优化）— 绕过强化学习，直接优化偏好)
