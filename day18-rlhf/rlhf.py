#!/usr/bin/env python3
"""
Day 18: RLHF（基于人类反馈的强化学习）— 让模型学会"讨好"人类

本脚本演示：
1. 偏好数据集构造：模拟人类对回答的偏好比较
2. Reward Model 训练：用 Bradley-Terry 损失训练打分器
3. PPO 训练循环：策略梯度优化（简化版）
4. KL 散度惩罚可视化
5. 对比实验：SFT 模型 vs RLHF 模型

运行方式：python3 rlhf.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import random

# ============================================================
# 0. 设备 & 随机种子
# ============================================================
device = "cpu"
torch.manual_seed(42)
random.seed(42)

print("=" * 70)
print("Day 18: RLHF（基于人类反馈的强化学习）— 让模型学会'讨好'人类")
print("=" * 70)

# ============================================================
# 1. 模型定义（复用 Day 17 的 MiniGPT 架构）
# ============================================================
print("\n" + "=" * 70)
print("第一部分：MiniGPT 模型定义（复用 SFT 架构）")
print("=" * 70)


class Head(nn.Module):
    """单头自注意力"""

    def __init__(self, d_model, head_size, block_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        scale = k.shape[-1] ** 0.5
        wei = q @ k.transpose(-2, -1) / scale
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """多头自注意力"""

    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        head_size = d_model // n_heads
        self.heads = nn.ModuleList([
            Head(d_model, head_size, block_size, dropout)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """前馈网络"""

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """一个完整的 Transformer Block"""

    def __init__(self, d_model, n_heads, block_size, dropout=0.1):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, n_heads, block_size, dropout)
        self.ffwd = FeedForward(d_model, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    """玩具级 GPT 模型"""

    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx, targets=None, loss_mask=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            B, T, V = logits.shape
            logits_flat = logits.view(B * T, V)
            targets_flat = targets.view(B * T)
            if loss_mask is not None:
                mask_flat = loss_mask.view(B * T)
                targets_masked = targets_flat.clone()
                targets_masked[mask_flat == 0] = -100
                loss = F.cross_entropy(logits_flat, targets_masked)
            else:
                loss = F.cross_entropy(logits_flat, targets_flat)
            return logits, loss
        return logits

    def get_hidden(self, idx):
        """获取隐藏状态（用于 Reward Model 和 Value Function）"""
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        return x  # (B, T, d_model)

    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        """自回归生成文本"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# 模型配置
VOCAB_SIZE = 200
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
BLOCK_SIZE = 128
DROPOUT = 0.1

print(f"模型配置: vocab_size={VOCAB_SIZE}, d_model={D_MODEL}, "
      f"n_heads={N_HEADS}, n_layers={N_LAYERS}, block_size={BLOCK_SIZE}")

# ============================================================
# 2. Tokenizer（复用 Day 17 的实现）
# ============================================================
print("\n" + "=" * 70)
print("第二部分：Tokenizer")
print("=" * 70)

SPECIAL_TOKENS = {
    "<|im_start|>": 0,
    "<|im_end|>": 1,
    "<|pad|>": 2,
}
char_to_id = {}
id_to_char = {}
for i, ch in enumerate(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789 .,!?;:'\"\n-+="
    "光合作用植物利用阳光和水中制造养分过程具体来说叶绿素吸收"
    "太阳能量将二氧化碳转化葡萄糖氧气释放出来今天气很北京晴最高"
    "温度度数回答问题什么关于春短诗歌风拂面柳丝长桃杏花开满院香"
    "燕子归来寻旧巢一池碧水映斜阳通俗好易懂黑洞太空中引力极强"
    "区域连光都逃不出想象个超级重球被压缩到小周围弯曲更大更有"
    "意思呢这个回答凑合还行吧不太是的我要把下面句子翻译成英文"
    "（）：，。？、！简洁详细专业学生"
):
    token_id = len(SPECIAL_TOKENS) + i
    char_to_id[ch] = token_id
    id_to_char[token_id] = ch

VOCAB_SIZE = max(id_to_char.keys()) + 10
UNK_ID = VOCAB_SIZE - 1


def encode(text):
    """文本 -> token ID 列表"""
    ids = []
    i = 0
    while i < len(text):
        matched = False
        for special in SPECIAL_TOKENS:
            if text[i:i + len(special)] == special:
                ids.append(SPECIAL_TOKENS[special])
                i += len(special)
                matched = True
                break
        if matched:
            continue
        ch = text[i]
        if ch in char_to_id:
            ids.append(char_to_id[ch])
        else:
            ids.append(UNK_ID)
        i += 1
    return ids


def decode(ids):
    """token ID 列表 -> 文本"""
    chars = []
    for tid in ids:
        if tid in id_to_char:
            chars.append(id_to_char[tid])
        elif tid == SPECIAL_TOKENS["<|im_start|>"]:
            chars.append("<|im_start|>")
        elif tid == SPECIAL_TOKENS["<|im_end|>"]:
            chars.append("<|im_end|>")
        elif tid == SPECIAL_TOKENS["<|pad|>"]:
            chars.append("")
        else:
            chars.append("?")
    return "".join(chars)


print("Tokenizer 就绪")

# ============================================================
# 3. 偏好数据集构造
# ============================================================
print("\n" + "=" * 70)
print("第三部分：偏好数据集构造（模拟人类偏好）")
print("=" * 70)

# 偏好数据：(prompt, chosen_response, rejected_response)
preference_data = [
    {
        "prompt": "解释什么是黑洞",
        "chosen": "黑洞是太空中引力极强的区域，连光都逃不出来。",
        "rejected": "黑洞是一种天体。它的质量很大。",
    },
    {
        "prompt": "什么是光合作用？",
        "chosen": "光合作用是植物利用阳光制造养分的过程，通俗好懂。",
        "rejected": "光合作用是过程。",
    },
    {
        "prompt": "今天的天气怎么样？",
        "chosen": "今天天气很好，晴朗温暖。",
        "rejected": "天气。",
    },
    {
        "prompt": "写一首关于春的短诗歌",
        "chosen": "春风拂面柳丝长，桃杏花开满院香。",
        "rejected": "春天来了花开了。",
    },
    {
        "prompt": "回答问题：1+1=?",
        "chosen": "1+1=2。",
        "rejected": "1+1等于不知道。",
    },
    {
        "prompt": "把句子翻译成英文：我是学生",
        "chosen": "I am a student.",
        "rejected": "I student.",
    },
    {
        "prompt": "什么是光合作用？",
        "chosen": "叶绿素吸收太阳光能量，将二氧化碳转化葡萄糖和氧气。",
        "rejected": "光合作用就是光合作用。",
    },
    {
        "prompt": "解释什么是黑洞",
        "chosen": "想象一个超级重的球被压缩到极小，周围弯曲了。",
        "rejected": "黑洞很大。",
    },
]

print(f"偏好数据: {len(preference_data)} 条")
print(f"\n示例偏好数据:")
for i in range(min(3, len(preference_data))):
    d = preference_data[i]
    print(f"\n  指令: '{d['prompt']}'")
    print(f"  [chosen]:  '{d['chosen']}'")
    print(f"  [rejected]: '{d['rejected']}'")


def build_reward_input(prompt, response, max_len=BLOCK_SIZE):
    """为 Reward Model 构建 padded 输入"""
    text = (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )
    ids = encode(text)
    ids = ids[:max_len]
    pad_len = max_len - len(ids)
    ids = ids + [SPECIAL_TOKENS["<|pad|>"]] * pad_len
    return ids


# ============================================================
# 4. Reward Model 定义与 Bradley-Terry 损失
# ============================================================
print("\n" + "=" * 70)
print("第四部分：Reward Model — Bradley-Terry 模型")
print("=" * 70)


class RewardModel(nn.Module):
    """
    奖励模型：给 (prompt, response) 打分
    结构 = Transformer backbone + value_head (-> 标量分数)
    """

    def __init__(self, backbone, d_model):
        super().__init__()
        self.backbone = backbone
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, input_ids):
        hidden = self.backbone.get_hidden(input_ids)  # (B, T, d_model)
        last_hidden = hidden[:, -1, :]  # (B, d_model)
        reward = self.value_head(last_hidden)  # (B, 1)
        return reward.squeeze(-1)  # (B,)


def bradley_terry_loss(rm, chosen_ids, rejected_ids):
    """
    Bradley-Terry 损失:
    L = -E[log sigma(r_chosen - r_rejected)]
    让 chosen 的分数高于 rejected
    """
    r_chosen = rm(chosen_ids)
    r_rejected = rm(rejected_ids)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss, r_chosen, r_rejected


print("Bradley-Terry 模型: P(chosen>rejected) = sigma(r_chosen - r_rejected)")
print("训练目标: 最小化 -log sigma(r_chosen - r_rejected)")

# ============================================================
# 5. Stage 1: 模拟 SFT（快速得到基础模型）
# ============================================================
print("\n" + "=" * 70)
print("第五部分：Stage 1 — SFT 基础模型训练（快速版）")
print("=" * 70)

sft_model = MiniGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE, DROPOUT).to(device)
total_params = sum(p.numel() for p in sft_model.parameters())
print(f"SFT 模型参数量: {total_params:,}")

all_text = (
    "光合作用是植物利用阳光制造养分的过程通俗好懂。"
    "叶绿素吸收太阳光能量将二氧化碳转化葡萄糖和氧气释放出来。"
    "今天天气很好晴朗温暖。"
    "春风拂面柳丝长桃杏花开满院香。"
    "燕子归来寻旧巢一池碧水映斜阳。"
    "1+1=2。8×7=56。"
    "黑洞是太空中引力极强的区域连光都逃不出来。"
    "想象一个超级重的球被压缩到极小周围弯曲了。"
    "I am a student."
)

all_ids = encode(all_text)
block_size = 64
pt_inputs = []
pt_targets = []
for i in range(0, len(all_ids) - block_size - 1, 4):
    pt_inputs.append(all_ids[i:i + block_size])
    pt_targets.append(all_ids[i + 1:i + block_size + 1])

pt_inputs_t = torch.tensor(pt_inputs, dtype=torch.long).to(device)
pt_targets_t = torch.tensor(pt_targets, dtype=torch.long).to(device)

optimizer = torch.optim.AdamW(sft_model.parameters(), lr=3e-4)
sft_model.train()
print("模拟 SFT 训练中...")
for step in range(300):
    idx = torch.randint(0, len(pt_inputs), (8,))
    x, y = pt_inputs_t[idx], pt_targets_t[idx]
    logits, loss = sft_model(x, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(sft_model.parameters(), 1.0)
    optimizer.step()
    if step % 100 == 0:
        print(f"  Step {step:3d} | Loss: {loss.item():.4f}")

print(f"  SFT 训练完成! 最终 Loss: {loss.item():.4f}")

# ============================================================
# 6. Stage 2: Reward Model 训练
# ============================================================
print("\n" + "=" * 70)
print("第六部分：Stage 2 — Reward Model 训练")
print("=" * 70)

rm_backbone = MiniGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE, DROPOUT).to(device)
rm_backbone.load_state_dict(sft_model.state_dict())
reward_model = RewardModel(rm_backbone, D_MODEL).to(device)

rm_params = sum(p.numel() for p in reward_model.parameters())
print(f"Reward Model 参数量: {rm_params:,}")

chosen_ids_list = []
rejected_ids_list = []
for d in preference_data:
    chosen_ids_list.append(build_reward_input(d["prompt"], d["chosen"]))
    rejected_ids_list.append(build_reward_input(d["prompt"], d["rejected"]))

chosen_tensor = torch.tensor(chosen_ids_list, dtype=torch.long).to(device)
rejected_tensor = torch.tensor(rejected_ids_list, dtype=torch.long).to(device)

rm_optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-4)
reward_model.train()
print("\n训练 Reward Model...")
for epoch in range(60):
    loss, r_chosen, r_rejected = bradley_terry_loss(
        reward_model, chosen_tensor, rejected_tensor
    )
    rm_optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(reward_model.parameters(), 1.0)
    rm_optimizer.step()

    if epoch % 15 == 0 or epoch == 59:
        correct = (r_chosen > r_rejected).float().mean().item()
        print(f"  Epoch {epoch:2d} | Loss: {loss.item():.4f} | "
              f"Acc: {correct:.1%} | "
              f"chosen_avg: {r_chosen.mean().item():.3f} | "
              f"rejected_avg: {r_rejected.mean().item():.3f}")

print("\n  Reward Model 训练完成! 学会区分好回答和差回答")

print("\nReward Model 打分验证:")
reward_model.eval()
with torch.no_grad():
    for i in range(min(3, len(preference_data))):
        d = preference_data[i]
        c_ids = torch.tensor(
            [build_reward_input(d["prompt"], d["chosen"])], dtype=torch.long
        ).to(device)
        r_ids = torch.tensor(
            [build_reward_input(d["prompt"], d["rejected"])], dtype=torch.long
        ).to(device)
        r_c = reward_model(c_ids).item()
        r_r = reward_model(r_ids).item()
        status = "OK" if r_c > r_r else "INV"
        print(f"  '{d['prompt'][:20]}...' -> "
              f"chosen: {r_c:.3f}, rejected: {r_r:.3f} [{status}]")

# ============================================================
# 7. Stage 3: PPO 训练循环
# ============================================================
print("\n" + "=" * 70)
print("第七部分：Stage 3 — PPO 强化学习优化")
print("=" * 70)

print("""
PPO 训练流程:
1. 采样 prompt, 用当前策略生成回答
2. Reward Model 给回答打分
3. 计算优势函数 A = r - V
4. PPO 裁剪目标更新策略
5. 加上 KL 散度惩罚
""")

# 策略模型（从 SFT 初始化）
policy_model = MiniGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE, DROPOUT).to(device)
policy_model.load_state_dict(copy.deepcopy(sft_model.state_dict()))

# 价值模型
value_backbone = MiniGPT(VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS, BLOCK_SIZE, DROPOUT).to(device)
value_backbone.load_state_dict(sft_model.state_dict())
value_head = nn.Sequential(
    nn.Linear(D_MODEL, D_MODEL // 2),
    nn.GELU(),
    nn.Linear(D_MODEL // 2, 1),
).to(device)


def get_value(input_ids):
    """价值函数: 估计给定序列的期望奖励"""
    hidden = value_backbone.get_hidden(input_ids)
    last_hidden = hidden[:, -1, :]
    return value_head(last_hidden).squeeze(-1)


# PPO 超参数
PPO_CLIP_EPS = 0.2
KL_COEFF = 0.1
PPO_LR = 5e-5
PPO_EPOCHS = 4

ppo_optimizer = torch.optim.AdamW(
    list(policy_model.parameters()) + list(value_head.parameters()),
    lr=PPO_LR,
)

prompts = [d["prompt"] for d in preference_data]


def encode_prompt(prompt):
    """编码 prompt 为 tensor"""
    text = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    ids = encode(text)
    return torch.tensor([ids], dtype=torch.long).to(device)


def compute_seq_log_prob(model, input_ids):
    """计算序列的 log 概率"""
    logits = model(input_ids[:, :-1])
    targets = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    return token_log_probs.sum(dim=-1)


print("开始 PPO 训练...")
reward_model.eval()

num_ppo_steps = 15
for step in range(num_ppo_steps):
    # 1. 采样 prompt, 生成回答
    batch_prompts = random.sample(prompts, min(4, len(prompts)))
    sequences = []
    old_log_probs_list = []
    reward_list = []
    value_list = []

    for prompt in batch_prompts:
        prompt_tensor = encode_prompt(prompt)
        policy_model.eval()
        with torch.no_grad():
            seq = policy_model.generate(prompt_tensor, max_new_tokens=12, temperature=0.8, top_k=20)
        sequences.append(seq)

        with torch.no_grad():
            old_lp = compute_seq_log_prob(policy_model, seq)
        old_log_probs_list.append(old_lp)

        with torch.no_grad():
            reward = reward_model(seq)
        reward_list.append(reward)

        with torch.no_grad():
            value = get_value(seq)
        value_list.append(value)

    # Padding 并合并为 batch
    max_seq_len = max(s.shape[1] for s in sequences)
    padded_seqs = []
    for s in sequences:
        pad_len = max_seq_len - s.shape[1]
        if pad_len > 0:
            s = torch.cat([s, torch.full((1, pad_len), SPECIAL_TOKENS["<|pad|>"],
                                          dtype=torch.long).to(device)], dim=1)
        padded_seqs.append(s)
    batch_seqs = torch.cat(padded_seqs, dim=0)
    old_log_probs = torch.cat(old_log_probs_list)
    rewards = torch.cat(reward_list)
    values = torch.cat(value_list)

    # 优势函数 A = r - V
    advantages = rewards - values
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 2. PPO 更新
    policy_model.train()
    value_backbone.train()
    for _ in range(PPO_EPOCHS):
        new_log_probs = compute_seq_log_prob(policy_model, batch_seqs)

        ratio = torch.exp(new_log_probs - old_log_probs.detach())

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - PPO_CLIP_EPS, 1 + PPO_CLIP_EPS) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        new_values = get_value(batch_seqs)
        value_loss = F.mse_loss(new_values, rewards.detach())

        kl_div = (new_log_probs - old_log_probs.detach()).mean().abs()

        total_loss = policy_loss + 0.5 * value_loss + KL_COEFF * kl_div

        ppo_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(policy_model.parameters()) + list(value_head.parameters()), 0.5
        )
        ppo_optimizer.step()

    avg_reward = rewards.mean().item()
    if step % 5 == 0 or step == num_ppo_steps - 1:
        print(f"  Step {step:2d}/{num_ppo_steps} | "
              f"Avg Reward: {avg_reward:.3f} | "
              f"KL: {kl_div.item():.4f} | "
              f"Loss: {total_loss.item():.4f}")

print("\n  PPO 训练完成!")

# ============================================================
# 8. KL 散度惩罚可视化
# ============================================================
print("\n" + "=" * 70)
print("第八部分：KL 散度惩罚可视化")
print("=" * 70)

print("""
KL 散度衡量两个概率分布的"距离"。
PPO 目标函数: L = L_clip + 0.5*V_loss + beta*KL(pi_theta || pi_sft)

beta 的效果:
  beta 太小 -> 模型自由变化，可能 Reward Hacking
  beta 太大 -> 模型不敢变化，等于没做 RLHF
""")

print("不同 beta 值下的 KL 惩罚效果:")
print("-" * 55)
kl_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
betas = [0.01, 0.05, 0.1, 0.5]
header = f"{'KL':>8} | " + " | ".join([f"b={b:.2f}" for b in betas])
print(header)
print("-" * 55)
for kl in kl_values:
    penalties = [b * kl for b in betas]
    row = f"{kl:>8.1f} | " + " | ".join([f"{p:>7.3f}" for p in penalties])
    print(row)

print("""
  可以看到:
  - beta=0.01 时, KL=10 的惩罚只有 0.1, 约束很弱
  - beta=0.50 时, KL=10 的惩罚高达 5.0, 约束很强
  - 实际中通常用自适应 beta: KL 太大就增大 beta
""")

# 奖励 vs KL 权衡图
print("奖励 vs KL 的权衡曲线:")
print()
print("  reward ^")
print("  |")
print("  |         /  RLHF model (high reward, controlled KL)")
print("  |       /")
print("  |     /    * Best zone: high reward + moderate KL")
print("  |   /  *")
print("  | /  *")
print("  |*   SFT model (low KL, low reward)")
print("  +-------------------------------------> KL")
print("  0")

# ============================================================
# 9. 对比实验：SFT vs RLHF
# ============================================================
print("\n" + "=" * 70)
print("第九部分：对比实验 -- SFT 模型 vs RLHF 模型")
print("=" * 70)

test_prompts = [
    "什么是光合作用？",
    "解释什么是黑洞",
    "回答问题：1+1=?",
]

print("\n用 Reward Model 对比打分:")
print("-" * 60)


def extract_response(text):
    """提取 assistant 回答部分"""
    if "<|im_start|>assistant\n" in text:
        resp = text.split("<|im_start|>assistant\n")[-1]
        resp = resp.split("<|im_end|>")[0] if "<|im_end|>" in resp else resp
        return resp.strip()
    return text


sft_total = 0.0
rlhf_total = 0.0

for prompt in test_prompts:
    prompt_tensor = encode_prompt(prompt)

    sft_model.eval()
    policy_model.eval()
    with torch.no_grad():
        sft_seq = sft_model.generate(prompt_tensor, max_new_tokens=15, temperature=0.8, top_k=20)
        rlhf_seq = policy_model.generate(prompt_tensor, max_new_tokens=15, temperature=0.8, top_k=20)

        sft_reward = reward_model(sft_seq).item()
        rlhf_reward = reward_model(rlhf_seq).item()

    sft_text = extract_response(decode(sft_seq[0].tolist()))
    rlhf_text = extract_response(decode(rlhf_seq[0].tolist()))

    winner = "RLHF wins" if rlhf_reward > sft_reward else "SFT wins"

    print(f"\n  Prompt: '{prompt}'")
    print(f"  SFT:   '{sft_text[:40]}' -> reward: {sft_reward:.3f}")
    print(f"  RLHF:  '{rlhf_text[:40]}' -> reward: {rlhf_reward:.3f}")
    print(f"  Result: {winner}")

    sft_total += sft_reward
    rlhf_total += rlhf_reward

n = len(test_prompts)
print(f"\nAverage rewards:")
print(f"  SFT:  {sft_total / n:.3f}")
print(f"  RLHF: {rlhf_total / n:.3f}")
diff = rlhf_total - sft_total
if abs(sft_total) > 1e-6:
    pct = diff / abs(sft_total) * 100
    print(f"  Change: {diff:+.3f} ({pct:+.1f}%)")
else:
    print(f"  Change: {diff:+.3f}")

# ============================================================
# 10. 总结
# ============================================================
print("\n" + "=" * 70)
print("第十部分：总结")
print("=" * 70)

print("""
RLHF 核心要点:

1. 三步走: SFT -> Reward Model -> PPO
   Stage 1: 用指令-回答数据训练基础模型
   Stage 2: 用人类偏好数据训练奖励模型 (Bradley-Terry loss)
   Stage 3: 用 PPO 强化学习优化策略模型

2. Bradley-Terry 模型:
   P(chosen > rejected) = sigma(r_chosen - r_rejected)
   把"二选一"变成连续分数

3. PPO 核心机制:
   - 裁剪 (clipping): 限制策略更新幅度
   - KL 惩罚: 防止偏离 SFT 模型太远
   - 价值函数: 估计期望奖励, 计算优势函数

4. 关键挑战:
   - 标注成本: 人类偏好数据很贵
   - Reward Hacking: 模型可能欺骗奖励模型
   - 训练不稳定: PPO 调参困难

5. 效果: RLHF 让小模型超越大模型
   InstructGPT 的 1.3B RLHF 模型比 175B GPT-3 更受偏好

下一步: DPO — 不需要强化学习的对齐方法
""")

print("Day 18 运行完毕!")
