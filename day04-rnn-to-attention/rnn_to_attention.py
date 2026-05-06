#!/usr/bin/env python3
"""
Day 4: 从 RNN 到 Attention
===========================
亲手实现 Simple RNN、LSTM 和简化版 Attention，
可视化梯度流动，理解为什么 Transformer 取代了 RNN。

运行方式：python3 rnn_to_attention.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ================================================================
# 1. Simple RNN 实现（从零手写）
# ================================================================

class SimpleRNN:
    """
    手写 RNN，不用 PyTorch 的 nn.RNN。
    
    公式：
        hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)
        yₜ = Wₕᵧ · hₜ + bᵧ
    
    用一个类比理解：
        h（隐藏状态）就像你听故事时的"当前理解"。
        每听到一个新词 x，你就把新词和之前的理解结合，更新你的理解。
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        初始化权重矩阵。
        
        Args:
            input_size:  输入向量维度（比如词向量大小）
            hidden_size: 隐藏状态维度（"脑容量"）
            output_size: 输出维度
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重（使用 Xavier 初始化，让梯度更稳定）
        # Wₓₕ: 输入 → 隐藏
        self.W_xh = torch.randn(input_size, hidden_size) * math.sqrt(2.0 / (input_size + hidden_size))
        # Wₕₕ: 隐藏 → 隐藏（"记忆"权重，核心！）
        self.W_hh = torch.randn(hidden_size, hidden_size) * math.sqrt(2.0 / (2 * hidden_size))
        # bₕ: 隐藏层偏置
        self.b_h = torch.zeros(hidden_size)
        # Wₕᵧ: 隐藏 → 输出
        self.W_hy = torch.randn(hidden_size, output_size) * math.sqrt(2.0 / (hidden_size + output_size))
        # bᵧ: 输出偏置
        self.b_y = torch.zeros(output_size)
    
    def forward(self, x_seq):
        """
        前向传播：逐个处理输入序列。
        
        Args:
            x_seq: 输入序列，形状 (seq_len, input_size)
        
        Returns:
            outputs: 每个时间步的输出，形状 (seq_len, output_size)
            hidden_states: 每个时间步的隐藏状态，形状 (seq_len, hidden_size)
        """
        seq_len = x_seq.shape[0]
        
        # 存储每个时间步的隐藏状态和输出
        hidden_states = []
        outputs = []
        
        # 初始隐藏状态为零向量
        h = torch.zeros(self.hidden_size)
        
        for t in range(seq_len):
            # 核心 RNN 公式：hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)
            h = torch.tanh(
                x_seq[t] @ self.W_xh +    # 当前输入的贡献
                h @ self.W_hh +             # 上一步隐藏状态的贡献（"记忆"）
                self.b_h                    # 偏置
            )
            # 输出：yₜ = Wₕᵧ · hₜ + bᵧ
            y = h @ self.W_hy + self.b_y
            
            hidden_states.append(h)
            outputs.append(y)
        
        return torch.stack(outputs), torch.stack(hidden_states)


# ================================================================
# 2. LSTM 实现（从零手写）
# ================================================================

class SimpleLSTM:
    """
    手写 LSTM，不用 PyTorch 的 nn.LSTM。
    
    LSTM 比 RNN 多了一个"细胞状态"（cell state）c，
    这是一条信息可以直接流过的"高速公路"。
    
    三个门控：
        遗忘门 f：决定从旧记忆中忘掉什么
        输入门 i：决定什么新信息写入记忆
        输出门 o：决定从记忆中输出什么
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        scale = math.sqrt(2.0 / (input_size + hidden_size))
        
        # 遗忘门权重：决定丢弃多少旧记忆
        self.Wf = torch.randn(input_size + hidden_size, hidden_size) * scale
        self.bf = torch.zeros(hidden_size)
        
        # 输入门权重：决定接收多少新信息
        self.Wi = torch.randn(input_size + hidden_size, hidden_size) * scale
        self.bi = torch.zeros(hidden_size)
        
        # 候选记忆权重：生成新的候选记忆
        self.Wc = torch.randn(input_size + hidden_size, hidden_size) * scale
        self.bc = torch.zeros(hidden_size)
        
        # 输出门权重：决定输出什么
        self.Wo = torch.randn(input_size + hidden_size, hidden_size) * scale
        self.bo = torch.zeros(hidden_size)
        
        # 隐藏 → 输出
        self.W_hy = torch.randn(hidden_size, output_size) * math.sqrt(2.0 / (hidden_size + output_size))
        self.b_y = torch.zeros(output_size)
    
    def forward(self, x_seq):
        """
        前向传播。
        
        Args:
            x_seq: 输入序列，形状 (seq_len, input_size)
        
        Returns:
            outputs: 每个时间步的输出
            hidden_states: 每个时间步的隐藏状态
            cell_states: 每个时间步的细胞状态
        """
        seq_len = x_seq.shape[0]
        hidden_states = []
        cell_states = []
        outputs = []
        
        # 初始状态
        h = torch.zeros(self.hidden_size)
        c = torch.zeros(self.hidden_size)  # 细胞状态！LSTM 的关键创新
        
        for t in range(seq_len):
            # 拼接当前输入和上一步隐藏状态
            combined = torch.cat([x_seq[t], h])
            
            # 遗忘门：决定丢弃多少旧记忆
            f = torch.sigmoid(combined @ self.Wf + self.bf)
            
            # 输入门：决定接收多少新信息
            i = torch.sigmoid(combined @ self.Wi + self.bi)
            
            # 候选记忆：可能记住的新内容
            c_tilde = torch.tanh(combined @ self.Wc + self.bc)
            
            # 更新细胞状态：旧记忆 × 遗忘门 + 新信息 × 输入门
            # 这是 LSTM 的核心！如果 f≈1，旧信息几乎无损保留
            c = f * c + i * c_tilde
            
            # 输出门：决定从细胞状态中输出什么
            o = torch.sigmoid(combined @ self.Wo + self.bo)
            
            # 最终隐藏状态 = 输出门 × tanh(细胞状态)
            h = o * torch.tanh(c)
            
            hidden_states.append(h)
            cell_states.append(c)
            outputs.append(h @ self.W_hy + self.b_y)
        
        return torch.stack(outputs), torch.stack(hidden_states), torch.stack(cell_states)


# ================================================================
# 3. 简化版 Self-Attention（从零手写）
# ================================================================

class SimpleSelfAttention:
    """
    手写 Self-Attention。
    
    核心思想：
        每个位置都去看序列中所有其他位置，
        根据"相关性"加权汇总信息。
    
    这就是为什么 Transformer 不需要 RNN 的递归结构：
        信息不需要一步步传递，可以一步直达！
    
    数学：
        Q (Query):  "我在找什么"
        K (Key):    "我是什么"（用来匹配 query）
        V (Value):  "我的内容"（匹配成功后返回的信息）
        
        Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
    """
    
    def __init__(self, embed_dim):
        """
        Args:
            embed_dim: 嵌入维度（Q, K, V 的维度）
        """
        self.embed_dim = embed_dim
        scale = math.sqrt(2.0 / embed_dim)
        
        # 三个投影矩阵：把输入变成 Query, Key, Value
        self.W_q = torch.randn(embed_dim, embed_dim) * scale  # Query 投影
        self.W_k = torch.randn(embed_dim, embed_dim) * scale  # Key 投影
        self.W_v = torch.randn(embed_dim, embed_dim) * scale  # Value 投影
    
    def forward(self, x_seq):
        """
        前向传播。
        
        Args:
            x_seq: 输入序列，形状 (seq_len, embed_dim)
        
        Returns:
            output: 注意力输出，形状 (seq_len, embed_dim)
            attention_weights: 注意力权重矩阵，形状 (seq_len, seq_len)
        """
        # Step 1: 线性投影得到 Q, K, V
        Q = x_seq @ self.W_q   # (seq_len, embed_dim) — 每个位置发出一个"查询"
        K = x_seq @ self.W_k   # (seq_len, embed_dim) — 每个位置提供一个"键"
        V = x_seq @ self.W_v   # (seq_len, embed_dim) — 每个位置提供一个"值"
        
        # Step 2: 计算注意力分数（Q 和 K 的点积）
        # 分数矩阵 scores[i][j] = 位置 i 对位置 j 的关注程度
        scores = Q @ K.T / math.sqrt(self.embed_dim)  # (seq_len, seq_len)
        
        # Step 3: Softmax 归一化 → 得到注意力权重（概率分布）
        attention_weights = F.softmax(scores, dim=-1)   # (seq_len, seq_len)
        
        # Step 4: 加权求和 → 每个位置融合所有位置的信息
        output = attention_weights @ V  # (seq_len, embed_dim)
        
        return output, attention_weights


# ================================================================
# 4. 实验：梯度流动对比
# ================================================================

def experiment_gradient_flow():
    """
    实验：对比 RNN 和 LSTM 在不同序列长度下的梯度流动情况。
    
    方法：对一个简单任务（序列求和）做前向传播，
         然后用 PyTorch 的 autograd 追踪梯度，
         看梯度是如何随距离衰减的。
    """
    print("=" * 70)
    print("📊 实验：RNN vs LSTM 梯度流动对比")
    print("=" * 70)
    print()
    
    hidden_size = 32
    input_size = 8
    
    # 测试不同序列长度
    seq_lengths = [10, 20, 50, 100]
    
    for seq_len in seq_lengths:
        print(f"--- 序列长度 = {seq_len} ---")
        
        # 生成随机输入，要求梯度
        x = torch.randn(seq_len, input_size, requires_grad=False)
        
        # ---- RNN 梯度测试 ----
        # 用 PyTorch 内置的 nn.RNNCell，方便追踪梯度
        rnn_cell = nn.RNNCell(input_size, hidden_size)
        h = torch.zeros(hidden_size)
        
        hidden_list = []
        for t in range(seq_len):
            h = rnn_cell(x[t], h)
            h.retain_grad()  # 对非叶张量保留梯度
            hidden_list.append(h)
        
        # 对最后一个隐藏状态求和，反向传播
        loss_rnn = hidden_list[-1].sum()
        loss_rnn.backward()
        
        # 收集每个时间步的梯度范数
        rnn_grad_norms = []
        for hh in hidden_list:
            if hh.grad is not None:
                rnn_grad_norms.append(hh.grad.norm().item())
        
        # ---- LSTM 梯度测试 ----
        lstm_cell = nn.LSTMCell(input_size, hidden_size)
        h2 = torch.zeros(hidden_size)
        c2 = torch.zeros(hidden_size)
        
        hidden_list2 = []
        cell_list2 = []
        for t in range(seq_len):
            h2, c2 = lstm_cell(x[t], (h2, c2))
            h2.retain_grad()  # 对非叶张量保留梯度
            c2.retain_grad()
            hidden_list2.append(h2)
            cell_list2.append(c2)
        
        loss_lstm = hidden_list2[-1].sum()
        loss_lstm.backward()
        
        lstm_grad_norms = []
        for hh in hidden_list2:
            if hh.grad is not None:
                lstm_grad_norms.append(hh.grad.norm().item())
        
        # 打印结果
        if rnn_grad_norms and lstm_grad_norms:
            rnn_first = rnn_grad_norms[0]
            rnn_last = rnn_grad_norms[-1]
            rnn_ratio = rnn_last / rnn_first if rnn_first > 0 else 0
            
            lstm_first = lstm_grad_norms[0]
            lstm_last = lstm_grad_norms[-1]
            lstm_ratio = lstm_last / lstm_first if lstm_first > 0 else 0
            
            print(f"  RNN  梯度：首步={rnn_first:.6f}  末步={rnn_last:.8f}  "
                  f"衰减比={rnn_ratio:.6f}")
            print(f"  LSTM 梯度：首步={lstm_first:.6f}  末步={lstm_last:.8f}  "
                  f"衰减比={lstm_ratio:.6f}")
            
            if rnn_ratio < 0.01:
                print(f"  ⚠️  RNN 梯度严重消失！末尾梯度不到首步的 1%")
            if lstm_ratio > rnn_ratio * 5:
                print(f"  ✅ LSTM 保持梯度能力是 RNN 的 {lstm_ratio/max(rnn_ratio, 1e-10):.1f} 倍")
        print()


# ================================================================
# 5. 实验：Attention 可视化
# ================================================================

def experiment_attention():
    """
    实验：可视化 Self-Attention 的权重矩阵。
    
    用一个模拟的句子，观察每个位置如何"关注"其他位置。
    """
    print("=" * 70)
    print("🔍 实验：Self-Attention 权重可视化")
    print("=" * 70)
    print()
    
    # 模拟句子："我 出生 在 北京 后来 去 了 上海"
    tokens = ["我", "出生", "在", "北京", "，", "后来", "去", "了", "上海"]
    seq_len = len(tokens)
    embed_dim = 16
    
    # 为了让结果更有意义，手动设计一些有"结构"的输入
    # 让"北京"和"出生/在"有相似的某些维度，"上海"和"去/了"有相似维度
    torch.manual_seed(42)  # 固定随机种子，让结果可复现
    x = torch.randn(seq_len, embed_dim)
    
    # 给"北京"（位置3）和"出生"（位置1）添加一些共同的信号
    x[1, 0:4] += 1.0  # "出生"
    x[3, 0:4] += 1.2  # "北京" — 和"出生"共享维度 0-3 的信号
    
    # 给"上海"（位置8）和"去"（位置6）添加一些共同的信号
    x[6, 4:8] += 1.0  # "去"
    x[8, 4:8] += 1.2  # "上海" — 和"去"共享维度 4-7 的信号
    
    attention = SimpleSelfAttention(embed_dim)
    output, weights = attention.forward(x)
    
    print(f"输入句子：{' '.join(tokens)}")
    print(f"\n注意力权重矩阵（每行代表一个 token 对所有 token 的关注度）：\n")
    
    # 打印表头
    header = "          " + "  ".join(f"{t:>5s}" for t in tokens)
    print(header)
    print("-" * len(header))
    
    for i, token in enumerate(tokens):
        row = f"{token:>5s}  │"
        for j in range(seq_len):
            w = weights[i, j].item()
            # 用不同符号表示权重大小
            if w > 0.20:
                bar = "████"
            elif w > 0.15:
                bar = "███▒"
            elif w > 0.10:
                bar = "██▒▒"
            elif w > 0.05:
                bar = "█▒▒▒"
            else:
                bar = " ·  "
            row += f" {bar}"
        print(row)
    
    print(f"\n（████ = 高关注  · = 低关注）")
    
    # 找出每个 token 最关注的其他 token
    print(f"\n每个 token 最关注谁？")
    for i, token in enumerate(tokens):
        weights_i = weights[i].clone()
        weights_i[i] = 0  # 排除自己
        most_attended = weights_i.argmax().item()
        weight_val = weights_i[most_attended].item()
        print(f"  {token} → 最关注 {tokens[most_attended]}（权重={weight_val:.3f}）")
    
    print(f"\n💡 观察要点：")
    print(f"   - Attention 让每个位置可以'一步直达'任何其他位置")
    print(f"   - 距离不再是障碍（不像 RNN 需要逐步传递）")
    print(f"   - 相关性由内容决定，而非位置决定")
    print()


# ================================================================
# 6. 实验：距离 vs 信息保持
# ================================================================

def experiment_distance():
    """
    实验：对比 RNN 和 Attention 在不同距离下的信息传递效率。
    
    模拟场景：句子开头有一个关键信息，观察它在不同位置的影响力。
    """
    print("=" * 70)
    print("📏 实验：距离 vs 信息保持能力")
    print("=" * 70)
    print()
    
    embed_dim = 16
    seq_len = 20
    
    tokens = [f"w{i}" for i in range(seq_len)]
    
    # 创建输入，w0 携带一个明显的"标记信号"
    x = torch.randn(seq_len, embed_dim) * 0.1
    x[0, :] = 2.0  # w0 有一个强烈的信号
    
    # ---- RNN 方式：信息需要一步步传递 ----
    print("RNN 信息传递（一步步传，信息会衰减）：")
    rnn = SimpleRNN(embed_dim, embed_dim, embed_dim)
    _, h_states = rnn.forward(x)
    
    # 计算每个隐藏状态和 w0 输入的相似度（余弦相似度）
    w0_signal = x[0]
    print(f"  w0 的信号强度：{w0_signal.norm():.3f}")
    for i in [1, 4, 9, 14, 19]:
        cos_sim = F.cosine_similarity(h_states[i].unsqueeze(0), w0_signal.unsqueeze(0)).item()
        print(f"  到 w{i:2d} 时，余弦相似度 = {cos_sim:+.4f}  "
              f"({'▓' * max(0, int((cos_sim + 1) * 10))}{'░' * max(0, 20 - int((cos_sim + 1) * 10))})")
    
    print()
    
    # ---- Attention 方式：信息一步直达 ----
    print("Attention 信息传递（一步直达，不衰减）：")
    attn = SimpleSelfAttention(embed_dim)
    output, weights = attn.forward(x)
    
    # 看 w0 对每个位置输出的贡献（注意力权重）
    print(f"  w0 对各位置的注意力权重：")
    for i in [1, 4, 9, 14, 19]:
        w = weights[i, 0].item()
        print(f"  w{i:2d} 关注 w0 的权重 = {w:.4f}  "
              f"({'▓' * int(w * 200)}{'░' * max(0, 20 - int(w * 200))})")
    
    print(f"\n💡 关键洞察：")
    print(f"   RNN：信息像水波纹，传播越远越弱")
    print(f"   Attention：信息像手电筒，无论多远都一照就到")
    print()


# ================================================================
# 7. 主函数
# ================================================================

if __name__ == "__main__":
    print()
    print("🧠 Day 4: 从 RNN 到 Attention")
    print("━" * 70)
    print("  今天我们通过代码回答三个问题：")
    print("  1. RNN 是怎么工作的？（带记忆的神经网络）")
    print("  2. RNN 有什么问题？（梯度消失 + 无法并行）")
    print("  3. Attention 如何解决这些问题？（一步直达）")
    print("━" * 70)
    print()
    
    # 实验 1：梯度流动对比
    experiment_gradient_flow()
    
    # 实验 2：Attention 可视化
    experiment_attention()
    
    # 实验 3：距离 vs 信息保持
    experiment_distance()
    
    print("=" * 70)
    print("🎉 实验完成！")
    print("=" * 70)
    print()
    print("📖 今日要点回顾：")
    print()
    print("  1. RNN 通过隐藏状态 h 在时间步间传递信息")
    print("     → 像传话游戏，信息逐步传递必然衰减")
    print()
    print("  2. LSTM 通过门控机制缓解梯度消失")
    print("     → 像在传话游戏里加了笔记本，重要的信息可以抄下来")
    print("     → 但仍然是串行的，无法并行计算")
    print()
    print("  3. Attention 让每个位置直接关注所有其他位置")
    print("     → 像手电筒，无论多远一照就到")
    print("     → 可以完全并行计算，训练速度快几个数量级")
    print()
    print("  4. Transformer = Attention + 一些工程技巧")
    print("     → 抛弃了 RNN 的串行结构，开启了 LLM 时代")
    print()
    print("🚀 明天 Day 5 预告：Transformer 完整架构 + 骨架代码搭建！")
