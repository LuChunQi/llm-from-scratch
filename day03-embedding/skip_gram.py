#!/usr/bin/env python3
"""
Day 3: Skip-gram Word2Vec 手写实现
===================================
用 PyTorch 从零实现 Skip-gram + 负采样，训练词嵌入并可视化。

核心思想：
    给定一个中心词，预测它周围可能出现的上下文词。
    训练完成后，权重矩阵的每一行就是一个词的向量表示。

运行: python3 skip_gram.py
依赖: pip install torch matplotlib scikit-learn numpy
"""

import math
import random
from collections import Counter

import numpy as np

# ===== 尝试导入 PyTorch =====
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:
    print("❌ 需要安装 PyTorch: pip install torch")
    exit(1)

# 设置随机种子，确保可复现
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ===================================================================
# 1. 训练语料 — 一个精心设计的小型英文句子集合
#    包含了有语义关系的词：king/queen, cat/dog, happy/sad 等
# ===================================================================
CORPUS = """
the king sat on his throne and ruled the kingdom wisely.
the queen sat beside the king and governed the kingdom.
a man walked through the castle gates with his sword.
a woman walked through the castle garden with her flowers.
the prince is the son of the king and the queen.
the princess is the daughter of the king and the queen.
the cat sat on the mat and watched the little mouse.
the dog sat on the porch and chased the little cat.
cats and dogs are popular pets in many homes.
the happy king smiled at his beautiful queen.
the sad queen wept for the lost kingdom.
the brave knight fought the dragon to save the kingdom.
the wise man read books in the quiet library.
the clever woman solved the puzzle in no time.
the little cat chased the mouse around the house.
the big dog guarded the castle through the night.
the king gave gold to the brave knight.
the queen gave flowers to the clever princess.
the man and his dog walked through the forest.
the woman and her cat sat by the fireplace.
the kingdom was peaceful when the wise king ruled.
the castle was beautiful when the good queen lived there.
he was a strong man who fought many battles.
she was a kind woman who helped many people.
the cat and the dog played together in the garden.
the prince will become king when his father dies.
the princess will become queen when her mother steps down.
the happy man danced in the beautiful garden.
the sad woman cried in the dark castle.
the knight rode his horse to the far kingdom.
""".strip().lower()

print("=" * 60)
print("📚 Day 3: Skip-gram Word2Vec 手写实现")
print("=" * 60)
print()

# ===================================================================
# 2. 数据预处理：分词 → 构建词表 → 建立索引
# ===================================================================
print("📝 第一步：数据预处理...")

# 按空格分词
tokens = CORPUS.split()
print(f"   语料总词数: {len(tokens)}")

# 统计词频
word_counts = Counter(tokens)
print(f"   不同词数: {len(word_counts)}")

# 过滤低频词（出现 < 2 次的忽略，减少噪声）
MIN_COUNT = 2
vocab = {w: c for w, c in word_counts.items() if c >= MIN_COUNT}
print(f"   过滤后词数 (频次 >= {MIN_COUNT}): {len(vocab)}")

# 构建词 → 索引的映射
word2idx = {w: i for i, w in enumerate(sorted(vocab.keys()))}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx)

print(f"   词表大小: {vocab_size}")
print(f"   示例词: {list(word2idx.keys())[:15]}")
print()

# ===================================================================
# 3. 构造训练样本：(中心词索引, 上下文词索引) 对
#    Skip-gram 的核心：滑动窗口，窗口内的词互为上下文
# ===================================================================
print("🔧 第二步：构造 Skip-gram 训练样本...")

WINDOW_SIZE = 2  # 上下文窗口大小（左右各看 2 个词）

# 只保留词表内的 token 的索引
token_indices = [word2idx[w] for w in tokens if w in word2idx]

training_pairs = []  # 存 (center_idx, context_idx) 对

for i, center_idx in enumerate(token_indices):
    # 在窗口范围内取上下文词
    for j in range(-WINDOW_SIZE, WINDOW_SIZE + 1):
        if j == 0:
            continue  # 跳过中心词自己
        context_pos = i + j
        if 0 <= context_pos < len(token_indices):
            context_idx = token_indices[context_pos]
            training_pairs.append((center_idx, context_idx))

print(f"   窗口大小: {WINDOW_SIZE}（左右各 {WINDOW_SIZE} 个词）")
print(f"   训练样本数: {len(training_pairs)}")
print(f"   示例: center={idx2word[training_pairs[0][0]]}, "
      f"context={idx2word[training_pairs[0][1]]}")
print()

# ===================================================================
# 4. 负采样表 — 按词频的 0.75 次方构建采样分布
#    高频词被采到更多，但做了平滑（0.75 次方），避免太偏
# ===================================================================
print("🎲 第三步：构建负采样表...")

NEG_SAMPLES = 5  # 每个正样本配几个负样本

# 按词频的 0.75 次方计算采样权重（Mikolov 论文的经验值）
freq_array = np.array([vocab[idx2word[i]] for i in range(vocab_size)])
freq_powered = np.power(freq_array, 0.75)  # 频率的 0.75 次方
neg_sampling_prob = freq_powered / freq_powered.sum()  # 归一化为概率

print(f"   负样本数: {NEG_SAMPLES}（每个正样本配 {NEG_SAMPLES} 个负样本）")
print(f"   采样分布: 已按 频率^0.75 平滑")
print()

# ===================================================================
# 5. 定义 Skip-gram 模型
#    两个 Embedding 层：
#    - center_embeddings: 中心词的向量矩阵 W (vocab_size × embed_dim)
#    - context_embeddings: 上下文词的向量矩阵 W' (vocab_size × embed_dim)
# ===================================================================
print("🧠 第四步：定义 Skip-gram 模型...")

EMBED_DIM = 20  # 嵌入维度（实际项目通常 100~300，这里小一点方便训练）


class SkipGramModel(nn.Module):
    """
    Skip-gram Word2Vec 模型 + 负采样

    原理：
        对于每个 (center, context) 正样本对：
        - 正样本损失：-log(sigmoid(u_context · v_center))
        - 负样本损失：-Σ log(sigmoid(-u_neg · v_center))

        训练目标是让真实的上下文词与中心词的点积尽量大，
        随机负样本与中心词的点积尽量小。
    """

    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        # 中心词嵌入矩阵（这就是最终要的词向量！）
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        # 上下文词嵌入矩阵
        self.context_embeddings = nn.Embedding(vocab_size, embed_dim)

        # Xavier 初始化，让初始向量分布更合理
        init_range = 0.5 / embed_dim
        self.center_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)

    def forward(self, center_ids, context_ids, negative_ids):
        """
        前向传播 + 计算损失

        参数:
            center_ids: (batch_size,) 中心词索引
            context_ids: (batch_size,) 正样本上下文词索引
            negative_ids: (batch_size, n_neg) 负样本索引

        返回:
            loss: 标量，负采样损失
        """
        # 取出中心词向量 (batch, embed_dim)
        v_center = self.center_embeddings(center_ids)
        # 取出正样本上下文向量 (batch, embed_dim)
        u_context = self.context_embeddings(context_ids)
        # 取出负样本向量 (batch, n_neg, embed_dim)
        u_negatives = self.context_embeddings(negative_ids)

        # === 正样本得分 ===
        # 点积: (batch,) — 每个中心词和对应正样本的相似度
        pos_score = torch.sum(v_center * u_context, dim=1)
        pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-10)  # 加小量防 log(0)

        # === 负样本得分 ===
        # v_center: (batch, 1, embed) → 和 (batch, n_neg, embed) 做点积
        neg_score = torch.bmm(u_negatives, v_center.unsqueeze(2)).squeeze(2)
        # (batch, n_neg) — 每个中心词和每个负样本的相似度
        neg_loss = -torch.sum(
            torch.log(torch.sigmoid(-neg_score) + 1e-10), dim=1
        )

        # 总损失 = 正样本损失 + 负样本损失
        return (pos_loss + neg_loss).mean()


# 实例化模型
model = SkipGramModel(vocab_size, EMBED_DIM)
print(f"   模型参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"   嵌入维度: {EMBED_DIM}")
print()

# ===================================================================
# 6. 训练循环
# ===================================================================
print("🚀 第五步：开始训练...")
print("   " + "-" * 50)

EPOCHS = 300          # 训练轮数
BATCH_SIZE = 64       # 批大小
LEARNING_RATE = 0.025 # 学习率（Word2Vec 常用值）

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
# 转换训练数据为 tensor
all_centers = torch.tensor([p[0] for p in training_pairs], dtype=torch.long)
all_contexts = torch.tensor([p[1] for p in training_pairs], dtype=torch.long)

for epoch in range(EPOCHS):
    # 打乱数据
    perm = torch.randperm(len(training_pairs))
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(training_pairs), BATCH_SIZE):
        # 取一个 batch 的数据
        batch_perm = perm[start:start + BATCH_SIZE]
        center_batch = all_centers[batch_perm]
        context_batch = all_contexts[batch_perm]
        bs = center_batch.shape[0]

        # 为每个样本采样 NEG_SAMPLES 个负样本
        # 用我们预先算好的 neg_sampling_prob
        neg_ids = torch.tensor(
            np.random.choice(vocab_size, size=(bs, NEG_SAMPLES),
                             replace=True, p=neg_sampling_prob),
            dtype=torch.long
        )

        # 前向 + 反向
        loss = model(center_batch, context_batch, neg_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)

    # 每 50 轮打印一次进度
    if (epoch + 1) % 50 == 0 or epoch == 0:
        print(f"   Epoch {epoch + 1:>3d}/{EPOCHS} | Loss: {avg_loss:.4f}")

print("   " + "-" * 50)
print(f"   ✅ 训练完成！最终 Loss: {avg_loss:.4f}")
print()

# ===================================================================
# 7. 提取词向量 & 计算相似度
# ===================================================================
print("📊 第六步：词向量分析...")

# 取出中心词嵌入矩阵作为最终词向量
embeddings = model.center_embeddings.weight.data.numpy()


def cosine_similarity(v1, v2):
    """计算两个向量的余弦相似度（范围 -1 到 1）"""
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot / (norm1 * norm2 + 1e-10)


def find_most_similar(word, top_k=5):
    """找到和指定词最相似的 top_k 个词"""
    if word not in word2idx:
        print(f"   ⚠️ '{word}' 不在词表中")
        return []
    idx = word2idx[word]
    vec = embeddings[idx]

    # 计算和所有词的余弦相似度
    sims = []
    for i in range(vocab_size):
        if i == idx:
            continue
        sim = cosine_similarity(vec, embeddings[i])
        sims.append((idx2word[i], sim))

    # 按相似度降序排列
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


# 展示一些有趣词的近邻
test_words = ["king", "queen", "man", "woman", "cat", "dog", "happy", "castle"]
print()
for word in test_words:
    if word in word2idx:
        neighbors = find_most_similar(word, top_k=3)
        neighbor_str = ", ".join([f"{w}({s:.3f})" for w, s in neighbors])
        print(f"   🔍 '{word}' 的近邻: {neighbor_str}")
print()

# ===================================================================
# 8. 经典词向量运算：king - man + woman ≈ ?
# ===================================================================
print("🧮 第七步：词向量算术运算...")
print()


def analogy(word_a, word_b, word_c, top_k=5):
    """
    经典词向量类比: word_a - word_b + word_c ≈ ?

    例如: king - man + woman ≈ queen
    逻辑: king 相对于 man 的"差异"加到 woman 上，应该得到 queen
    """
    # 检查词是否在词表中
    for w in [word_a, word_b, word_c]:
        if w not in word2idx:
            print(f"   ⚠️ '{w}' 不在词表中")
            return

    # 计算目标向量: a - b + c
    vec_a = embeddings[word2idx[word_a]]
    vec_b = embeddings[word2idx[word_b]]
    vec_c = embeddings[word2idx[word_c]]
    target = vec_a - vec_b + vec_c

    # 排除 a, b, c 三个词，找最近的
    exclude = {word2idx[word_a], word2idx[word_b], word2idx[word_c]}
    sims = []
    for i in range(vocab_size):
        if i in exclude:
            continue
        sim = cosine_similarity(target, embeddings[i])
        sims.append((idx2word[i], sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    results = sims[:top_k]

    result_str = ", ".join([f"{w}({s:.3f})" for w, s in results])
    print(f"   {word_a} - {word_b} + {word_c} ≈ {result_str}")
    return results


# 测试几个经典类比
analogy("king", "man", "woman")
analogy("queen", "woman", "man")
analogy("king", "queen", "woman")
analogy("cat", "dog", "king")
print()

# ===================================================================
# 9. PCA 降维可视化
# ===================================================================
print("🎨 第八步：PCA 降维可视化...")

try:
    from sklearn.decomposition import PCA

    # 选一些有代表性的词来做可视化
    visual_words = [
        "king", "queen", "prince", "princess", "man", "woman",
        "cat", "dog", "castle", "kingdom",
        "happy", "sad", "brave", "wise", "strong",
    ]
    visual_words = [w for w in visual_words if w in word2idx]

    # 取这些词的向量
    visual_indices = [word2idx[w] for w in visual_words]
    visual_vectors = embeddings[visual_indices]

    # PCA 降到 2 维
    pca = PCA(n_components=2)
    coords = pca.fit_transform(visual_vectors)

    print(f"   降维词数: {len(visual_words)}")
    print(f"   PCA 解释方差比: {pca.explained_variance_ratio_[0]:.2%}, "
          f"{pca.explained_variance_ratio_[1]:.2%}")
    print()
    print("   词向量 2D 坐标:")
    print("   " + "-" * 40)
    for word, (x, y) in zip(visual_words, coords):
        print(f"   {word:>10s}  →  ({x:+.3f}, {y:+.3f})")
    print("   " + "-" * 40)

    # 尝试画图保存
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无头模式
        import matplotlib.pyplot as plt

        # 设置中文字体（如果没有中文字体也没关系，图表标题用英文）
        plt.figure(figsize=(12, 8))
        plt.rcParams["font.size"] = 14

        # 绘制散点图
        plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=100, c="steelblue")

        # 标注每个词
        for i, word in enumerate(visual_words):
            plt.annotate(
                word, (coords[i, 0], coords[i, 1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=12, fontweight="bold"
            )

        plt.title("Word2Vec Skip-gram Embeddings (PCA 2D Projection)",
                  fontsize=16)
        plt.xlabel("PC1", fontsize=13)
        plt.ylabel("PC2", fontsize=13)
        plt.grid(True, alpha=0.3)

        output_path = "word2vec_visualization.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"\n   📊 可视化图已保存: {output_path}")

    except Exception as e:
        print(f"   ⚠️ matplotlib 画图失败: {e}")
        print("   （不影响主要结果，词向量分析已输出）")

except ImportError:
    print("   ⚠️ scikit-learn 未安装，跳过 PCA 可视化")
    print("   安装: pip install scikit-learn")

# ===================================================================
# 10. 保存词向量
# ===================================================================
print()
print("💾 第九步：保存词向量...")

output_file = "word_vectors.txt"
with open(output_file, "w") as f:
    f.write(f"{vocab_size} {EMBED_DIM}\n")
    for word in sorted(word2idx.keys()):
        idx = word2idx[word]
        vec_str = " ".join([f"{v:.6f}" for v in embeddings[idx]])
        f.write(f"{word} {vec_str}\n")

print(f"   ✅ 词向量已保存: {output_file}")
print(f"   格式: Word2Vec text 格式（可用 gensim 加载）")

# ===================================================================
# 总结
# ===================================================================
print()
print("=" * 60)
print("🎉 Day 3 完成！今天你学到了:")
print("=" * 60)
print()
print("   1. ❌ One-hot 的问题：高维、稀疏、无语义")
print("   2. ✅ 词嵌入的思路：低维、稠密、语义相关")
print("   3. 📖 分布式假设：上下文相似的词语义相近")
print("   4. 🏗️ Skip-gram 架构：中心词预测上下文")
print("   5. 🎲 负采样技巧：用少量负样本近似 softmax")
print("   6. 🧮 词向量运算：king - man + woman ≈ queen")
print()
print("   📂 输出文件:")
print(f"      - {output_file} (词向量文本)")
print(f"      - word2vec_visualization.png (可视化图)")
print()
print("   🔮 明天预告: Day 4 — 从 RNN 到 Attention")
print("      为什么循环网络处理不了长文本？")
print("=" * 60)
