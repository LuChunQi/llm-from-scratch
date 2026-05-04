#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N-gram 语言模型 —— 从掷骰子到文字生成
=========================================

这个脚本用最直白的方式实现了一个 N-gram 语言模型：
1. 读入语料，统计 N-gram 频次
2. 根据频次计算条件概率（带 Laplace 平滑）
3. 用学到的概率分布预测下一个词
4. 自动生成文本（基于概率采样）

运行方式：python3 ngram_lm.py
"""

import random
import math
from collections import defaultdict, Counter

# ============================================================
# 第一部分：语料准备
# ============================================================

# 我们的训练语料 —— 几段中文句子
# 真实场景中，语料库会是几百万甚至几十亿篇文章
# 这里用少量句子方便演示和理解
CORPUS = """
我 喜欢 吃 苹果 。 他 也 喜欢 吃 苹果 。
今天 天气 真 好 ， 我 想 出去 走走 。
机器 学习 是 人工智能 的 核心 技术 。
深度 学习 是 机器 学习 的 一个 分支 。
自然 语言 处理 让 计算机 理解 人类 语言 。
今天 天气 很 好 ， 适合 出去 散步 。
我 喜欢 在 好天气 里 出去 跑步 。
语言 模型 是 自然 语言 处理 的 基础 。
他 喜欢 用 Python 写 代码 。
深度 学习 模型 需要 大量 数据 来 训练 。
人工智能 正在 改变 我们 的 生活 。
机器 学习 模型 可以 处理 复杂 的 任务 。
今天 我 在 家 写 代码 ， 没有 出去 。
语言 模型 能够 预测 下 一个 词 的 概率 。
""".strip()


class NGramLanguageModel:
    """
    N-gram 语言模型
    
    核心思想：根据前 N-1 个词，预测第 N 个词的概率。
    使用 Laplace（加一）平滑处理未登录的 N-gram。
    """
    
    def __init__(self, n=2, smoothing=True):
        """
        初始化模型
        
        参数:
            n: N-gram 的 N 值。n=2 是 Bigram，n=3 是 Trigram
            smoothing: 是否使用 Laplace 平滑（默认开启）
        """
        self.n = n  # N-gram 的阶数
        self.smoothing = smoothing  # 是否平滑
        
        # ngram_counts: 存储 N-gram 及其出现次数
        # 例如 Bigram: {("我", "喜欢"): 3, ("喜欢", "吃"): 2, ...}
        self.ngram_counts = defaultdict(int)
        
        # context_counts: 存储上下文（前 N-1 个词）出现的总次数
        # 例如 Bigram: {("我",): 5, ("喜欢",): 4, ...}
        self.context_counts = defaultdict(int)
        
        # vocabulary: 不重复的词汇表
        self.vocabulary = set()
        
        # vocab_size: 词汇表大小
        self.vocab_size = 0
    
    def train(self, corpus_text):
        """
        训练模型：统计语料中的 N-gram 频次
        
        参数:
            corpus_text: 训练语料，用空格分好词的文本
        """
        # 将语料按行分割，每行是一个句子
        sentences = corpus_text.strip().split('\n')
        
        for sentence in sentences:
            # 将句子拆分成词列表
            tokens = sentence.strip().split()
            
            # 为句子添加起始标记 <s>，帮助模型学习句首的概率
            # 例如 Bigram 需要一个 <s>，Trigram 需要两个 <s>
            tokens = ['<s>'] * (self.n - 1) + tokens
            
            # 把每个词加入词汇表
            for token in tokens:
                self.vocabulary.add(token)
            
            # 滑动窗口，统计每个 N-gram
            for i in range(len(tokens) - self.n + 1):
                # 取出当前 N-gram
                ngram = tuple(tokens[i:i + self.n])
                # 取出上下文（前 N-1 个词）
                context = tuple(tokens[i:i + self.n - 1])
                
                # 计数 +1
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
        
        # 记录词汇表大小（用于 Laplace 平滑的分母）
        self.vocab_size = len(self.vocabulary)
        
        print(f"  ✅ 模型训练完成！")
        print(f"  📊 词汇表大小: {self.vocab_size} 个词")
        print(f"  📊 N-gram 数量: {len(self.ngram_counts)} 种")
        print(f"  📊 上下文数量: {len(self.context_counts)} 种")
    
    def get_probability(self, word, context):
        """
        计算条件概率 P(word | context)
        
        公式（带 Laplace 平滑）:
            P(word|context) = (count(context, word) + 1) / (count(context) + V)
            
        不平滑的版本:
            P(word|context) = count(context, word) / count(context)
        
        参数:
            word: 要预测的词
            context: 上下文（前 N-1 个词的元组）
        
        返回:
            条件概率（浮点数）
        """
        ngram = context + (word,)
        
        if self.smoothing:
            # Laplace 平滑：分子 +1，分母 + V（词汇表大小）
            # 这样即使没见过的组合，概率也不为 0
            count_ngram = self.ngram_counts.get(ngram, 0)
            count_context = self.context_counts.get(context, 0)
            probability = (count_ngram + 1) / (count_context + self.vocab_size)
        else:
            # 不平滑：直接除，可能出现 0
            count_ngram = self.ngram_counts.get(ngram, 0)
            count_context = self.context_counts.get(context, 0)
            if count_context == 0:
                return 0.0
            probability = count_ngram / count_context
        
        return probability
    
    def predict_next(self, context, top_k=5):
        """
        给定上下文，预测下一个词的概率分布（返回 top_k 个最可能的词）
        
        参数:
            context: 上下文（前 N-1 个词的元组）
            top_k: 返回概率最高的 k 个候选
        
        返回:
            列表，每项是 (词, 概率) 的元组，按概率从高到低排列
        """
        candidates = []
        
        # 遍历词汇表中的每个词，计算它的条件概率
        for word in self.vocabulary:
            # 跳过起始标记 <s>，它不应该被预测为下一个词
            if word == '<s>':
                continue
            prob = self.get_probability(word, context)
            candidates.append((word, prob))
        
        # 按概率从高到低排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        return candidates[:top_k]
    
    def generate_text(self, max_length=20, start_context=None):
        """
        生成文本：根据学到的概率分布，逐词生成
        
        每一步：
        1. 根据当前上下文，计算所有候选词的概率
        2. 按概率随机采样（不是取最高概率！这样更自然）
        3. 把选中的词加入生成序列，更新上下文
        4. 遇到句号或达到最大长度就停止
        
        参数:
            max_length: 最多生成多少个词
            start_context: 起始上下文（如果为 None，随机选一个）
        
        返回:
            生成的文本（字符串）
        """
        # 如果没有指定起始上下文，从语料中随机选一个
        if start_context is None:
            # 收集所有非起始的上下文
            valid_contexts = [ctx for ctx in self.context_counts.keys()
                             if ctx[0] != '<s>' or len(ctx) > 1]
            if valid_contexts:
                start_context = random.choice(valid_contexts)
            else:
                start_context = ('<s>',) * (self.n - 1)
        
        # 确保上下文长度正确
        context = tuple(start_context)
        if len(context) != self.n - 1:
            context = tuple(list(context)[:self.n - 1])
        
        generated = list(context)
        
        for _ in range(max_length):
            # 获取所有候选词的概率
            candidates = []
            for word in self.vocabulary:
                if word == '<s>':
                    continue
                prob = self.get_probability(word, context)
                candidates.append((word, prob))
            
            if not candidates:
                break
            
            # 按概率采样（不是贪心选最高概率）
            # 这样生成的文本更多样化，不会每次都一样
            words, probs = zip(*candidates)
            total_prob = sum(probs)
            if total_prob <= 0:
                break
            # 归一化概率（确保总和为 1）
            probs = [p / total_prob for p in probs]
            
            # 按概率分布随机选一个词
            chosen = random.choices(words, weights=probs, k=1)[0]
            generated.append(chosen)
            
            # 如果生成了句号，句子结束
            if chosen in ['。', '！', '？', '，'] and chosen == '。':
                break
            
            # 更新上下文：保留最后 N-1 个词
            context = tuple(generated[-(self.n - 1):])
        
        # 拼接生成结果，去掉起始标记
        result = [w for w in generated if w != '<s>']
        return ''.join(result)
    
    def sentence_probability(self, sentence_text):
        """
        计算一个句子的概率（对数概率，避免数值下溢）
        
        P(句子) = P(w1) * P(w2|w1) * P(w3|w1,w2) * ...
        
        用对数将乘法变加法：log(P) = sum(log(P(wi|context)))
        
        参数:
            sentence_text: 句子文本（空格分词）
        
        返回:
            对数概率（浮点数）
        """
        tokens = sentence_text.strip().split()
        tokens = ['<s>'] * (self.n - 1) + tokens
        
        log_prob = 0.0
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - self.n + 1:i])
            word = tokens[i]
            prob = self.get_probability(word, context)
            # 加一个小常数避免 log(0)
            log_prob += math.log(prob + 1e-10)
        
        return log_prob


# ============================================================
# 第二部分：演示与实验
# ============================================================

def demo():
    """主演示函数"""
    
    print("=" * 60)
    print("🎲 N-gram 语言模型演示")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # 实验 1：训练 Bigram 模型（看前 1 个词预测下一个词）
    # ----------------------------------------------------------
    print("\n" + "─" * 60)
    print("📝 实验 1：训练 Bigram 模型 (N=2)")
    print("─" * 60)
    
    bigram_model = NGramLanguageModel(n=2)
    bigram_model.train(CORPUS)
    
    # 展示一些 Bigram 的概率
    print("\n  📊 一些 Bigram 概率示例：")
    examples = [
        (("喜欢",), "吃"),
        (("喜欢",), "在"),
        (("天气",), "很"),
        (("天气",), "真"),
        (("语言",), "模型"),
        (("出去",), "走走"),
    ]
    for context, word in examples:
        prob = bigram_model.get_probability(word, context)
        print(f"     P(\"{word}\" | \"{context[0]}\") = {prob:.4f}")
    
    # ----------------------------------------------------------
    # 实验 2：预测下一个词
    # ----------------------------------------------------------
    print("\n" + "─" * 60)
    print("🔮 实验 2：预测下一个词（Top 5）")
    print("─" * 60)
    
    test_contexts = [
        ("我",),
        ("喜欢",),
        ("语言",),
        ("今天",),
    ]
    
    for ctx in test_contexts:
        print(f"\n  上下文: \"{ctx[0]}\" → 下一个词的预测：")
        top5 = bigram_model.predict_next(ctx, top_k=5)
        for i, (word, prob) in enumerate(top5, 1):
            bar = "█" * int(prob * 100)  # 可视化概率大小
            print(f"    {i}. \"{word}\" — 概率 {prob:.4f}  {bar}")
    
    # ----------------------------------------------------------
    # 实验 3：文本生成
    # ----------------------------------------------------------
    print("\n" + "─" * 60)
    print("✍️  实验 3：Bigram 模型自动生成文本")
    print("─" * 60)
    
    print("\n  生成 5 段文本：")
    for i in range(5):
        text = bigram_model.generate_text(max_length=15)
        print(f"    [{i+1}] {text}")
    
    # ----------------------------------------------------------
    # 实验 4：训练 Trigram 模型（看前 2 个词预测下一个词）
    # ----------------------------------------------------------
    print("\n" + "─" * 60)
    print("📝 实验 4：训练 Trigram 模型 (N=3)")
    print("─" * 60)
    
    trigram_model = NGramLanguageModel(n=3)
    trigram_model.train(CORPUS)
    
    # 对比 Bigram 和 Trigram 的预测
    print("\n  📊 Bigram vs Trigram 预测对比：")
    
    # 上下文 "语言" vs "自然 语言"
    print("\n  上下文: \"语言\" (Bigram) →")
    top3_bi = bigram_model.predict_next(("语言",), top_k=3)
    for word, prob in top3_bi:
        print(f"    \"{word}\" — {prob:.4f}")
    
    print("\n  上下文: \"自然 语言\" (Trigram) →")
    top3_tri = trigram_model.predict_next(("自然", "语言"), top_k=3)
    for word, prob in top3_tri:
        print(f"    \"{word}\" — {prob:.4f}")
    
    # ----------------------------------------------------------
    # 实验 5：Trigram 文本生成对比
    # ----------------------------------------------------------
    print("\n" + "─" * 60)
    print("✍️  实验 5：Trigram 模型自动生成文本")
    print("─" * 60)
    
    print("\n  生成 5 段文本（Trigram 通常更连贯）：")
    for i in range(5):
        text = trigram_model.generate_text(max_length=15)
        print(f"    [{i+1}] {text}")
    
    # ----------------------------------------------------------
    # 实验 6：句子概率比较
    # ----------------------------------------------------------
    print("\n" + "─" * 60)
    print("📊 实验 6：句子概率比较（越高越像人话）")
    print("─" * 60)
    
    sentences = [
        "我 喜欢 吃 苹果",       # 合理的句子
        "苹果 喜欢 吃 我",       # 词序混乱
        "语言 模型 是 基础",     # 合理的句子
        "基础 是 模型 语言",     # 词序混乱
    ]
    
    print(f"\n  {'句子':<30s} {'Bigram log P':>15s}")
    print(f"  {'-'*30} {'-'*15}")
    for sent in sentences:
        log_prob = bigram_model.sentence_probability(sent)
        display = sent.replace(' ', '')
        print(f"  {display:<30s} {log_prob:>15.4f}")
    
    # ----------------------------------------------------------
    # 总结
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("🎯 总结")
    print("=" * 60)
    print("""
  1. N-gram 模型通过统计"词对/词组"出现频率来预测下一个词
  2. Bigram 只看前 1 个词，Trigram 看前 2 个词
  3. 更多上下文 → 更连贯，但也更稀疏
  4. Laplace 平滑解决"没见过 = 不可能"的问题
  5. 生成文本时按概率采样，比贪心选择更自然

  N-gram 的局限：
  - 只看局部，无法捕捉长距离依赖
  - 纯统计，无法理解语义（不知道"猫"和"狗"相似）
  - 这就是为什么我们需要神经网络语言模型！

  → 下一课我们学习 Tokenizer：文字怎么变成数字？
""")
    print("=" * 60)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    # 设置随机种子，保证每次运行结果一致
    random.seed(42)
    
    # 运行演示
    demo()
