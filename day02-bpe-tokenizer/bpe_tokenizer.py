#!/usr/bin/env python3
"""
Day 2 - 手写 BPE (Byte Pair Encoding) 分词器
============================================
从零实现一个完整的 BPE 分词器，包括训练、编码、解码三步。

BPE 的核心思想：不断合并出现频率最高的相邻 token 对，
直到达到目标词表大小。

运行方式：python3 bpe_tokenizer.py
依赖：仅使用 Python 标准库（collections）
"""

from collections import Counter
import re


# ============================================================
# 第一部分：BPE 训练器 —— 从语料中学习合并规则
# ============================================================

class BPETrainer:
    """
    BPE 训练器：从训练语料中学习合并规则。
    
    工作流程：
    1. 把训练文本拆成单词（用空格分割）
    2. 每个单词表示为字符序列（用 Unicode 码点）
    3. 统计相邻字符对的频率
    4. 合并频率最高的字符对
    5. 重复 3-4 直到达到目标合并次数
    
    内部用一个自增 ID 表示每次合并产生的新 token，
    避免嵌套元组的问题。
    """
    
    # 特殊 ID：空格 token
    SPACE_ID = 0
    
    def __init__(self, num_merges=100):
        """
        初始化训练器。
        
        参数:
            num_merges: 合并次数，决定了词表在基础字符之上增加多少
        """
        self.num_merges = num_merges
        self.merges = []           # 合并规则列表：(token_a_id, token_b_id) → new_id
        self.merge_map = {}        # 快速查找：(a_id, b_id) → merged_id
        self.vocab = {}            # token_id → 字符串表示
        self.char_to_id = {}       # 字符码点 → token_id
        
    def _flatten_token(self, token_id):
        """获取一个 token 对应的字符串。"""
        return self.vocab.get(token_id, f"<{token_id}>")
    
    def _get_pair_counts(self, word_freqs):
        """
        统计所有相邻 token 对的出现频率（加权）。
        
        参数:
            word_freqs: dict，key=单词的token元组，value=出现次数
            
        返回:
            Counter，key=(token_a_id, token_b_id)，value=加权频率
        """
        pair_counts = Counter()
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += freq
        return pair_counts
    
    def _merge_pair(self, word_freqs, pair, new_id):
        """
        在所有单词中，把指定的 token 对合并为一个新的 token ID。
        
        参数:
            word_freqs: 当前单词频率表
            pair: (token_a_id, token_b_id) 要合并的对
            new_id: 合并后分配的新 ID
            
        返回:
            新的 word_freqs 字典
        """
        new_word_freqs = {}
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                # 检查当前位置是否匹配要合并的 pair
                if (i < len(word) - 1 
                    and word[i] == pair[0] 
                    and word[i + 1] == pair[1]):
                    new_word.append(new_id)
                    i += 2  # 跳过已合并的两个 token
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        return new_word_freqs
    
    def train(self, text):
        """
        从文本中训练 BPE 合并规则。
        
        参数:
            text: 训练文本（字符串）
        """
        print("=" * 60)
        print("🏋️  BPE 训练开始")
        print("=" * 60)
        
        # ---- Step 1：把文本拆成单词，统计频率 ----
        words = re.findall(r'\S+', text)
        word_freqs = Counter(words)
        
        print(f"\n📝 训练语料统计：")
        print(f"   总单词数（去重前）: {len(words)}")
        print(f"   不同单词数: {len(word_freqs)}")
        
        # ---- Step 2：分配 token ID ----
        # 空格先分配 ID=0
        next_id = 1
        
        # 收集所有出现的字符，分配 ID
        all_chars = set()
        for word in word_freqs.keys():
            for c in word:
                all_chars.add(ord(c))
        
        # 按码点排序，给每个字符分配 ID
        for cp in sorted(all_chars):
            self.char_to_id[cp] = next_id
            self.vocab[next_id] = chr(cp)
            next_id += 1
        
        # 把空格也加入词表
        self.vocab[self.SPACE_ID] = " "
        
        print(f"   初始词表大小（不同字符数）: {len(self.char_to_id)}")
        print(f"   目标合并次数: {self.num_merges}")
        print()
        
        # ---- Step 3：把每个单词转成 token ID 序列 ----
        byte_word_freqs = {}
        for word, freq in word_freqs.items():
            token_word = tuple(self.char_to_id[ord(c)] for c in word)
            byte_word_freqs[token_word] = freq
        
        # ---- Step 4：迭代合并 ----
        for i in range(self.num_merges):
            # 统计当前所有相邻 token 对的频率
            pair_counts = self._get_pair_counts(byte_word_freqs)
            
            if not pair_counts:
                print(f"   ⚠️  没有更多可合并的 token 对了，提前停止")
                break
            
            # 找到频率最高的 pair
            best_pair = max(pair_counts, key=pair_counts.get)
            best_count = pair_counts[best_pair]
            
            # 分配新 ID
            new_id = next_id
            next_id += 1
            
            # 记录合并规则
            self.merges.append((best_pair, new_id))
            self.merge_map[best_pair] = new_id
            
            # 新 token 的字符串 = 两个 token 的字符串拼接
            str_a = self.vocab[best_pair[0]]
            str_b = self.vocab[best_pair[1]]
            self.vocab[new_id] = str_a + str_b
            
            # 执行合并
            byte_word_freqs = self._merge_pair(byte_word_freqs, best_pair, new_id)
            
            # 打印进度
            if (i + 1) % 20 == 0 or i < 10:
                print(f"   合并 #{i+1:3d}: '{str_a}' + '{str_b}' → '{str_a}{str_b}' (频率: {best_count})")
        
        print(f"\n✅ 训练完成！共学习了 {len(self.merges)} 条合并规则")
        print(f"   最终词表大小: {len(self.vocab)}")


# ============================================================
# 第二部分：BPE 分词器 —— 用学到的规则进行编码/解码
# ============================================================

class BPETokenizer:
    """
    BPE 分词器：用训练好的合并规则进行文本编码和解码。
    
    编码：对新文本按顺序回放训练时学到的合并规则。
    解码：把 token ID 还原回文本。
    """
    
    def __init__(self, trainer):
        """
        用训练好的 BPETrainer 初始化分词器。
        """
        self.merges = trainer.merges           # [(pair, new_id), ...]
        self.merge_map = trainer.merge_map     # {(a_id, b_id): merged_id}
        self.vocab = trainer.vocab             # token_id → string
        self.char_to_id = trainer.char_to_id   # codepoint → id
        self.space_id = trainer.SPACE_ID
    
    def _chars_to_ids(self, word):
        """把一个单词的字符转成初始 token ID 列表。"""
        ids = []
        for c in word:
            cp = ord(c)
            if cp in self.char_to_id:
                ids.append(self.char_to_id[cp])
            else:
                # 未见过的字符，分配临时 ID（实际中应该处理 OOV）
                ids.append(cp)
        return ids
    
    def _apply_merges(self, tokens):
        """
        对 token 列表按顺序应用所有合并规则。
        
        关键：必须按照训练时的合并顺序逐一尝试！
        """
        for (merge_pair, new_id) in self.merges:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if (i < len(tokens) - 1 
                    and tokens[i] == merge_pair[0] 
                    and tokens[i + 1] == merge_pair[1]):
                    new_tokens.append(new_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def encode(self, text):
        """
        把文本编码为 token ID 列表。
        
        策略：按空格分割文本，每个单词独立编码，空格作为单独 token。
        这和 GPT-2 的做法类似。
        
        参数:
            text: 要编码的文本
            
        返回:
            (token_ids, token_strings) 元组
        """
        all_token_ids = []
        all_token_strs = []
        
        # 用正则分割，保留空格位置
        # 按 "单词 + 空格" 分组
        parts = text.split(' ')
        
        for idx, part in enumerate(parts):
            if not part:
                continue
            
            # Step 1: 字符 → 初始 token ID
            tokens = self._chars_to_ids(part)
            
            # Step 2: 应用合并规则
            tokens = self._apply_merges(tokens)
            
            # Step 3: 收集结果
            for tid in tokens:
                all_token_ids.append(tid)
                all_token_strs.append(self.vocab.get(tid, f"<{tid}>"))
            
            # 在单词之间加空格 token
            if idx < len(parts) - 1:
                all_token_ids.append(self.space_id)
                all_token_strs.append(" ")
        
        return all_token_ids, all_token_strs
    
    def decode(self, token_ids):
        """
        把 token ID 列表解码回文本。
        """
        parts = []
        for tid in token_ids:
            if tid in self.vocab:
                parts.append(self.vocab[tid])
            else:
                parts.append(f"<UNK:{tid}>")
        return "".join(parts)
    
    def tokenize_and_show(self, text):
        """
        编码文本并漂亮地展示结果。
        """
        print(f"\n{'─' * 60}")
        print(f"📄 原文: 「{text}」")
        print(f"{'─' * 60}")
        
        token_ids, token_strs = self.encode(text)
        
        print(f"🔤 分词结果 ({len(token_strs)} 个 tokens):")
        print(f"   ", end="")
        for ts in token_strs:
            # 用 ▁ 代替空格，更直观
            display = "▁" if ts == " " else ts
            print(f"[{display}]", end=" ")
        print()
        
        print(f"🔢 Token IDs:")
        print(f"   {token_ids}")
        
        # 解码验证
        decoded = self.decode(token_ids)
        print(f"🔄 解码验证: 「{decoded}」")
        match = "✅ 一致" if decoded == text else "❌ 不一致"
        print(f"   {match}")
        
        return token_ids, token_strs


# ============================================================
# 第三部分：演示 —— 训练和测试
# ============================================================

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Day 2 - BPE Tokenizer 从零实现                         ║")
    print("║   手写字节对编码：训练 → 编码 → 解码                      ║")
    print("╚══════════════════════════════════════════════════════════╝")
    
    # ---- 准备训练语料 ----
    training_text = """
    the cat sat on the mat the cat ate the rat the rat sat on the hat
    the cat is cute the rat is fat the mat is flat the hat is that
    lower lower lowest lowest newer newer newest newest
    hello hello world world hello world
    chat chatgpt chatbot chatting chatter
    你好世界你好中国你好北京上海广州深圳
    深度学习机器学习自然语言处理大语言模型
    transformer attention embedding tokenization
    the quick brown fox jumps over the lazy dog
    she sells sea shells by the sea shore
    peter piper picked a peck of pickled peppers
    我喜欢学习自然语言处理和深度学习
    大语言模型使用transformer架构
    chatgpt是一个大语言模型应用
    tokenization是自然语言处理的第一步
    """
    
    print(f"\n📚 训练语料长度: {len(training_text)} 字符")
    
    # ---- 训练 BPE ----
    trainer = BPETrainer(num_merges=50)
    trainer.train(training_text)
    
    # ---- 创建分词器 ----
    tokenizer = BPETokenizer(trainer)
    
    # ---- 测试编码/解码 ----
    print("\n" + "=" * 60)
    print("🧪 测试编码/解码")
    print("=" * 60)
    
    test_texts = [
        "the cat",                    # 英文简单词
        "lower lowest",               # BPE 经典案例
        "chat chatbot",               # 共享前缀
        "hello world",                # 常见组合
        "tokenization",               # 长单词
        "你好世界",                    # 中文
        "深度学习",                    # 中文专业术语
        "chatgpt",                    # 混合新词
        "the cat sat on the mat",     # 完整句子
        "大语言模型",                   # 中文长词
    ]
    
    for text in test_texts:
        tokenizer.tokenize_and_show(text)
    
    # ---- 展示合并规则的 Top 10 ----
    print("\n" + "=" * 60)
    print("📊 合并规则 Top 10（按学习顺序）")
    print("=" * 60)
    
    for i, (pair, new_id) in enumerate(trainer.merges[:10]):
        str_a = trainer.vocab[pair[0]]
        str_b = trainer.vocab[pair[1]]
        str_merged = trainer.vocab[new_id]
        # 显示时用 ▁ 表示空格
        disp_a = "▁" if str_a == " " else str_a
        disp_b = "▁" if str_b == " " else str_b
        disp_m = "▁" if str_merged == " " else str_merged
        print(f"   规则 #{i+1:2d}: '{disp_a}' + '{disp_b}' → '{disp_m}'")
    
    # ---- 展示词表统计 ----
    print("\n" + "=" * 60)
    print("📈 词表统计")
    print("=" * 60)
    
    single_char_tokens = [tid for tid, s in trainer.vocab.items() if len(s) == 1]
    merged_tokens = [tid for tid, s in trainer.vocab.items() if len(s) > 1]
    
    print(f"   单字符 token: {len(single_char_tokens)} 个")
    print(f"   合并 token:   {len(merged_tokens)} 个")
    print(f"   词表总大小:   {len(trainer.vocab)} 个")
    
    # 展示一些有趣的合并 token
    print(f"\n   🔍 部分合并 token 示例：")
    interesting = [(tid, s) for tid, s in trainer.vocab.items() if len(s) > 1][:15]
    for tid, s in interesting:
        print(f"      ID {tid:3d} → '{s}'")
    
    # ---- 压缩率展示 ----
    print("\n" + "=" * 60)
    print("🗜️  压缩效果展示")
    print("=" * 60)
    
    examples = [
        ("the cat sat on the mat", "英文句子"),
        ("tokenization", "英文长单词"),
        ("你好世界", "中文短句"),
        ("tokenization是自然语言处理的第一步", "中英混合"),
    ]
    
    for demo_text, desc in examples:
        char_count = len(demo_text)
        token_ids, token_strs = tokenizer.encode(demo_text)
        token_count = len(token_strs)
        ratio = char_count / token_count if token_count > 0 else 0
        
        print(f"\n   [{desc}] 「{demo_text}」")
        print(f"   字符数: {char_count}  |  Token 数: {token_count}  |  压缩率: {ratio:.2f}x")
        # 显示 token 切分
        display_tokens = ["▁" if t == " " else t for t in token_strs]
        print(f"   切分: {' | '.join(display_tokens)}")
    
    print("\n" + "=" * 60)
    print("🎉 BPE Tokenizer 演示完成！")
    print("=" * 60)
    print("""
💡 回顾一下今天学到的：

1. BPE 从字符级别开始，不断合并高频相邻对
2. 合并顺序由训练语料的统计频率决定
3. 编码时按训练时的顺序回放合并规则
4. 词表大小 = 初始字符数 + 合并次数
5. 压缩率 = 字符数 / token 数，越高越好
6. 空格通常作为特殊 token 保留

下一步思考：token 变成数字后，模型怎么"理解"这些数字的含义？
→ 那就是 Day 3 的主题：词嵌入（Word Embedding）！
    """)


if __name__ == "__main__":
    main()
