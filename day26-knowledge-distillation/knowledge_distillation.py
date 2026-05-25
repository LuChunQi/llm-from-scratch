#!/usr/bin/env python3
"""
Day 26: 知识蒸馏 — 大模型教小模型的"师徒传承"

今天我们从零实现 Hinton 2015 年的经典知识蒸馏方法，包括：
1. 教师模型 + 学生模型（MLP 分类器）
2. 带温度的 Softmax
3. 经典蒸馏损失 = α × 软标签损失 + (1-α) × 硬标签损失
4. 完整蒸馏训练 + 对比实验
5. 温度参数扫描
6. 暗知识可视化
7. Feature-Based 中间层蒸馏
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
from copy import deepcopy

# ============================================================
# 工具函数
# ============================================================

def set_seed(seed=42):
    """固定随机种子，确保可复现"""
    torch.manual_seed(seed)
    np.random.seed(seed)

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  设备: {device}")


# ============================================================
# 1. 带温度的 Softmax
# ============================================================

def softmax_with_temperature(logits, T=1.0):
    """
    带温度参数的 Softmax
    
    公式: P(i) = exp(z_i / T) / Σ exp(z_j / T)
    
    T=1: 标准 Softmax
    T>1: 分布更"软"（均匀），暗知识更明显
    T<1: 分布更"硬"（尖锐），趋向 one-hot
    
    Args:
        logits: 未归一化的分数, shape [batch, num_classes]
        T: 温度参数
    Returns:
        概率分布, shape [batch, num_classes]
    """
    # 除以温度后再做 softmax
    return F.softmax(logits / T, dim=-1)


def demo_temperature_effect():
    """演示温度参数对输出分布的影响"""
    print("=" * 60)
    print("🌡️  温度参数效果演示")
    print("=" * 60)
    
    # 模拟教师模型对一个"猫"图片的 logits 输出
    # logits[0] 最大 → 模型认为是"猫"
    logits = torch.tensor([[10.0, 2.0, 1.0, 0.5, 0.1]])
    class_names = ['猫', '狗', '老虎', '豹子', '汽车']
    
    print(f"\n原始 logits: {logits[0].tolist()}")
    print(f"类别: {class_names}")
    
    for T in [0.5, 1, 2, 5, 10, 20]:
        probs = softmax_with_temperature(logits, T)
        probs_list = [f"{p:.4f}" for p in probs[0].tolist()]
        print(f"\n  T = {T:5.1f}: {probs_list}")
    
    print("\n→ 温度越高，分布越'软'，暗知识（猫和狗/老虎的相似性）越明显！")
    
    # 可视化不同温度下的分布
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Temperature Effect on Softmax Distribution', fontsize=16, fontweight='bold')
    
    temperatures = [0.5, 1, 2, 5, 10, 20]
    for idx, T in enumerate(temperatures):
        ax = axes[idx // 3][idx % 3]
        probs = softmax_with_temperature(logits, T).detach().numpy()[0]
        bars = ax.bar(class_names, probs, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax.set_title(f'T = {T}', fontsize=14)
        ax.set_ylim(0, 1.0)
        # 在柱子上标注概率值
        for bar, prob in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('temperature_effect.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n📊 温度效果图已保存到 temperature_effect.png")


# ============================================================
# 2. 教师模型 & 学生模型
# ============================================================

class TeacherNet(nn.Module):
    """
    教师模型：较大的 MLP
    - 3 个隐藏层: 512 → 256 → 128
    - 参数量约 ~130K
    """
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        """返回 logits 和中间层特征（用于 Feature-Based 蒸馏）"""
        h1 = self.relu(self.fc1(x))       # 第1个隐藏层输出
        h1 = self.dropout(h1)
        h2 = self.relu(self.fc2(h1))      # 第2个隐藏层输出
        h2 = self.dropout(h2)
        h3 = self.relu(self.fc3(h2))      # 第3个隐藏层输出
        logits = self.fc4(h3)             # 最终输出
        return logits, [h1, h2, h3]


class StudentNet(nn.Module):
    """
    学生模型：较小的 MLP
    - 1 个隐藏层: 64
    - 参数量约 ~50K（不到教师的 40%）
    """
    def __init__(self, input_dim=784, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h1 = self.relu(self.fc1(x))       # 隐藏层输出
        logits = self.fc2(h1)             # 最终输出
        return logits, [h1]


# ============================================================
# 3. 生成模拟数据
# ============================================================

def generate_synthetic_data(n_samples=3000, input_dim=784, num_classes=10, difficulty='medium'):
    """
    生成模拟的分类数据，控制难度使蒸馏效果可观察
    
    难度控制：
    - 'easy': 类间距离大，噪声小 → 太容易，全部100%
    - 'medium': 部分类重叠 + 高噪声 → 蒸馏效果可见
    - 'hard': 大量类重叠 → 即使教师也做不到100%
    """
    print("\n" + "=" * 60)
    print("📊 生成模拟数据")
    print("=" * 60)
    
    set_seed(42)
    samples_per_class = n_samples // num_classes
    X = []
    y = []
    
    # 先生成类中心，让某些类天然相似（模拟真实世界中的猫/豹、6/8 等混淆）
    # 10个类分成几组，组内的类中心比较接近
    base_centers = torch.randn(5, input_dim) * 3.0  # 5个"基中心"
    
    class_centers = []
    for c in range(num_classes):
        # 每两个类共享一个基中心，加上小偏移
        base_idx = c // 2
        offset = torch.randn(input_dim) * (1.0 if difficulty == 'easy' else 0.5 if difficulty == 'medium' else 0.2)
        center = base_centers[base_idx] + offset
        class_centers.append(center)
    
    # 噪声水平：难度越高噪声越大
    noise_scale = {'easy': 1.0, 'medium': 2.5, 'hard': 4.0}[difficulty]
    
    for c in range(num_classes):
        noise = torch.randn(samples_per_class, input_dim) * noise_scale
        class_data = class_centers[c].unsqueeze(0) + noise
        X.append(class_data)
        y.extend([c] * samples_per_class)
    
    X = torch.cat(X, dim=0)
    y = torch.tensor(y, dtype=torch.long)
    
    # 打乱顺序
    perm = torch.randperm(len(y))
    X = X[perm]
    y = y[perm]
    
    # 划分训练集和测试集（80/20）
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"  难度: {difficulty}")
    print(f"  训练集: {X_train.shape[0]} 样本")
    print(f"  测试集: {X_test.shape[0]} 样本")
    print(f"  输入维度: {input_dim}")
    print(f"  类别数: {num_classes}")
    print(f"  噪声水平: {noise_scale}")
    
    return X_train, X_test, y_train, y_test


# ============================================================
# 4. 蒸馏损失函数
# ============================================================

def distillation_loss(student_logits, teacher_logits, labels, T=5.0, alpha=0.7):
    """
    经典知识蒸馏损失函数（Hinton 2015）
    
    L_total = α × L_soft + (1 - α) × L_hard
    
    L_soft = KL_div(softmax(z_teacher/T), softmax(z_student/T)) × T²
    L_hard = CrossEntropy(softmax(z_student), y_true)
    
    Args:
        student_logits: 学生模型的 logits
        teacher_logits: 教师模型的 logits（已经 detach，不传梯度）
        labels: 真实标签
        T: 温度参数
        alpha: 软标签的权重（0~1）
    Returns:
        total_loss: 总损失
        soft_loss: 软标签损失
        hard_loss: 硬标签损失
    """
    # 软标签损失：用 KL 散度衡量学生和教师的分布差异
    # F.kl_div 的 input 是 log-probabilities，target 是 probabilities
    soft_student = F.log_softmax(student_logits / T, dim=-1)   # 学生的 log-prob
    soft_teacher = F.softmax(teacher_logits / T, dim=-1)       # 教师的 prob
    
    # KL 散度，除以 batch_size 求平均，再乘 T² 补偿梯度缩小
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T * T)
    
    # 硬标签损失：标准交叉熵
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # 加权组合
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return total_loss, soft_loss, hard_loss


def feature_distillation_loss(student_features, teacher_features, labels=None, T=5.0, alpha=0.7, feature_weight=0.1):
    """
    Feature-Based 蒸馏损失 = 输出蒸馏损失 + 中间层对齐损失
    
    中间层对齐: L_feat = Σ MSE(H_student^l, transform(H_teacher^m))
    
    Args:
        student_features: 学生中间层特征列表
        teacher_features: 教师中间层特征列表
        其余参数同 distillation_loss
    """
    # 这里只做简单的 MSE 对齐（实际中可能需要投影层来匹配维度）
    feat_loss = torch.tensor(0.0, device=student_features[0].device)
    
    # 对齐最后一层特征（维度不同时用投影）
    s_feat = student_features[-1]  # 学生最后一层特征
    t_feat = teacher_features[-1]  # 教师最后一层特征
    
    # 如果维度不匹配，用线性投影（简单的均值池化）
    if s_feat.shape != t_feat.shape:
        # 教师特征维度通常更大，我们只比较统计量
        s_mean = s_feat.mean(dim=-1)
        t_mean = t_feat.mean(dim=-1)
        if s_mean.shape == t_mean.shape:
            feat_loss = F.mse_loss(s_mean, t_mean.detach())
        else:
            # 维度实在不匹配，比较范数
            s_norm = s_feat.norm(dim=-1)
            t_norm = t_feat.norm(dim=-1)
            feat_loss = F.mse_loss(s_norm, t_norm.detach())
    else:
        feat_loss = F.mse_loss(s_feat, t_feat.detach())
    
    return feature_weight * feat_loss


# ============================================================
# 5. 训练函数
# ============================================================

def train_teacher(model, X_train, y_train, X_test, y_test, epochs=50, lr=0.001):
    """训练教师模型（标准交叉熵）"""
    print("\n" + "=" * 60)
    print("👨‍🏫 训练教师模型")
    print("=" * 60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    train_acc_history = []
    test_acc_history = []
    
    for epoch in range(epochs):
        model.train()
        logits, _ = model(X_train.to(device))
        loss = F.cross_entropy(logits, y_train.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 计算准确率
        with torch.no_grad():
            train_pred = logits.argmax(dim=-1)
            train_acc = (train_pred == y_train.to(device)).float().mean().item()
            train_acc_history.append(train_acc)
            
            test_logits, _ = model(X_test.to(device))
            test_pred = test_logits.argmax(dim=-1)
            test_acc = (test_pred == y_test.to(device)).float().mean().item()
            test_acc_history.append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    print(f"\n✅ 教师模型最终测试准确率: {test_acc_history[-1]:.4f}")
    return train_acc_history, test_acc_history


def train_student_baseline(model, X_train, y_train, X_test, y_test, epochs=50, lr=0.001):
    """训练学生模型（基线：只用硬标签，无蒸馏）"""
    print("\n" + "=" * 60)
    print("👶 训练学生模型（基线：无蒸馏）")
    print("=" * 60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    
    train_acc_history = []
    test_acc_history = []
    
    for epoch in range(epochs):
        model.train()
        logits, _ = model(X_train.to(device))
        loss = F.cross_entropy(logits, y_train.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            train_pred = logits.argmax(dim=-1)
            train_acc = (train_pred == y_train.to(device)).float().mean().item()
            train_acc_history.append(train_acc)
            
            test_logits, _ = model(X_test.to(device))
            test_pred = test_logits.argmax(dim=-1)
            test_acc = (test_pred == y_test.to(device)).float().mean().item()
            test_acc_history.append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    print(f"\n📊 基线学生最终测试准确率: {test_acc_history[-1]:.4f}")
    return train_acc_history, test_acc_history


def train_student_distilled(student, teacher, X_train, y_train, X_test, y_test,
                             T=5.0, alpha=0.7, epochs=50, lr=0.001,
                             use_feature_distill=False):
    """
    训练学生模型（知识蒸馏）
    
    Args:
        student: 学生模型
        teacher: 教师模型（已训练好）
        T: 温度参数
        alpha: 软标签权重
        use_feature_distill: 是否使用 Feature-Based 蒸馏
    """
    print("\n" + "=" * 60)
    label = "Feature-Based" if use_feature_distill else "Response-Based"
    print(f"👶🔥 训练学生模型（{label} 蒸馏，T={T}, α={alpha}）")
    print("=" * 60)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    student.to(device)
    teacher.eval()  # 教师模型冻结，不训练
    teacher.to(device)
    
    train_acc_history = []
    test_acc_history = []
    
    for epoch in range(epochs):
        student.train()
        
        # 前向传播：学生和教师都计算 logits
        student_logits, student_feats = student(X_train.to(device))
        with torch.no_grad():  # 教师不需要梯度
            teacher_logits, teacher_feats = teacher(X_train.to(device))
        
        # 计算蒸馏损失
        total_loss, soft_loss, hard_loss = distillation_loss(
            student_logits, teacher_logits, y_train.to(device), T=T, alpha=alpha
        )
        
        # 如果启用 Feature-Based 蒸馏
        if use_feature_distill:
            feat_loss = feature_distillation_loss(student_feats, teacher_feats)
            total_loss = total_loss + feat_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            train_pred = student_logits.argmax(dim=-1)
            train_acc = (train_pred == y_train.to(device)).float().mean().item()
            train_acc_history.append(train_acc)
            
            test_logits, _ = student(X_test.to(device))
            test_pred = test_logits.argmax(dim=-1)
            test_acc = (test_pred == y_test.to(device)).float().mean().item()
            test_acc_history.append(test_acc)
        
        if (epoch + 1) % 10 == 0:
            extra = f" | Feature: {feat_loss.item():.4f}" if use_feature_distill else ""
            print(f"  Epoch {epoch+1:3d} | Total: {total_loss.item():.4f} "
                  f"(Soft: {soft_loss.item():.4f}, Hard: {hard_loss.item():.4f}){extra} | "
                  f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
    
    print(f"\n📊 蒸馏学生最终测试准确率: {test_acc_history[-1]:.4f}")
    return train_acc_history, test_acc_history


# ============================================================
# 6. 暗知识可视化
# ============================================================

def visualize_dark_knowledge(teacher, X_test, y_test, num_classes=10):
    """
    可视化教师模型的暗知识：
    - 教师对每个类别的平均预测分布
    - 类间相似性矩阵
    """
    print("\n" + "=" * 60)
    print("🔍 暗知识可视化")
    print("=" * 60)
    
    teacher.eval()
    with torch.no_grad():
        logits, _ = teacher(X_test.to(device))
        probs = F.softmax(logits, dim=-1).cpu().numpy()
    
    y_np = y_test.numpy()
    
    # 计算每个真实类别下的平均预测分布
    class_names = [str(i) for i in range(num_classes)]
    avg_probs = np.zeros((num_classes, num_classes))
    
    for c in range(num_classes):
        mask = y_np == c
        if mask.sum() > 0:
            avg_probs[c] = probs[mask].mean(axis=0)
    
    # 可视化 1：每个类别的平均预测分布
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Teacher Dark Knowledge: Per-Class Average Prediction Distribution', 
                 fontsize=14, fontweight='bold')
    
    for c in range(num_classes):
        ax = axes[c // 5][c % 5]
        bars = ax.bar(class_names, avg_probs[c], 
                      color=['#FF6B6B' if i == c else '#4ECDC4' for i in range(num_classes)])
        ax.set_title(f'True Class: {c}', fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel('Predicted Class')
    
    plt.tight_layout()
    plt.savefig('dark_knowledge_per_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 暗知识（每类平均预测）图已保存到 dark_knowledge_per_class.png")
    
    # 可视化 2：类间相似性矩阵（去掉对角线，看"错误答案"的分布）
    # 对于每个类别，找概率第二高的类（这就是暗知识最直观的体现）
    similarity_matrix = avg_probs.copy()
    np.fill_diagonal(similarity_matrix, 0)  # 去掉对角线
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(similarity_matrix, cmap='YlOrRd')
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted Class', fontsize=12)
    ax.set_ylabel('True Class', fontsize=12)
    ax.set_title('Inter-Class Similarity (Dark Knowledge Heatmap)', fontsize=14, fontweight='bold')
    
    # 在格子中标注数值
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                ax.text(j, i, f'{similarity_matrix[i, j]:.3f}', 
                       ha='center', va='center', fontsize=8,
                       color='white' if similarity_matrix[i, j] > 0.3 else 'black')
    
    plt.colorbar(im, ax=ax, label='Average Probability')
    plt.savefig('dark_knowledge_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("📊 暗知识（类间相似性热力图）已保存到 dark_knowledge_heatmap.png")
    
    # 打印每个类别最相似的"错误"类别（Top-2 混淆）
    print("\n  各类最容易混淆的类别（暗知识）：")
    for c in range(num_classes):
        sorted_idx = np.argsort(avg_probs[c])[::-1]
        top2 = [(sorted_idx[i], avg_probs[c][sorted_idx[i]]) for i in range(3) if sorted_idx[i] != c]
        top2_str = ", ".join([f"类{idx}({prob:.3f})" for idx, prob in top2[:2]])
        print(f"    类 {c} 最容易混淆: {top2_str}")


# ============================================================
# 7. 温度参数扫描实验
# ============================================================

def temperature_sweep(teacher, X_train, y_train, X_test, y_test, 
                      temperatures=[1, 2, 3, 5, 7, 10, 15, 20], alpha=0.7, epochs=50):
    """
    扫描不同温度参数，找到最优温度
    """
    print("\n" + "=" * 60)
    print("🌡️  温度参数扫描实验")
    print("=" * 60)
    
    results = {}
    
    for T in temperatures:
        # 每次都用全新的学生模型
        student = StudentNet(input_dim=784, num_classes=10)
        _, test_acc = train_student_distilled(
            student, teacher, X_train, y_train, X_test, y_test,
            T=T, alpha=alpha, epochs=epochs, lr=0.001
        )
        final_acc = test_acc[-1]
        results[T] = final_acc
        print(f"  T = {T:5.1f} → Test Acc = {final_acc:.4f}")
    
    # 可视化
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    temps = list(results.keys())
    accs = list(results.values())
    
    ax.plot(temps, accs, 'o-', color='#FF6B6B', linewidth=2, markersize=10)
    ax.set_xlabel('Temperature (T)', fontsize=14)
    ax.set_ylabel('Test Accuracy', fontsize=14)
    ax.set_title('Temperature vs. Student Performance', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 标注最优温度
    best_T = max(results, key=results.get)
    best_acc = results[best_T]
    ax.annotate(f'Best: T={best_T}, Acc={best_acc:.4f}',
                xy=(best_T, best_acc), xytext=(best_T + 2, best_acc - 0.02),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('temperature_sweep.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n📊 温度扫描图已保存到 temperature_sweep.png")
    print(f"🏆 最优温度: T = {best_T}, 准确率 = {best_acc:.4f}")
    
    return results


# ============================================================
# 8. 对比实验：三种训练方式全面对比
# ============================================================

def comparison_experiment(teacher, X_train, y_train, X_test, y_test,
                          T=5.0, alpha=0.7, epochs=50):
    """
    对比三种训练方式：
    1. 学生基线（无蒸馏）
    2. 学生 + Response-Based 蒸馏
    3. 学生 + Feature-Based 蒸馏
    """
    print("\n" + "=" * 60)
    print("⚔️  三种训练方式对比实验")
    print("=" * 60)
    
    # 方式 1：无蒸馏基线
    student_baseline = StudentNet(input_dim=784, num_classes=10)
    bl_train, bl_test = train_student_baseline(
        student_baseline, X_train, y_train, X_test, y_test, epochs=epochs
    )
    
    # 方式 2：Response-Based 蒸馏
    student_resp = StudentNet(input_dim=784, num_classes=10)
    resp_train, resp_test = train_student_distilled(
        student_resp, teacher, X_train, y_train, X_test, y_test,
        T=T, alpha=alpha, epochs=epochs
    )
    
    # 方式 3：Feature-Based 蒸馏
    student_feat = StudentNet(input_dim=784, num_classes=10)
    feat_train, feat_test = train_student_distilled(
        student_feat, teacher, X_train, y_train, X_test, y_test,
        T=T, alpha=alpha, epochs=epochs, use_feature_distill=True
    )
    
    # 获取教师的准确率
    teacher.eval()
    with torch.no_grad():
        t_logits, _ = teacher(X_test.to(device))
        teacher_acc = (t_logits.argmax(dim=-1) == y_test.to(device)).float().mean().item()
    
    # 可视化训练曲线对比
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs_range = range(1, epochs + 1)
    
    ax1.plot(epochs_range, bl_train, '-', color='#95a5a6', linewidth=2, label='Student Baseline (Train)')
    ax1.plot(epochs_range, resp_train, '-', color='#3498db', linewidth=2, label='Distilled-Response (Train)')
    ax1.plot(epochs_range, feat_train, '-', color='#e74c3c', linewidth=2, label='Distilled-Feature (Train)')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, bl_test, '-', color='#95a5a6', linewidth=2, label='Student Baseline')
    ax2.plot(epochs_range, resp_test, '-', color='#3498db', linewidth=2, label='Distilled-Response')
    ax2.plot(epochs_range, feat_test, '-', color='#e74c3c', linewidth=2, label='Distilled-Feature')
    ax2.axhline(y=teacher_acc, color='#2ecc71', linestyle='--', linewidth=2, label=f'Teacher ({teacher_acc:.4f})')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Test Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_experiment.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印最终对比结果
    print("\n" + "=" * 60)
    print("📊 最终对比结果")
    print("=" * 60)
    print(f"  👨‍🏫 教师模型:        {teacher_acc:.4f}")
    print(f"  👶 学生基线（无蒸馏）: {bl_test[-1]:.4f}")
    print(f"  👶🔥 Response蒸馏:   {resp_test[-1]:.4f}  (提升 {resp_test[-1] - bl_test[-1]:+.4f})")
    print(f"  👶🔥 Feature蒸馏:    {feat_test[-1]:.4f}  (提升 {feat_test[-1] - bl_test[-1]:+.4f})")
    print(f"\n📊 对比图已保存到 comparison_experiment.png")
    
    return {
        'teacher': teacher_acc,
        'baseline': bl_test[-1],
        'response_distill': resp_test[-1],
        'feature_distill': feat_test[-1],
    }


# ============================================================
# 9. 主函数
# ============================================================

def main():
    print("🧠 Day 26: 知识蒸馏 — 大模型教小模型的'师徒传承'")
    print("=" * 60)
    
    # ----------------------------------------------------------
    # Step 1: 温度参数效果演示
    # ----------------------------------------------------------
    demo_temperature_effect()
    
    # ----------------------------------------------------------
    # Step 2: 生成模拟数据
    # ----------------------------------------------------------
    X_train, X_test, y_train, y_test = generate_synthetic_data(
        n_samples=3000, input_dim=784, num_classes=10, difficulty='medium'
    )
    
    # ----------------------------------------------------------
    # Step 3: 训练教师模型
    # ----------------------------------------------------------
    teacher = TeacherNet(input_dim=784, num_classes=10)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"\n👨‍🏫 教师模型参数量: {teacher_params:,}")
    
    train_teacher(teacher, X_train, y_train, X_test, y_test, epochs=80, lr=0.001)
    
    # ----------------------------------------------------------
    # Step 4: 训练学生基线（无蒸馏）
    # ----------------------------------------------------------
    student_baseline = StudentNet(input_dim=784, num_classes=10)
    student_params = sum(p.numel() for p in student_baseline.parameters())
    print(f"\n👶 学生模型参数量: {student_params:,} (是教师的 {student_params/teacher_params*100:.1f}%)")
    
    train_student_baseline(student_baseline, X_train, y_train, X_test, y_test, epochs=80, lr=0.001)
    
    # ----------------------------------------------------------
    # Step 5: 训练蒸馏学生模型
    # ----------------------------------------------------------
    student_distilled = StudentNet(input_dim=784, num_classes=10)
    train_student_distilled(
        student_distilled, teacher, X_train, y_train, X_test, y_test,
        T=5.0, alpha=0.7, epochs=80, lr=0.001
    )
    
    # ----------------------------------------------------------
    # Step 6: 暗知识可视化
    # ----------------------------------------------------------
    visualize_dark_knowledge(teacher, X_test, y_test, num_classes=10)
    
    # ----------------------------------------------------------
    # Step 7: 温度参数扫描
    # ----------------------------------------------------------
    temp_results = temperature_sweep(
        teacher, X_train, y_train, X_test, y_test,
        temperatures=[1, 2, 3, 5, 7, 10, 15, 20],
        alpha=0.7, epochs=50
    )
    
    # ----------------------------------------------------------
    # Step 8: 三种训练方式全面对比
    # ----------------------------------------------------------
    comparison = comparison_experiment(
        teacher, X_train, y_train, X_test, y_test,
        T=5.0, alpha=0.7, epochs=60
    )
    
    # ----------------------------------------------------------
    # 总结
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("🎉 Day 26 总结")
    print("=" * 60)
    print(f"""
    📌 核心概念：
    1. 暗知识 — 教师模型输出中错误答案之间的相对关系
    2. 温度参数 — T>1 让分布变软，暴露暗知识
    3. T² 补偿 — 温度软化导致梯度缩小，需要乘 T² 来补偿
    4. 蒸馏损失 = α × KL散度(软标签) + (1-α) × 交叉熵(硬标签)
    
    📊 实验结果：
    - 教师模型准确率:     {comparison['teacher']:.4f}
    - 学生基线准确率:     {comparison['baseline']:.4f}
    - Response蒸馏准确率: {comparison['response_distill']:.4f} ({comparison['response_distill'] - comparison['baseline']:+.4f})
    - Feature蒸馏准确率:  {comparison['feature_distill']:.4f} ({comparison['feature_distill'] - comparison['baseline']:+.4f})
    
    💡 关键发现：
    - 蒸馏让学生模型学到了教师的"暗知识"
    - 温度参数有最优值（通常 T=3~10）
    - Feature-Based 蒸馏比 Response-Based 更进一步
    """)


if __name__ == '__main__':
    main()
