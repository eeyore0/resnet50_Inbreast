"""
INbreast 乳腺病灶二分类训练脚本
基于 ResNet-50 实现良性/恶性病灶的深度学习分类
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ============================================================================
# 1. 数据集类定义
# ============================================================================
class INbreastDataset(Dataset):
    """
    INbreast 数据集类
    功能:
    - 加载指定路径下的图像文件
    - 应用数据转换(增强/归一化)
    - 将单通道灰度图转换为三通道输入
    """
    def __init__(self, image_paths, labels, transform=None):
        """
        参数:
            image_paths: 图像文件路径列表
            labels: 对应的标签列表 (0=Benign, 1=Malignant)
            transform: torchvision transforms 对象
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 读取图像(灰度模式)
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 'L' 模式 = 灰度图

        # 关键步骤: 将单通道转换为三通道 (复制灰度值到 RGB 三个通道)
        image = Image.merge('RGB', (image, image, image))

        # 应用数据转换
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


# ============================================================================
# 2. 数据预处理函数
# ============================================================================
def pad_to_square(img):
    """
    保持长宽比的正方形填充
    实现逻辑:
    1. 找到图像最长边
    2. 创建正方形黑色画布
    3. 将原图居中粘贴
    """
    width, height = img.size
    max_dim = max(width, height)

    # 创建黑色正方形画布
    new_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))

    # 计算粘贴位置(居中)
    paste_x = (max_dim - width) // 2
    paste_y = (max_dim - height) // 2

    # 粘贴原图
    new_img.paste(img, (paste_x, paste_y))
    return new_img


def get_transforms(is_training=True):
    """
    获取数据转换流程
    参数:
        is_training: 是否为训练集(训练集需要数据增强)
    """
    if is_training:
        # 训练集: 包含数据增强
        return transforms.Compose([
            transforms.Lambda(pad_to_square),  # 先填充为正方形
            transforms.Resize((224, 224)),     # 缩放到 ResNet 输入尺寸
            transforms.RandomHorizontalFlip(), # 随机水平翻转
            transforms.RandomRotation(10),     # 随机旋转 ±10 度
            transforms.ToTensor(),             # 转为张量
            transforms.Normalize(              # 标准化(使用 ImageNet 统计值)
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        # 验证集: 仅基础转换,无随机增强
        return transforms.Compose([
            transforms.Lambda(pad_to_square),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


# ============================================================================
# 3. 数据加载函数
# ============================================================================
def load_dataset(data_root, train_ratio=0.8):
    """
    从文件夹加载数据并拆分为训练集和验证集

    参数:
        data_root: 数据根目录(包含 'Benign' 和 'Malignant' 子文件夹)
        train_ratio: 训练集比例(默认 80%)

    返回:
        train_paths, train_labels, val_paths, val_labels
    """
    benign_folder = os.path.join(data_root, 'Benign')
    malignant_folder = os.path.join(data_root, 'Malignant')

    # 收集所有图像路径和标签
    image_paths = []
    labels = []

    # 加载良性样本 (标签=0)
    for filename in os.listdir(benign_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_paths.append(os.path.join(benign_folder, filename))
            labels.append(0)

    # 加载恶性样本 (标签=1)
    for filename in os.listdir(malignant_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_paths.append(os.path.join(malignant_folder, filename))
            labels.append(1)

    print(f"总样本数: {len(image_paths)} (良性: {labels.count(0)}, 恶性: {labels.count(1)})")

    # 拆分数据集(分层拆分,保持类别比例)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels,
        train_size=train_ratio,
        random_state=42,
        stratify=labels  # 确保训练集和验证集的类别比例一致
    )

    print(f"训练集样本数: {len(train_paths)}")
    print(f"验证集样本数: {len(val_paths)}")

    return train_paths, train_labels, val_paths, val_labels


# ============================================================================
# 4. 模型构建函数
# ============================================================================
def build_model(num_classes=2, pretrained=True):
    """
    构建 ResNet-50 模型

    关键修改:
    1. 加载 ImageNet 预训练权重
    2. 替换最后的全连接层,输出节点改为 2(二分类)
    """
    # 加载预训练的 ResNet-50
    model = models.resnet50(pretrained=pretrained)

    # 获取原始全连接层的输入特征数
    num_features = model.fc.in_features

    # 替换全连接层: 1000 类 -> 2 类
    model.fc = nn.Linear(num_features, num_classes)

    print(f"模型构建完成: ResNet-50 (预训练={pretrained}, 输出类别={num_classes})")
    return model


# ============================================================================
# 5. 训练函数
# ============================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    训练一个 epoch
    返回: 平均损失, 准确率
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 使用 tqdm 显示进度条
    pbar = tqdm(dataloader, desc="训练中", ncols=100)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ============================================================================
# 6. 验证函数
# ============================================================================
def validate(model, dataloader, criterion, device):
    """
    在验证集上评估模型
    返回: 平均损失, 准确率
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ============================================================================
# 7. 类别权重计算函数
# ============================================================================
def calculate_class_weights(labels, device):
    """
    计算类别权重以处理样本不平衡问题

    参数:
        labels: 标签列表
        device: 计算设备

    返回:
        class_weights: 类别权重张量

    计算方法:
        权重 = 总样本数 / (类别数 × 该类别样本数)
        这样少数类会获得更高的权重,平衡损失函数
    """
    from collections import Counter

    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)

    # 计算每个类别的权重
    class_weights = []
    for i in range(num_classes):
        weight = total_samples / (num_classes * label_counts[i])
        class_weights.append(weight)

    class_weights = torch.FloatTensor(class_weights).to(device)

    print(f"\n类别分布:")
    print(f"  良性 (0): {label_counts[0]} 样本, 权重: {class_weights[0]:.4f}")
    print(f"  恶性 (1): {label_counts[1]} 样本, 权重: {class_weights[1]:.4f}")
    print(f"  权重比: {class_weights[1]/class_weights[0]:.2f}:1 (恶性:良性)\n")

    return class_weights


# ============================================================================
# 8. 绘图函数
# ============================================================================
def plot_metrics(history, save_path='training_metrics.png'):
    """
    绘制训练过程中的损失和准确率曲线
    """
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线 (Loss Curve)
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 准确率曲线 (Accuracy Curve)
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n训练指标图表已保存至: {save_path}")
    plt.close()


# ============================================================================
# 9. 主训练流程
# ============================================================================
def main():
    # ------------------------------------------------------------------------
    # 配置参数
    # ------------------------------------------------------------------------
    DATA_ROOT = 'iocbd_b_m_data'  # 数据根目录(请根据实际路径修改)
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.0001
    TRAIN_RATIO = 0.8
    PATIENCE = 10  # 早停耐心值: 验证准确率连续10个epoch不提升则停止训练

    # 检测设备(GPU 优先)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cpu':
        print("警告: 未检测到 GPU, 训练将在 CPU 上进行(速度较慢)")

    # ------------------------------------------------------------------------
    # 加载数据
    # ------------------------------------------------------------------------
    train_paths, train_labels, val_paths, val_labels = load_dataset(
        DATA_ROOT, train_ratio=TRAIN_RATIO
    )

    # 创建数据集对象
    train_dataset = INbreastDataset(
        train_paths, train_labels,
        transform=get_transforms(is_training=True)
    )
    val_dataset = INbreastDataset(
        val_paths, val_labels,
        transform=get_transforms(is_training=False)
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=4, pin_memory=True
    )

    # ------------------------------------------------------------------------
    # 构建模型
    # ------------------------------------------------------------------------
    model = build_model(num_classes=2, pretrained=True)
    model = model.to(device)

    # 计算类别权重(处理样本不平衡)
    class_weights = calculate_class_weights(train_labels, device)

    # 定义损失函数(加权交叉熵损失 - 自动平衡类别不平衡问题)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 定义优化器(Adam - 主流且鲁棒的优化算法, 添加 L2 正则化防止过拟合)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # 定义学习率调度器(ReduceLROnPlateau - 当验证损失不再下降时自动降低学习率)
    # mode='min': 监控指标越小越好(损失函数)
    # factor=0.1: 学习率衰减为原来的 0.1 倍
    # patience=5: 验证损失连续 5 个 epoch 不下降时触发
    # verbose=True: 打印学习率变化信息
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )

    # ------------------------------------------------------------------------
    # 训练循环
    # ------------------------------------------------------------------------
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }

    best_val_acc = 0.0  # 记录历史最佳验证准确率
    patience_counter = 0  # 早停计数器: 记录验证准确率连续未提升的epoch数

    print("\n" + "="*60)
    print("开始训练".center(60))
    print("="*60 + "\n")

    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 验证阶段
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 更新学习率(基于验证损失 - ReduceLROnPlateau 会自动判断是否需要降低学习率)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # 记录指标
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # 打印 epoch 日志
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | "
              f"训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.4f} | "
              f"验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.4f} | "
              f"学习率: {current_lr:.6f}")

        # 保存最佳模型(基于验证准确率) 并更新早停计数器
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0  # 重置早停计数器
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  >> 最佳模型已更新! (验证准确率: {val_acc:.4f})")
        else:
            patience_counter += 1  # 验证准确率未提升,计数器加1
            print(f"  >> 验证准确率未提升 (耐心计数: {patience_counter}/{PATIENCE})")

        # 早停检查
        if patience_counter >= PATIENCE:
            print(f"\n{'='*60}")
            print(f"训练已早停,因为验证准确率连续 {PATIENCE} 个 epoch 未提升".center(60))
            print(f"{'='*60}\n")
            break

    # ------------------------------------------------------------------------
    # 训练结束 - 生成报告
    # ------------------------------------------------------------------------
    print("\n" + "="*60)
    print("训练完成".center(60))
    print("="*60)
    print(f"实际训练轮数: {len(history['train_loss'])} / {NUM_EPOCHS}")
    print(f"最佳验证准确率: {best_val_acc:.4f}")
    print(f"最佳模型已保存至: best_model.pth")

    # 绘制训练曲线
    plot_metrics(history, save_path='training_metrics.png')


# ============================================================================
# 程序入口
# ============================================================================
if __name__ == '__main__':
    main()