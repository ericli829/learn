import torch  # 导入PyTorch库
from torchvision import datasets, transforms  # 从torchvision库中导入数据集和数据变换模块
import torch.nn as nn  # 导入神经网络模块
import torch.optim as optim  # 导入优化器模块
import matplotlib.pyplot as plt  # 导入绘图模块
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns  # 导入Seaborn库用于绘制混淆矩阵
import numpy as np
import pickle

# 定义数据预处理和变换，包括数据增强
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 随机裁剪
    transforms.RandomHorizontalFlip(p=0.8),  # 水平翻转
    transforms.RandomVerticalFlip(p=0.8),
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化图像
])

transform_test = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化图像
])

# 加载训练数据集
train_data_1 = datasets.CIFAR10(root=r"D:\Desktop\python\Comsen\cifar10\train", train=True, download=True, transform=transform_train)
train_data_2 = datasets.CIFAR10(root=r"D:\Desktop\python\Comsen\cifar10\train", train=True, download=True, transform=transform_test)
train_data = torch.utils.data.ConcatDataset([train_data_1, train_data_2])
# 创建训练数据加载器
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)  # 打乱并分批次读取数据

# 加载测试数据集
test_data = datasets.CIFAR10(root=r"D:\Desktop\python\Comsen\cifar10\test", train=False, download=True, transform=transform_test)
# 创建测试数据加载器
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 第一个卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(0.2)

        # 第二个卷积层
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(0.3)

        # 第三个卷积层
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(0.4)

        # 第四个卷积层
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(0.5)

        # 全连接层
        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 实例化模型
model = CNN()
# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 记录损失和准确率
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

if __name__ == "__main__":
    for epoch in range(20):  # 训练20个epoch
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data  # 获取输入和标签
            optimizer.zero_grad()  # 梯度清零
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            running_loss += loss.item()  # 累加损失

            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果
            total += labels.size(0)  # 累加总数
            correct += (predicted == labels).sum().item()  # 累加正确预测数

            if i % 100 == 99:  # 每100个批次打印一次损失和准确率
                train_loss = running_loss / 100
                train_accuracy = 100 * correct / total
                print(f"[{epoch + 1}, {i + 1}] train_loss: {train_loss:.3f}, train_accuracy: {train_accuracy:.2f}%")
                running_loss = 0.0

                # 计算测试损失和准确率
                model.eval()
                test_loss = 0.0
                correct = 0
                total = 0
                all_labels = []
                all_preds = []
                all_probs = []
                with torch.no_grad():
                    for data in test_loader:
                        images, labels = data
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        test_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                        all_labels.extend(labels.cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())
                        all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
                test_loss /= len(test_loader)
                test_accuracy = 100 * correct / total
                print(f"[{epoch + 1}, {i + 1}] test_loss: {test_loss:.3f}, test_accuracy: {test_accuracy:.2f}%")

                train_losses.append(train_loss)
                test_losses.append(test_loss)
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)

    print("Finished Training")  # 训练结束

    # 保存模型
    torch.save(model.state_dict(), "cnn_cifar10_3.pth")

    # 绘制损失和准确率随epoch变化的图表
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 标准化混淆矩阵

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.show()

    # 绘制ROC曲线和PR曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    pr_auc = dict()
    for i in range(10):
        fpr[i], tpr[i], _ = roc_curve([1 if label == i else 0 for label in all_labels], [prob[i] for prob in all_probs])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve([1 if label == i else 0 for label in all_labels], [prob[i] for prob in all_probs])
        pr_auc[i] = auc(recall[i], precision[i])

    # 绘制所有类别的ROC曲线
    plt.figure(figsize=(12, 8))
    for i in range(10):
        plt.plot(fpr[i], tpr[i], label=f'{classes[i]} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    # 绘制所有类别的PR曲线
    plt.figure(figsize=(12, 8))
    for i in range(10):
        plt.plot(recall[i], precision[i], label=f'{classes[i]} (area = {pr_auc[i]:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall (PR) Curve')
    plt.legend(loc='lower left')
    plt.show()

    data_to_save = {
    'train_losses': train_losses,
    'test_losses': test_losses,
    'train_accuracies': train_accuracies,
    'test_accuracies': test_accuracies,
    'all_labels': all_labels,
    'all_preds': all_preds,
    'all_probs': all_probs,
    'classes': classes
    }

    with open('plot_data_lev2.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)