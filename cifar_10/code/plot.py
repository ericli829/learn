import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# 读取保存的数据
with open('plot_data.pkl', 'rb') as f:
    data = pickle.load(f)

train_losses = data['train_losses']
test_losses = data['test_losses']
train_accuracies = data['train_accuracies']
test_accuracies = data['test_accuracies']
all_labels = data['all_labels']
all_preds = data['all_preds']
all_probs = data['all_probs']
classes = data['classes']

# 绘制损失和准确率随epoch变化的图表
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, test_accuracies, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 绘制混淆矩阵
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