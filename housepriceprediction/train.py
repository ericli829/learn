import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X_test = pd.read_csv('D:\\Desktop\\python\\Comsen\\winter_vacation\\test.csv') #读取测试集和训练集
X_train = pd.read_csv('D:\\Desktop\\python\\Comsen\\winter_vacation\\train.csv')
all_features = pd.concat((X_train.iloc[:, 1:-1], X_test.iloc[:, 1:])) #同时处理两组数据

num_features = all_features.dtypes[all_features.dtypes != 'object'].index #获取数字特征的索引
all_features[num_features] = all_features[num_features].apply( #标准化数据
    lambda x: (x - x.mean()) / x.std()
)
all_features[num_features] = all_features[num_features].fillna(0) #用平均值0填充na数据
all_features = pd.get_dummies(all_features, dummy_na=True) #独热编码分类数据（同时处理na数据）
all_features = all_features.astype(np.float32) #将np.bool转化为np.float

n_train = X_train.shape[0]
train_features = torch.tensor(all_features.iloc[:n_train].values, dtype=torch.float32) #重新分割训练集和测试集
test_features = torch.tensor(all_features.iloc[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(X_train.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

train_features, val_features, train_labels, val_labels = train_test_split( #分割验证集
    train_features, train_labels, test_size=0.2, random_state=42 
)

class PredictModel(nn.Module): #构建神经网络
    def __init__(self, size_1, size_2, size_3):
        super(PredictModel, self).__init__()
        self.fc1 = nn.Linear(train_features.shape[1], size_1)
        self.fc2 = nn.Linear(size_1, size_2)
        self.fc3 = nn.Linear(size_2, size_3)
        self.fc4 = nn.Linear(size_3, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

lr = 0.002 #合理设置参数
size_1, size_2, size_3 = [64, 32, 32]
epochs = 20000
patience = 100

train_losses = []
val_losses = []
best_loss = float('inf')
patience_counter = 0

model = PredictModel(size_1, size_2, size_3)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

for epoch in range(epochs): #训练
    model.train()
    optimizer.zero_grad()
    output = model(train_features)
    loss = criterion(output, train_labels)
    train_losses.append(loss.detach().numpy())
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(val_features)
        val_loss = criterion(val_output, val_labels)
        val_losses.append(val_loss.detach().numpy())

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}],train_loss: {train_losses[-1]:.0f}, val_loss: {val_losses[-1]:.0f}')

    if val_loss < best_loss: #早停法
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'model.pth')
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping at epoch {epoch + 1}')
        break

model.load_state_dict(torch.load('model.pth'))

model.eval() 
with torch.no_grad():
    predictions = model(test_features) #预测

predictions = predictions.detach().numpy()
predictions = pd.DataFrame(predictions, columns=['SalePrice'])
predictions.to_csv('predictions2.csv', index=True) #保存为csv

plt.plot(train_losses, label='Train Loss') #绘制误差图线
plt.plot(val_losses, label='Validation Loss')
plt.ylabel('Loss')
plt.xlabel('epoch')
plt.legend()
plt.show()