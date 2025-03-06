from PIL import Image
import torch
from torchvision import transforms
from lenet_v2 import CNN
from torchvision.datasets import CIFAR10

model = CNN()
model.load_state_dict(torch.load(r"D:\Desktop\python\Comsen\cnn_cifar10_3.pth"))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((32, 32)),  # 调整图像大小为模型输入大小
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化图像
])

img_path = r"D:\Desktop\cat2.jpg"  # 替换为你的图像路径
image = Image.open(img_path).convert('RGB')

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)  # 添加批次维度

# 确保模型在不计算梯度的情况下进行预测
with torch.no_grad():
    output = model(input_batch)

# 获取预测结果
_, predicted = torch.max(output, 1)

# 类别名称
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# 打印预测的类别名称
predicted_class = classes[predicted.item()]
print(f"Predicted class: {predicted_class}")