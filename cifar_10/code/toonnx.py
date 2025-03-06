import torch
import torch.onnx
from train3 import CNN

# 定义模型参数

# 初始化模型
model = CNN()

# 加载训练好的模型参数
model.load_state_dict(torch.load(r'D:\Desktop\python\Comsen\cnn_cifar10_v2.pth'))

# 设置模型为评估模式
model.eval()

# 创建一个示例输入张量
dummy_input = torch.randn(1, 3, 32, 32)  # Assuming the input size is (3, 32, 32) for CIFAR-10

# 导出模型为 ONNX 格式
torch.onnx.export(model, dummy_input, "lenet_v2.onnx", 
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

print("模型已成功转换为 ONNX 格式并保存为 model.onnx")