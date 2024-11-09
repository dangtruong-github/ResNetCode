import torch
from utils import GetModel

sample_data = torch.randn((1, 3, 224, 224))
print(f"Shape of sample data: {sample_data.shape}")

resnet18 = GetModel(3, 10, "ResNet18")
output18 = resnet18(sample_data)
print(f"Shape of output of ResNet-18: {output18.shape}")

resnet34 = GetModel(3, 10, "ResNet34")
output34 = resnet34(sample_data)
print(f"Shape of output of ResNet-34: {output34.shape}")

resnet50 = GetModel(3, 10, "ResNet50")
output50 = resnet50(sample_data)
print(f"Shape of output of ResNet-50: {output50.shape}")

resnet101 = GetModel(3, 10, "ResNet101")
output101 = resnet101(sample_data)
print(f"Shape of output of ResNet-101: {output101.shape}")

resnet152 = GetModel(3, 10, "ResNet152")
output152 = resnet152(sample_data)
print(f"Shape of output of ResNet-152: {output152.shape}")
