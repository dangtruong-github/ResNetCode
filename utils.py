import torch
import torchvision
import torchvision.transforms as transforms

from ResNetLarge import ResnetLarge
from ResNetSmall import ResnetSmall

device = "cuda" if torch.cuda.is_available() else "cpu"

small_models = ["ResNet18", "ResNet34"]
large_models = ["ResNet50", "ResNet101", "ResNet152"]
valid_models = small_models + large_models
print(valid_models)


def GetModel(in_channels, num_classes, type_model="ResNet18"):
    if type_model not in valid_models:
        raise ValueError(f"{type_model} is not a valid type of model."
                         f"Try one of {valid_models} instead")

    if type_model in small_models:
        num_residual_block = None
        if type_model == "ResNet18":
            num_residual_block = [2, 2, 2, 2]
        else:
            num_residual_block = [3, 4, 6, 3]

        model = ResnetSmall(
            in_channels=in_channels,
            num_residual_block=num_residual_block,
            num_class=num_classes,
        ).to(device)

        return model
    else:
        num_residual_block = None
        if type_model == "ResNet50":
            num_residual_block = [3, 4, 6, 3]
        elif type_model == "ResNet101":
            num_residual_block = [3, 4, 23, 3]
        else:
            num_residual_block = [3, 8, 36, 3]

        model = ResnetLarge(
            in_channels=in_channels,
            num_residual_block=num_residual_block,
            num_class=num_classes,
        ).to(device)

        return model


def GetData():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    batch_size = 32

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, testloader
