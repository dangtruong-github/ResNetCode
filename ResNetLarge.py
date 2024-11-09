import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class BasicBlockLarge(nn.Module):
    def __init__(
        self,
        in_features=256,
        out_features=256,
        stride=[1, 1, 1],
        down_sample=False
    ):
        super(BasicBlockLarge, self).__init__()
        self.conv1 = nn.Conv2d(
            in_features, out_features // 4,
            kernel_size=1, stride=stride[0],
            bias=False
        )
        # weight layer
        self.bn1 = nn.BatchNorm2d(out_features // 4)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            out_features // 4, out_features // 4,
            kernel_size=3, stride=stride[1],
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_features // 4)
        self.conv3 = nn.Conv2d(
            out_features // 4, out_features,
            kernel_size=3, stride=stride[2],
            padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_features)

        self.downsample = None
        if down_sample:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_features, out_features,
                    kernel_size=1, stride=2,
                    bias=False
                ),
                nn.BatchNorm2d(out_features)
            )

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x0 = x.clone()
        x = self.conv1(x)
        # print(f"Conv1 shape: {x.shape}")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # print(f"Conv2 shape: {x.shape}")
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        # print(f"Conv3 shape: {x.shape}")
        x = self.bn3(x)

        if self.downsample is not None:
            x0 = self.downsample(x0)

        # print(f"Pre-residual shape: {x0.shape}")
        # print(f"Output shape: {x.shape}")

        x = x + x0
        x = self.relu(x)
        return x


class ResnetLarge(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_residual_block=[3, 4, 6, 3],
        num_class=1000
    ):
        super(ResnetLarge, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, 256,
            kernel_size=7, stride=2,
            padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resnet, out_channels = self.__layers(num_residual_block)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(
            in_features=out_channels,
            out_features=num_class,
            bias=True
        )
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.resnet(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sm(x)
        # print(x.shape)
        return x

    def __layers(self, num_residual_block):
        layer = []
        layer += [BasicBlockLarge()]*num_residual_block[0]
        in_channels = 256
        out_channels = 0

        for numOfBlock in num_residual_block[1:]:
            stride = [2, 1, 1]
            downsample = True
            out_channels = in_channels*2
            for _ in range(numOfBlock):
                layer.append(BasicBlockLarge(
                    in_channels, out_channels,
                    stride, down_sample=downsample
                ))
                in_channels = out_channels
                downsample = False
                stride = [1, 1, 1]

        return nn.Sequential(*layer), out_channels
