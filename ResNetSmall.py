import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class BasicBlockSmall(nn.Module):
    def __init__(
        self,
        in_features=256,
        out_features=256,
        stride=[1, 1],
        down_sample=False
    ):
        super(BasicBlockSmall, self).__init__()
        self.conv1 = nn.Conv2d(
            in_features, out_features,
            kernel_size=3, stride=stride[0],
            padding=1, bias=False
        )
        # weight layer
        self.bn1 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(
            out_features, out_features,
            kernel_size=3, stride=stride[1],
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_features)

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
        x0 = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample is not None:
            x0 = self.downsample(x0)

        x = x + x0
        x = self.relu(x)
        return x


class ResnetSmall(nn.Module):
    def __init__(
        self,
        in_channels=3,
        num_residual_block=[3, 4, 6, 3],
        num_class=1000
    ):
        super(ResnetSmall, self).__init__()

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
        layer += [BasicBlockSmall()]*num_residual_block[0]
        in_channels = 256
        out_channels = 0

        for numOfBlock in num_residual_block[1:]:
            stride = [2, 1]
            downsample = True
            out_channels = in_channels*2
            for _ in range(numOfBlock):
                layer.append(BasicBlockSmall(
                    in_channels, out_channels,
                    stride, down_sample=downsample
                ))
                in_channels = out_channels
                downsample = False
                stride = [1, 1]

        return nn.Sequential(*layer), out_channels
