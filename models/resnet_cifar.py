# https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.datasets import get_normalize_layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv_layer(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, conv_layer, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv_layer(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv_layer(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_layer(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv_layer(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, conv_layer, linear_layer, block, num_blocks, num_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv_layer = conv_layer
        self.in_channels = in_channels

        self.conv1 = conv_layer(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = linear_layer(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.conv_layer, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8 if self.in_channels == 3 else 7)  # Special treatment for MNIST datat
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# NOTE: Only supporting default (kaiming_init) initialization.
def resnet18(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for ResNets"
    return ResNet(conv_layer, linear_layer, BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet20(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for ResNets"
    return ResNet(conv_layer, linear_layer, BasicBlock, [3, 3, 3], **kwargs)


def resnet34(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for ResNets"
    return ResNet(conv_layer, linear_layer, BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for ResNets"
    return ResNet(conv_layer, linear_layer, Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for ResNets"
    return ResNet(conv_layer, linear_layer, Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet110(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for ResNets"
    if args.normalize:
        return nn.Sequential(
            get_normalize_layer(dataset=args.dataset),
            ResNet(conv_layer, linear_layer, BasicBlock, [18, 18, 18, 18], **kwargs)
        )
    else:
        return ResNet(conv_layer, linear_layer, BasicBlock, [18, 18, 18, 18], **kwargs)


def resnet152(conv_layer, linear_layer, init_type, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for ResNets"
    return ResNet(conv_layer, linear_layer, Bottleneck, [3, 8, 36, 3], **kwargs)


def test():
    net = resnet20(nn.Conv2d, nn.Linear, "kaiming_normal", None)
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())


if __name__ == "__main__":
    test()
