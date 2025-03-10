import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ensemble import Subspace, Ensemble
from models.layers import CurvesConv, LinesConv, CurvesLinear, CurvesBN, LinesLinear, LinesBN, SubnetConv, SubnetLinear
from utils.datasets import get_normalize_layer


class BasicBlock(nn.Module):
    def __init__(self, conv_layer, in_planes, out_planes, stride, drop_rate=0.0, bn_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.bn1 = bn_layer(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv_layer(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = bn_layer(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.drop_rate = drop_rate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and conv_layer(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(
            self, nb_layers, in_planes, out_planes, block, conv_layer, stride, dropRate=0.0, bn_layer=nn.BatchNorm2d
    ):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            conv_layer, block, in_planes, out_planes, nb_layers, stride, dropRate, bn_layer=bn_layer
        )

    def _make_layer(
            self, conv_layer, block, in_planes, out_planes, nb_layers, stride, dropRate, bn_layer
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    conv_layer,
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    dropRate,
                    bn_layer=bn_layer
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
            self,
            conv_layer,
            linear_layer,
            bn_layer=nn.BatchNorm2d,
            depth=34,
            num_classes=10,
            widen_factor=10,
            drop_rate=0.0,
    ):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        self.num_classes = num_classes
        # 1st conv before any network block
        self.conv1 = conv_layer(
            3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        # 1st block
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, conv_layer, 1, drop_rate, bn_layer=bn_layer
        )
        # 1st sub-block
        self.sub_block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, conv_layer, 1, drop_rate, bn_layer=bn_layer
        )
        # 2nd block
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, conv_layer, 2, drop_rate, bn_layer=bn_layer
        )
        # 3rd block
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, conv_layer, 2, drop_rate, bn_layer=bn_layer
        )
        # global average pooling and classifier
        self.bn1 = bn_layer(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = linear_layer(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, linear_layer):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        if not isinstance(self.conv1, CurvesConv):
            out = out.view(-1, self.fc.in_features)

        return self.fc(out).reshape(-1, self.num_classes)


# NOTE: Only supporting default (kaiming_init) initialization.
def wrn_28_10(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=28, widen_factor=10, **kwargs)


# 0.1059 * (wrn_28_4 * 3) -> (34, 2)
def wrn_34_2(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    if args.normalize:
        return nn.Sequential(
            get_normalize_layer(dataset=args.dataset),
            WideResNet(conv_layer, linear_layer, depth=34, widen_factor=2, **kwargs)
        )
    else:
        return WideResNet(conv_layer, linear_layer, depth=34, widen_factor=2, **kwargs)


# 0.0616 * (wrn_28_4 * 3) -> (22, 2)
def wrn_10_4(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    if args.normalize:
        return nn.Sequential(
            get_normalize_layer(dataset=args.dataset),
            WideResNet(conv_layer, linear_layer, depth=10, widen_factor=4, **kwargs)
        )
    else:
        return WideResNet(conv_layer, linear_layer, depth=10, widen_factor=4, **kwargs)


# 0.01 * (wrn_28_4 * 3)
def wrn_16_1(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    if args.normalize:
        return nn.Sequential(
            get_normalize_layer(dataset=args.dataset),
            WideResNet(conv_layer, linear_layer, depth=16, widen_factor=1, **kwargs)
        )
    else:
        return WideResNet(conv_layer, linear_layer, depth=16, widen_factor=1, **kwargs)


def wrn_28_4(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    if args.normalize:
        return nn.Sequential(
            get_normalize_layer(dataset=args.dataset),
            WideResNet(conv_layer, linear_layer, depth=28, widen_factor=4, **kwargs)
        )
    else:
        return WideResNet(conv_layer, linear_layer, depth=28, widen_factor=4, **kwargs)


def wrn_28_1(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=28, widen_factor=1, **kwargs)


def wrn_34_10(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=34, widen_factor=10, **kwargs)


def wrn_40_2(conv_layer, linear_layer, init_type, args, **kwargs):
    assert init_type == "kaiming_normal", "only supporting default init for WRN"
    return WideResNet(conv_layer, linear_layer, depth=40, widen_factor=2, **kwargs)


if __name__ == "__main__":
    argv = Namespace()
    argv.normalize = False
    for d in [10, 16, 22, 28, 34]:
        for w in range(1, 10):
            wrn = WideResNet(nn.Conv2d, nn.Linear, depth=d, widen_factor=w)
            print(d, w, round(sum(p.numel() for p in wrn.parameters()) / 18354798, 4))
