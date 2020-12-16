from typing import List

import torch
from torch import nn
from torch.nn import functional as F


class FPNTopBlocks(nn.Module):
    def __init__(self, out_channels):
        super(FPNTopBlocks, self).__init__()

        self.p6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, x):
        outs = []

        out = self.p6(x)
        outs.append(out)
        out = self.p7(out)
        outs.append(out)

        return outs


class FPN(nn.Module):
    def __init__(self, in_channels_list: List[int], out_channels: int):
        super(FPN, self).__init__()
        self.in_channels_list = in_channels_list
        self.out_channels = out_channels

        self.in_convs = nn.ModuleList()
        self.out_convs = nn.ModuleList()

        for in_channels in self.in_channels_list:
            self.in_convs.append(nn.Conv2d(in_channels=in_channels,
                                           out_channels=self.out_channels,
                                           kernel_size=1))

            self.out_convs.append(nn.Conv2d(in_channels=self.out_channels,
                                            out_channels=self.out_channels,
                                            kernel_size=3,
                                            padding=1))

        self.top_blocks = FPNTopBlocks(out_channels)

    def forward(self, xs):
        outs = []

        inner = self.in_convs[-1](xs[-1])
        out = self.out_convs[-1](inner)
        outs.insert(0, out)

        out = self.top_blocks(inner)
        outs.extend(out)

        for x, in_conv, out_conv in zip(xs[:-1][::-1], self.in_convs[:-1][::-1], self.out_convs[:-1][::-1]):
            upsampled = F.interpolate(inner, scale_factor=2, mode='nearest')

            inner = in_conv(x)
            out = out_conv(inner)
            outs.insert(0, out)

            inner = inner + upsampled

        return outs


class MatchingFCOSHead(nn.Module):
    pass


class MatchingFCOS(nn.Module):
    def __init__(self):
        super(MatchingFCOS, self).__init__()

        self.fpn = FPN(in_channels_list=[2048, 1024, 512], out_channels=256)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.head = nn.Identity()

    def forward(self, q, s):
        q_features = self.fpn(q)
        s_features = self.fpn(s)
        s_features = list(map(self.avg_pool, s_features))

        return None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
