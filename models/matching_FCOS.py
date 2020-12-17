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

        for x, in_conv, out_conv in zip(xs[-2::-1], self.in_convs[-2::-1], self.out_convs[-2::-1]):
            upsampled = F.interpolate(inner, scale_factor=2, mode='nearest')

            inner = in_conv(x)
            out = out_conv(inner)
            outs.insert(0, out)

            inner = inner + upsampled

        return outs


class Scale(nn.Module):
    def __init__(self, init_val):
        super(Scale, self).__init__()

        self.init_val = init_val
        self.scale = nn.Parameter(torch.tensor([self.init_val]))

    def forward(self, x):
        out = self.scale * x
        return out


class MatchingFCOSHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(MatchingFCOSHead, self).__init__()

        self.in_channels = in_channels
        self.n_classes = n_classes

        self.n_convs = 4

        self.cos_similarity = nn.CosineSimilarity(dim=1)

        self.cls_head = []
        self.bbox_head = []

        for i in range(self.n_convs):
            self.cls_head.append(
                nn.Conv2d(bool(i) * in_channels + (not bool(i)) * 1, in_channels,
                          kernel_size=3, padding=1, bias=False)
            )
            self.cls_head.append(nn.GroupNorm(32, in_channels))
            self.cls_head.append(nn.ReLU())

            self.bbox_head.append(
                nn.Conv2d(bool(i) * in_channels + (not bool(i)) * 1, in_channels,
                          kernel_size=3, padding=1, bias=False)
            )
            self.bbox_head.append(nn.GroupNorm(32, in_channels))
            self.bbox_head.append(nn.ReLU())

        self.cls_head = nn.Sequential(*self.cls_head)
        self.bbox_head = nn.Sequential(*self.bbox_head)

        self.cls_conv = nn.Conv2d(in_channels, n_classes, kernel_size=3, padding=1)
        self.bbox_conv = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.scales = nn.ModuleList([Scale(1.) for _ in range(4)])

    def forward(self, query_feats, support_reprs):
        cls_outs = []
        bbox_outs = []
        centerness_outs = []

        for query_feat, support_repr, scale in zip(query_feats, support_reprs, self.scales):
            sim_map = self.cos_similarity(query_feat, support_repr).unsqueeze(1)

            cls_head_out = self.cls_head(sim_map)
            cls_outs.append(self.cls_conv(cls_head_out))

            bbox_head_out = self.bbox_head(sim_map)
            bbox_outs.append(
                torch.exp(scale(self.bbox_conv(bbox_head_out)))
            )

            centerness_outs.append(self.centerness_conv(cls_head_out))

        return cls_outs, bbox_outs, centerness_outs


class MatchingFCOS(nn.Module):
    def __init__(self, backbone):
        super(MatchingFCOS, self).__init__()

        self.backbone = backbone
        self.fpn = FPN(in_channels_list=[512, 1024, 2048], out_channels=256)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.head = MatchingFCOSHead(in_channels=256, n_classes=10)

    def forward(self, query, support):
        x = self.backbone(query)
        query_feats = self.fpn(x)
        support_feats = self.fpn(self.backbone(support))

        support_reprs = list(map(self.avg_pool, support_feats))

        cls_outs, bbox_outs, centerness_outs = self.head(query_feats, support_reprs)

        return cls_outs, bbox_outs, centerness_outs

