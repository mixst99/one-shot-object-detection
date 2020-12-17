import torch
from torch import nn

from torchvision import models


class ResNet50_backbone(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_backbone, self).__init__()

        self.pull_off_layers = {'layer2': 'C3', 'layer3': 'C4', 'layer4': 'C5'}
        self.pretrained = pretrained
        self.backbone = models.resnet50(pretrained=pretrained)

    def forward(self, x):
        outs = []

        out = x
        for name, module in self.backbone.named_children():
            out = module(out)

            if name in self.pull_off_layers:
                outs.append(out)

            if len(outs) == len(self.pull_off_layers):
                break

        return outs


class Backbone(nn.Module):
    def __init__(self, backbone, pull_off_layers: dict):
        super(Backbone, self).__init__()

        self.pull_off_layers = pull_off_layers
        self.backbone = backbone

    def forward(self, x):
        outs = {}

        out = x
        for name, module in self.backbone.named_children():
            out = module(out)

            if name in self.pull_off_layers:
                outs[self.pull_off_layers[name]] = out

            if len(outs) == len(self.pull_off_layers):
                break

        return outs
