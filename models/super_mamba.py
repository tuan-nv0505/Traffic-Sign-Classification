from collections import OrderedDict

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from torchinfo import summary

from models.conv import ConvNet
from models.vmamba import PatchMerging2D, VSSBlock

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args).contiguous()

class SuperMamba(nn.Module):
    def __init__(self, dims=3, depth=4, num_classes=43):
        super().__init__()
        self.depth = depth
        self.preembd = ConvNet()
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.depth + 1)]
        self.num_features = dims[-1]
        self.dims = dims
        self.layers = nn.ModuleList()
        for i_layer in range(self.depth):
            downsample = PatchMerging2D(
                dim=self.dims[i_layer],
                out_dim=self.dims[i_layer + 1],
                norm_layer=nn.LayerNorm,
            )
            vss_block = VSSBlock(hidden_dim=self.dims[i_layer + 1])
            self.layers.append(downsample)
            self.layers.append(vss_block)

        self.classifier = nn.Sequential(OrderedDict(
            permute_in=Permute(0, 2, 3, 1),
            norm=nn.LayerNorm(self.num_features),  # B, H, W, C
            permute_out=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.preembd(x)
        for layers in self.layers:
            x = layers(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    x = torch.randn(16, 3, 32, 32)
    model = SuperMamba(dims=3, depth=4, num_classes=43)
    out = model(x)
    summary(model, (32, 3, 32, 32))