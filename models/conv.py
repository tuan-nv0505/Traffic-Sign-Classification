import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    def __init__(self, C, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Linear(
            in_features=C,
            out_features=C // reduction,
            bias=False
        )
        self.fc_2 = nn.Linear(
            in_features=C // reduction,
            out_features=C,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape

        out = self.avg_pool(x).view(B, C)
        out = self.fc_1(out)
        out = F.relu(out)
        out = self.fc_2(out)
        out = F.sigmoid(out)

        return x * out.view(B, C, 1, 1)