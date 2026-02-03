import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_scan(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape

    scan = torch.stack([
        x.view(B, C, H * W),
        x.view(B, C, H * W).flip(dims=[-1]),
        x.permute(0, 1, 3, 2).reshape(B, C, H * W),
        x.permute(0, 1, 3, 2).reshape(B, C, H * W).flip(dims=[-1])
    ])

    return scan

def cross_merge(y: torch.Tensor, H: int, W: int) -> torch.Tensor:
    B, K, C, L = y.shape
    y1, y2, y3, y4 = y.chunk(4, dim=1)



class VSSBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int, d_conv: int, ratio: int):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * ratio

        self.in_layer_norm = nn.LayerNorm(normalized_shape=d_model)
        self.in_projection = nn.Linear(
            in_features=self.d_model,
            out_features=self.d_inner * 2
        )
        self.conv_dw = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_model,
            kernel_size=self.d_conv,
            padding=(self.d_conv - 1) // 2
        )