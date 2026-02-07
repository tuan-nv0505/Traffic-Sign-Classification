import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import repeat

from models.conv import SE


class VSSBlock(nn.Module):
    def __init__(self,
                 # basic dims ===========
                 d_model=96,
                 d_state=16,
                 ssm_ratio=2,
                 ssm_rank_ratio=2,
                 delta_rank="auto",
                 activation_layer=nn.SiLU,
                 # conv_dw ===============
                 d_conv=3,
                 conv_bias=True,
                 # ======================
                 dropout=0.0,
                 bias=False,
                 # delta init ==============
                 delta_min=0.001,
                 delta_max=0.1,
                 delta_init="random",
                 delta_scale=1.0,
                 delta_init_floor=1e-4,
                 simple_init=False,
                 ):
        super().__init__()
        d_expand = d_model * ssm_ratio
        d_inner = int(min(ssm_ratio, ssm_rank_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.delta_rank = math.ceil(d_model / 16) if delta_rank == "auto" else delta_rank
        self.d_conv = d_conv

        self.in_projection = nn.Linear(in_features=d_model, out_features=d_expand * 2, bias=bias)

        self.activation_layer = activation_layer

        self.conv2d = nn.Conv2d(
            in_channels=d_expand,
            out_channels=d_expand,
            kernel_size=self.d_conv,
            padding=(self.d_conv - 1) // 2,
            groups=d_expand
        )

        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(in_channels=d_expand, out_channels=d_inner, kernel_size=1, bias=False)
            self.out_rank = nn.Linear(in_features=d_inner, out_features=d_expand, bias=False)

        self.x_projection = [
            nn.Linear(in_features=d_inner, out_features=self.delta_rank + self.d_state * 2, bias=False)
            for _ in range(4)
        ]
        self.x_projection_weight = nn.Parameter(torch.stack([k.weight for k in self.x_projection], dim=1))
        del self.x_projection

        self.delta_projection = [
            self.delta_init(self.delta_rank, d_inner)
            for _ in range(4)
        ]
        self.delta_projection_weight = nn.Parameter(torch.stack([k.weight for k in self.delta_projection], dim=0))
        self.delta_projection_bias = nn.Parameter(torch.stack([k.bias for k in self.delta_projection], dim=0))
        del self.delta_projection

        self.A_log = self.A_log_init(d_state, d_inner)
        self.Ds = self.D_init(d_inner)

        self.out_projection = nn.Linear(d_expand, d_model, bias=bias)
        self.effn = EFFN(d_expand)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if simple_init:
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((4 * d_inner, self.d_state)))
            self.delta_projection_weight = nn.Parameter(torch.randn((4, d_inner, self.dt_rank)))
            self.delta_projection_bias = nn.Parameter(torch.randn((4, d_inner)))


    @staticmethod
    def delta_init(delta_rank,
                   d_inner,
                   delta_scale=1.0,
                   delta_init="random",
                   delta_min=0.001,
                   delta_max=0.1,
                   delta_init_floor=1e-4,
                   **factory_kwargs
    ):
        delta_projection = nn.Linear(delta_rank, d_inner, bias=True, **factory_kwargs)

        delta_init_std = delta_rank ** -0.5 * delta_scale
        if delta_init == "constant":
            nn.init.constant_(delta_projection.weight, delta_init_std)
        elif delta_init == "random":
            nn.init.uniform_(delta_projection.weight, -delta_init_std, delta_init_std)
        else:
            raise NotImplementedError

        delta = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(delta_max) - math.log(delta_min))
            + math.log(delta_min)
        ).clamp(min=delta_init_floor)
        inv_delta = delta + torch.log(-torch.expm1(-delta))
        with torch.no_grad():
            delta_projection.bias.copy_(inv_delta)

        return delta_projection

    @staticmethod
    def A_log_init(d_state, d_inner):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        A_log = repeat(A_log, "d n -> k d n", k=4)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner):
        D = torch.ones(d_inner)
        D = repeat(D, "n1 -> k n1", k=4)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D


class PathMerging2D(nn.Module):
    pass

class EFFN(nn.Module):
    def __init__(self, in_channels):
        super(EFFN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 2, kernel_size=1, padding=0),
            nn.BatchNorm2d(num_features=in_channels * 2),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels * 2, kernel_size=3, padding=1, groups=in_channels * 2),
            nn.BatchNorm2d(num_features=in_channels * 2),
            nn.ReLU()
        )

        self.conv3 = nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels // 2, kernel_size=1, padding=0)

        self.se = SE(in_channels=in_channels // 2, reduction=2)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se(x)
        x = x.permute(0, 2, 3, 1)
        return x


























