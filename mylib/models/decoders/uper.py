from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as nnf

from monai.networks.blocks import Convolution

class UPerHead(nn.Module):
    def __init__(self, in_channels: Sequence[int], channels: int, pool_scales: Sequence[int]):
        super().__init__()

        self.psp_modules = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(scale),
                Convolution(spatial_dims=3, in_channels=in_channels[-1], out_channels=channels, kernel_size=1),
            )
            for scale in pool_scales
        ])
        self.psp_bottleneck = Convolution(
            spatial_dims=3,
            in_channels=len(pool_scales) * channels + in_channels[-1],
            out_channels=channels,
            kernel_size=3,
        )
        self.lateral_convs = nn.ModuleList([
            Convolution(spatial_dims=3, in_channels=in_c, out_channels=channels, kernel_size=1)
            for in_c in in_channels[:-1]
        ])
        self.fpn_convs = nn.ModuleList([
            Convolution(
                spatial_dims=3,
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
            )
            for _ in range(len(in_channels) - 1)
        ])
        self.fpn_bottleneck = Convolution(
            spatial_dims=3,
            in_channels=len(in_channels) * channels,
            out_channels=channels,
            kernel_size=3,
        )

    def psp_forward(self, x: torch.Tensor):
        psp_outs = [
            nnf.interpolate(module(x), x.shape[2:], mode='trilinear')
            for module in self.psp_modules
        ]
        psp_outs.append(x)
        output = self.psp_bottleneck(torch.cat(psp_outs, dim=1))
        return output

    def forward(self, hidden_states: list[torch.Tensor], *args, **kwargs):
        laterals = [
            conv(x)
            for x, conv in zip(hidden_states, self.lateral_convs)
        ]
        laterals.append(self.psp_forward(hidden_states[-1]))

        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += nnf.interpolate(laterals[i], laterals[i - 1].shape[2:], mode='trilinear')

        fpn_outs = [
            conv(lateral)
            for conv, lateral in zip(self.fpn_convs, laterals)
        ]
        fpn_outs.append(laterals[-1])
        for i in range(1, len(fpn_outs)):
            fpn_outs[i] = nnf.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:], mode='trilinear')
        output = self.fpn_bottleneck(torch.cat(fpn_outs, dim=1))
        return [output]
