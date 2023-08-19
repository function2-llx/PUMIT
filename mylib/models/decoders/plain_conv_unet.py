from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from mylib.types import spatial_param_seq_t
from ..blocks import UNetUpLayer, get_conv_layer
from ..init import init_common
from ..layers import Act, Norm

__all__ = ['PlainConvUNetDecoder']

class PlainConvUNetDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        layer_channels: list[int],
        kernel_sizes: spatial_param_seq_t[int],
        upsample_strides: spatial_param_seq_t[int],
        norm: tuple | str | None = Norm.INSTANCE,
        act: tuple | str | None = Act.LEAKYRELU,
        lateral_channels: Sequence[int] | None = None,
        lateral_kernel_sizes: spatial_param_seq_t[int] | None = None,
    ) -> None:
        super().__init__()
        num_layers = len(layer_channels) - 1
        self.layers = nn.ModuleList([
            UNetUpLayer(spatial_dims, layer_channels[i + 1], layer_channels[i], kernel_sizes[i], upsample_strides[i])
            for i in range(num_layers)
        ])
        if lateral_kernel_sizes is None:
            self.laterals = nn.ModuleList([
                nn.Identity()
                for _ in range(num_layers + 1)
            ])
        else:
            if lateral_channels is None:
                lateral_channels = layer_channels
            self.laterals = nn.ModuleList([
                get_conv_layer(spatial_dims, lc, channels, kernel_size, norm=norm, act=act)
                for lc, channels, kernel_size in zip(lateral_channels, layer_channels, lateral_kernel_sizes)
            ])

        self.apply(init_common)

    def forward(self, backbone_feature_maps: Sequence[torch.Tensor]):
        feature_maps = []
        backbone_feature_maps = [
            lateral(x)
            for x, lateral in zip(backbone_feature_maps, self.laterals)
        ]
        x = backbone_feature_maps[-1]
        for layer, skip in zip(self.layers[::-1], backbone_feature_maps[-2::-1]):
            x = layer(x, skip)
            feature_maps.append(x)
        return feature_maps
