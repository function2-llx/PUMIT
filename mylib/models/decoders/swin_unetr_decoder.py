from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from monai.networks.blocks import UnetResBlock, UnetrUpBlock

class SwinUnetrDecoder(nn.Module):
    def __init__(
        self,
        in_channels: Optional[int] = None,
        feature_size: int = 24,
        num_layers: int = 4,
        norm_name: tuple | str = "instance",
        spatial_dims: int = 3,
        input_stride: Optional[Sequence[int] | int] = None,
    ) -> None:
        super().__init__()
        assert spatial_dims == 3

        self.bottleneck = UnetResBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size << num_layers - 1,
            out_channels=feature_size << num_layers - 1,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )

        self.ups = nn.ModuleList([
            UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size << i,
                out_channels=feature_size << i - 1,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
            for i in range(1, num_layers)
        ])

        self.lateral_convs = nn.ModuleList([
            UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size << i,
                out_channels=feature_size << i,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
            )
            for i in range(num_layers - 1)
        ])

        if input_stride is not None:
            self.last_lateral = UnetResBlock(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=feature_size,
                kernel_size=3,
                stride=input_stride,
                norm_name=norm_name,
            )

            # don't prepend to self.ups, or will not load pre-trained weights properly
            self.last_up = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=True,
            )
        else:
            self.last_lateral = None

    def forward(self, hidden_states: list[torch.Tensor], x_in: torch.Tensor) -> DecoderOutput:
        x = self.bottleneck(hidden_states[-1])
        feature_maps = []
        for z, lateral_conv, up in zip(hidden_states[-2::-1], self.lateral_convs[::-1], self.ups[::-1]):
            up: UnetrUpBlock
            z = lateral_conv(z)
            x = up.forward(x, z)
            feature_maps.append(x)
        if self.last_lateral is not None:
            x = self.last_up(x, self.last_lateral(x_in))
            feature_maps.append(x)
        return DecoderOutput(feature_maps)
