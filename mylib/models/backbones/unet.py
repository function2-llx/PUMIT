from collections.abc import Sequence

import torch
from torch import nn

from monai.networks.blocks import UnetBasicBlock
from mylib.models.layers import Act, Norm

class UNetBackbone(nn.Module):
    def __init__(
        self,
        in_channels: int,
        layer_channels: list[int],
        kernel_sizes: list[int | Sequence[int]],
        strides: list[int | Sequence[int]],
        norm: str | tuple = Norm.INSTANCE,
        act: str | tuple = Act.LEAKYRELU,
    ):
        super().__init__()
        num_layers = len(layer_channels)
        self.layers = nn.ModuleList([
            UnetBasicBlock(
                3,
                layer_channels[i - 1] if i else in_channels,
                layer_channels[i],
                kernel_sizes[i],
                strides[i],
                norm,
                act,
            )
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, *args, **kwargs) -> BackboneOutput:
        feature_maps = []
        for layer in self.layers:
            x = layer(x)
            feature_maps.append(x)
        return BackboneOutput(feature_maps=feature_maps)
