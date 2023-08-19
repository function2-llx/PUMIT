from typing import Type

import torch
from torch import nn

from monai.networks.layers import Norm, Act
from mylib.utils import ChannelFirst, ChannelLast

# make PyCharm come here
Act = Act
Norm = Norm

class LayerNormNd(nn.Sequential):
    def __init__(self, num_channels: int, contiguous: bool = True):
        # assume input shape is (batch, channel, *spatial)
        super().__init__(
            ChannelLast(),
            nn.LayerNorm(num_channels),
            ChannelFirst(),
        )
        if contiguous:
            self.append(Contiguous())

class Contiguous(nn.Module):
    def forward(self, x: torch.Tensor):
        return x.contiguous()

@Norm.factory_function("layernd")
def layer_factory(_dim) -> Type[LayerNormNd]:
    return LayerNormNd
