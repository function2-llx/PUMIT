from collections.abc import Sequence

import torch
from torch import nn

from mylib.types import spatial_param_seq_t
from ..blocks import BasicConvLayer, UNetUpLayer
from ..layers import Act, Norm

__all__ = ['FullResAdapter']

class FullResAdapter(nn.Module):
    def __init__(
        self,
        inner: nn.Module,
        spatial_dims: int,
        num_input_channels: int,
        layer_channels: Sequence[int],
        kernel_sizes: spatial_param_seq_t[int],
        strides: spatial_param_seq_t[int],
        layer_blocks: list[int] | None = None,
        norm: tuple | str = Norm.INSTANCE,
        act: tuple | str = Act.LEAKYRELU,
    ):
        super().__init__()
        num_layers = len(layer_channels) - 1
        if layer_blocks is None:
            layer_blocks = [1] * num_layers
        self.inner = inner
        self.encode_layers = nn.ModuleList([
            BasicConvLayer(
                spatial_dims=spatial_dims,
                num_blocks=layer_blocks[i],
                in_channels=num_input_channels if i == 0 else layer_channels[i - 1],
                out_channels=layer_channels[i],
                kernel_size=kernel_sizes[i],
                stride=1 if i == 0 else strides[i - 1],
                norm=norm,
                act=act,
                res_block=False,
            )
            for i in range(num_layers)
        ])
        self.decode_layers = nn.ModuleList([
            UNetUpLayer(spatial_dims, layer_channels[i + 1], layer_channels[i], kernel_sizes[i], strides[i])
            for i in range(num_layers)
        ])

    def forward(self, backbone_feature_maps: list[torch.Tensor], x_in: torch.Tensor):
        ret: list[torch.Tensor] = self.inner(backbone_feature_maps)
        encodes = []
        for encode_layer in self.encode_layers:
            x_in = encode_layer(x_in)
            encodes.append(x_in)
        for decode_layer, skip in zip(self.decode_layers[::-1], encodes[::-1]):
            ret.append(decode_layer(ret[-1], skip))
        return ret
