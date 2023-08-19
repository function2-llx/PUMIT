from __future__ import annotations

from collections.abc import Sequence
import itertools as it

import numpy as np
import torch
from torch import nn

from mylib.models.init import init_common
from mylib.models.layers import LayerNormNd, Norm
from .common3d import SwinLayer

__all__ = []

class SwinBackbone(nn.Module):
    """
    Modify from MONAI implementation, support 3D only
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microssoft/Swin-Transformer
    """

    def __init__(
        self,
        in_channels: int,
        patch_size: int | Sequence[int],
        layer_channels: int | Sequence[int],
        window_sizes: Sequence[int | Sequence[int]],
        layer_depths: Sequence[int],
        strides: Sequence[int | Sequence[int]],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
    ):
        """
        Args:
            in_channels: dimension of input channels.
            layer_channels: number of channels for each layer.
            layer_depths: number of block in each layer.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.patch_embed = nn.Conv3d(in_channels, layer_channels[0], patch_size, patch_size)
        num_layers = len(layer_depths)
        if isinstance(layer_channels, int):
            layer_channels = [layer_channels << i for i in range(num_layers)]
        layer_drop_path_rates = np.split(
            np.linspace(0, drop_path_rate, sum(layer_depths)),
            np.cumsum(layer_depths[:-1]),
        )
        self.layers = nn.ModuleList([
            SwinLayer(
                layer_channels[i],
                layer_depths[i],
                num_heads[i],
                _max_window_size := window_sizes[i],
                layer_drop_path_rates[i],
                mlp_ratio,
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                Norm.LAYER,
                use_checkpoint,
            )
            for i in range(num_layers)
        ])

        self.downsamplings = nn.ModuleList([
            nn.Conv3d(layer_channels[i], layer_channels[i + 1], kernel_size=strides[i], stride=strides[i])
            for i in range(num_layers - 1)
        ])
        self.downsamplings.append(nn.Identity())

        # SwinLayer is pre-norm, additional norm for their output feature maps
        self.norms = nn.ModuleList([
            LayerNormNd(layer_channels[i])
            for i in range(num_layers)
        ])

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.apply(init_common)

    def no_weight_decay(self):
        nwd = set()
        for name, _ in self.named_parameters():
            if 'relative_position_bias_table' in name:
                nwd.add(name)
        return nwd

    def forward_layers(self, x: torch.Tensor):
        feature_maps = []
        for layer, norm, downsampling in zip(
            self.layers[self.num_conv_layers:],
            self.norms[self.num_conv_layers:],
            self.downsamplings[self.num_conv_layers:],
        ):
            x = layer(x)
            x = norm(x)
            feature_maps.append(x)
            x = downsampling(x)

        return feature_maps

    def forward(self, x: torch.Tensor):
        feature_maps = []
        x = self.patch_embed(x)
        for layer, norm, downsampling in zip(it.chain(self.layers), self.norms, self.downsamplings):
            x = layer(x)
            x = norm(x)
            feature_maps.append(x)
            x = downsampling(x)

        return feature_maps
