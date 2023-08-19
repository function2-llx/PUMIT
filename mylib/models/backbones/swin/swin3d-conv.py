from __future__ import annotations

from collections.abc import Sequence
import itertools as it

import numpy as np
import torch
from torch import nn

from monai.networks.layers import get_act_layer, get_norm_layer

from mylib.models.adaptive_resampling import AdaptiveDownsampling
from mylib.models.blocks import BasicConvLayer, get_conv_layer
from mylib.models.init import init_common
from mylib.models.layers import Act, LayerNormNd, Norm
from .common3d import SwinLayer

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
        num_conv_layers: int,
        layer_channels: int | Sequence[int],
        kernel_sizes: Sequence[int | Sequence[int]],
        layer_depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        use_checkpoint: bool = False,
        *,
        stem_kernel: int = 3,
        stem_stride: int = 1,
        stem_channels: int = None,
        conv_in_channels: int | None = None,
        conv_norm: str = 'instance',
        conv_act: str = 'leakyrelu',
        # **_kwargs,
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
        num_layers = len(layer_depths)
        if isinstance(layer_channels, int):
            layer_channels = [layer_channels << i for i in range(num_layers)]

        if stem_stride == 1:
            self.stem = get_conv_layer(in_channels, layer_channels[0], stem_kernel, stem_stride, norm=conv_norm, act=conv_act)
        else:
            if stem_channels is None:
                stem_channels = layer_channels[0] >> 1
            self.stem = nn.Sequential(
                *[
                    AdaptiveDownsampling(
                        in_channels if i == 0 else stem_channels,
                        stem_channels if i < stem_stride.bit_length() - 2 else layer_channels[0],
                        stem_kernel,
                    )
                    for i in range(stem_stride.bit_length() - 1)
                ],
                get_norm_layer(conv_norm, 3, layer_channels[0]),
                get_act_layer(conv_act),
            )

        if conv_in_channels is not None:
            self.conv_in_layers = nn.ModuleList()
            if stem_stride > 1:
                self.conv_in_layers.append(BasicConvLayer(
                    _num_blocks := 1,
                    in_channels,
                    conv_in_channels,
                    _kernel_size := 3,
                    drop_rate,
                    Norm.INSTANCE,
                    Act.LEAKYRELU,
                ))
                for i in range(1, stem_stride.bit_length() - 1):
                    self.conv_in_layers.append(nn.Sequential(
                        AdaptiveDownsampling(
                            conv_in_channels,
                            conv_in_channels,
                            kernel_size=3,
                        ),
                        BasicConvLayer(
                            _num_blocks := 1,
                            conv_in_channels,
                            conv_in_channels,
                            _kernel_size := 3,
                            drop_rate,
                            Norm.INSTANCE,
                            Act.LEAKYRELU,
                        )
                    ))
        else:
            self.conv_in_layers = None

        layer_drop_path_rates = np.split(
            np.linspace(0, drop_path_rate, sum(layer_depths)),
            np.cumsum(layer_depths[:-1]),
        )

        self.num_conv_layers = num_conv_layers
        self.layers = nn.ModuleList([
            BasicConvLayer(
                layer_depths[i],
                layer_channels[i],
                layer_channels[i],
                kernel_sizes[i],
                drop_rate,
                conv_norm,
                conv_act,
                layer_drop_path_rates[i],
            ) if i < num_conv_layers
            else SwinLayer(
                layer_channels[i],
                layer_depths[i],
                num_heads[i],
                _max_window_size := kernel_sizes[i],
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
            AdaptiveDownsampling(
                layer_channels[i],
                layer_channels[i + 1],
                kernel_size=2,
            )
            for i in range(num_layers - 1)
        ])

        # dummy, the last downsampling is not used for segmentation task
        self.downsamplings.append(nn.Identity())

        # SwinLayer is pre-norm, additional norm for their outputs
        self.norms = nn.ModuleList([
            nn.Identity() if i < num_conv_layers
            else LayerNormNd(layer_channels[i])
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

    def patch_embed(self, x: torch.Tensor):
        x = self.stem(x)
        for layer, norm, downsampling in zip(self.layers[:self.num_conv_layers], self.norms, self.downsamplings):
            x = layer(x)
            x = norm(x)
            x = downsampling(x)
        return x

    @property
    def embed_dim(self):
        if self.num_conv_layers > 1:
            return self.downsamplings[self.num_conv_layers - 1]
        else:
            raise NotImplementedError

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

    def forward(self, x: torch.Tensor, *args) -> BackboneOutput:
        feature_maps = []
        if self.conv_in_layers is not None:
            y = x
            for layer in self.conv_in_layers:
                y = layer(y)
                feature_maps.append(y)

        x = self.stem(x)

        for layer, norm, downsampling in zip(it.chain(self.layers), self.norms, self.downsamplings):
            x = layer(x)
            x = norm(x)
            feature_maps.append(x)
            x = downsampling(x)

        return BackboneOutput(
            cls_feature=self.pool(feature_maps[-1]),
            feature_maps=feature_maps,
        )
