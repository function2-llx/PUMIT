from __future__ import annotations

from collections.abc import Sequence
from typing import Type

import torch
from torch import nn
from torch.nn import LayerNorm

from monai.networks.blocks import Convolution, PatchEmbed as PatchEmbedBase, ResidualUnit
from monai.networks.layers import Act
from monai.networks.nets.swin_unetr import BasicLayer, PatchMergingV2
from monai.umei import Backbone, BackboneOutput

__all__ = ['SwinTransformer']

from umei.utils import ChannelFirst, channel_first, channel_last

class PatchEmbed(PatchEmbedBase):
    def __init__(
        self,
        patch_size: Sequence[int] | int,
        in_chans: int = 1,
        embed_dim: int = 48,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
        conv: bool = True,
    ) -> None:
        super().__init__(patch_size, in_chans, embed_dim, norm_layer, spatial_dims)
        if conv:
            # TODO: support for patch size 4!
            assert all(x == 2 for x in patch_size)
            assert norm_layer is None
            self.proj = nn.Sequential(
                Convolution(
                    spatial_dims=3,
                    in_channels=in_chans,
                    out_channels=embed_dim,
                    kernel_size=3,
                    strides=2,
                    act=Act.GELU,
                ),
                ResidualUnit(
                    spatial_dims=3,
                    in_channels=embed_dim,
                    out_channels=embed_dim,
                    kernel_size=3,
                    strides=1,
                    subunits=2,
                    act=Act.GELU,
                )
            )


class SwinTransformer(nn.Module):
    """
    Modify from MONAI implementation, support 3D only
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.LayerNorm] = nn.LayerNorm,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        resample_cls: Type[PatchMergingV2] = PatchMergingV2,
        conv_stem: bool = False,
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        assert spatial_dims == 3
        num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm and not conv_stem else None,
            spatial_dims=spatial_dims,
            conv=conv_stem,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList([
            BasicLayer(
                dim=embed_dim << i_layer,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]): sum(depths[:i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=resample_cls if i_layer + 1 < num_layers else None,
                use_checkpoint=use_checkpoint,
            )
            for i_layer in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            norm_layer(embed_dim << i)
            for i in range(num_layers)
        ])
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward_layers(self, x: torch.Tensor) -> list[torch.Tensor]:
        hidden_states = []
        for layer, norm in zip(self.layers, self.norms):
            z = layer(x)
            if isinstance(z, tuple):
                z, z_ds = z
            z = channel_last(z)
            z = norm(z)
            z = channel_first(z)
            hidden_states.append(z)
            x = z_ds

        return hidden_states

    def pool_hidden_states(self, hidden_states: list[torch.Tensor]):
        return self.avg_pool(hidden_states[-1]).flatten(1)

    def forward(self, x: torch.Tensor, *args) -> BackboneOutput:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        feature_maps = self.forward_layers(x)
        return BackboneOutput(
            cls_feature=self.pool_hidden_states(feature_maps),
            feature_maps=feature_maps,
        )
