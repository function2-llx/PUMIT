from collections.abc import Sequence

import torch
from torch import nn

from luolib.models import spadop

__all__ = [
    'PatchDiscriminatorBase',
    'PatchDiscriminator',
    'SimplePatchDiscriminator',
    'SwinPatchDiscriminator',
]

class PatchDiscriminatorBase(nn.Module):
    # save state
    grad_vq_ema: torch.Tensor
    grad_gan_ema: torch.Tensor

    def forward(self, x: spadop.SpatialTensor):
        """
        Returns:
            patch wise logit of (batch_size, 1, *spatial_shape / patch_size)
        """

class PatchDiscriminator(PatchDiscriminatorBase):
    def __init__(self, in_channels: int, num_downsample_layers: int, base_channels: int):
        super().__init__()
        layer_channels = [
            base_channels << min(i, 3)
            for i in range(num_downsample_layers + 1)
        ]
        self.main = nn.Sequential()
        self.main.extend([
            spadop.InputConv3D(in_channels, layer_channels[0], (3, 4, 4), stride=2),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        for i in range(1, num_downsample_layers + 1):
            stride = 2 if i < num_downsample_layers else 1
            self.main.extend([
                spadop.Conv3d(
                    layer_channels[i - 1], layer_channels[i], (3, 4, 4), stride, padding=1, bias=False,
                ),
                nn.InstanceNorm3d(layer_channels[i], affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.main.append(
            spadop.Conv3d(layer_channels[-1], 1, (3, 4, 4), stride=1, padding=1),
        )
        # init_common(self)

    def forward(self, x: spadop.SpatialTensor):
        return self.main(x)

class SimplePatchDiscriminator(nn.Sequential, PatchDiscriminatorBase):
    def __init__(self, in_channels: int, layer_channels: Sequence[int], num_post_convs: int = 1):
        super().__init__()
        self.extend([
            spadop.InputConv3D(in_channels, layer_channels[0], 3, stride=2, padding=1),
            nn.GroupNorm(16, layer_channels[0]),
            nn.LeakyReLU(inplace=True),
        ])
        for i in range(1, len(layer_channels)):
            self.extend([
                spadop.InputConv3D(layer_channels[i - 1], layer_channels[i], 3, stride=2, padding=1),
                nn.GroupNorm(16, layer_channels[i]),
                nn.LeakyReLU(inplace=True),
            ])
        for _ in range(num_post_convs):
            self.extend([
                spadop.InputConv3D(layer_channels[-1], layer_channels[-1], 3, padding=1),
                nn.GroupNorm(16, layer_channels[-1]),
                nn.LeakyReLU(inplace=True),
            ])
        self.append(spadop.InputConv3D(layer_channels[-1], 1, 3, padding=1))

class SwinPatchDiscriminator(nn.Sequential, PatchDiscriminatorBase):
    def __init__(
        self,
        in_channels: int,
        patch_size: int,
        patch_embed_kernel_size: int,
        dim: int,
        depth: int,
        num_heads: int,
    ):
        super().__init__(
            spadop.PatchEmbed(
                in_channels, dim, patch_size, patch_embed_kernel_size, True,
            ),
            spadop.SwinLayer(dim, depth, num_heads, 4, last_norm=True),
            spadop.Conv3d(dim, 1, 1),
        )
