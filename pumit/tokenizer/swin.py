from torch import nn

from luolib.models import spadop
from luolib.utils import ChannelFirst, ChannelLast

from .base import VQVisualTokenizer

__all__ = [
    'SwinVQVT',
    'SwinConvVQVT',
]

class SwinVQVT(VQVisualTokenizer):
    def __init__(
        self,
        in_channels: int,
        patch_embed_kernel_size: int,
        encoder_dim: int, encoder_depth: int, encoder_num_heads: int,
        decoder_dim: int, decoder_depth: int, decoder_num_heads: int,
        output_scale: bool,
        grad_ckpt: bool = False,
        *args, **kwargs,
    ):
        """
        Args:
            output_scale: whether to output (log) scale of the distribution (e.g., b for Laplace, σ for normal)
        """
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            spadop.PatchEmbed(in_channels, encoder_dim, 16, patch_embed_kernel_size, True),
            spadop.SwinLayer(
                encoder_dim, encoder_depth, encoder_num_heads, 4,
                last_norm=True, grad_ckpt=grad_ckpt,
            ),
            ChannelLast(),
        )
        self.decoder = nn.Sequential(
            ChannelFirst(),
            spadop.SwinLayer(
                decoder_dim, decoder_depth, decoder_num_heads, 4,
                last_norm=True, grad_ckpt=grad_ckpt,
            ),
            spadop.InversePatchEmbed(
                decoder_dim, in_channels * 2 if output_scale else in_channels,
                16, True,
            ),
        )

    def encode(self, x: spadop.SpatialTensor) -> spadop.SpatialTensor:
        return self.encoder(x)

    def decode(self, z_q: spadop.SpatialTensor) -> spadop.SpatialTensor:
        return self.decoder(z_q)

class ResNetBasicBlock(nn.Module):
    """pre-activation, maybe see timm.models.ResNetV2"""
    def __init__(self, channels: int, num_groups: int):
        super().__init__()
        self.channels = channels
        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            spadop.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups, channels),
            nn.SiLU(),
            spadop.Conv3d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: spadop.SpatialTensor):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        return res + x

class SwinConvVQVT(VQVisualTokenizer):
    def __init__(
        self,
        in_channels: int,
        patch_embed_kernel_size: int,
        dim: int,
        encoder_depth: int,
        encoder_num_heads: int,
        num_groups: int,
        grad_ckpt: bool = False,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.encoder = nn.Sequential(
            spadop.PatchEmbed(
                in_channels, dim, 16, patch_embed_kernel_size, True,
            ),
            spadop.SwinLayer(
                dim, encoder_depth, encoder_num_heads, 4, last_norm=True, grad_ckpt=grad_ckpt,
            ),
            ChannelLast(),
        )
        self.decoder = nn.Sequential(
            ChannelFirst(),
            *[
                nn.Sequential(
                    nn.Identity() if i == 0 else
                    spadop.TransposedConv3d(dim >> i - 1, dim >> i, 2, 2),
                    ResNetBasicBlock(dim >> i, num_groups >> i),
                )
                for i in range(3)
            ],
            spadop.TransposedConv3d(dim >> 2, in_channels * 2, 4, 4),
        )

    def encode(self, x: spadop.SpatialTensor) -> spadop.SpatialTensor:
        return self.encoder(x)

    def decode(self, z_q: spadop.SpatialTensor) -> spadop.SpatialTensor:
        return self.decoder(z_q)
