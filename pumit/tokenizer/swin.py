from torch import nn

from luolib.models import spadop
from luolib.utils import ChannelFirst, ChannelLast

from .base import VQVisualTokenizer

__all__ = [
    'SwinVQVT',
]

class SwinVQVT(VQVisualTokenizer):
    def __init__(
        self,
        in_channels: int,
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
            spadop.PatchEmbed(in_channels, encoder_dim, 16, True),
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
