from collections.abc import Sequence

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
        encoder_layer_channels: Sequence[int],
        encoder_layer_depths: Sequence[int],
        encoder_layer_num_heads: Sequence[int],
        encoder_output_channels: int,
        decoder_layer_channels: Sequence[int],
        decoder_layer_depths: Sequence[int],
        decoder_layer_num_heads: Sequence[int],
        grad_ckpt: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        encoder_num_layers = len(encoder_layer_channels)
        decoder_num_layers = len(encoder_layer_channels)
        assert encoder_num_layers == decoder_num_layers == 3
        self.encoder = nn.Sequential(
            spadop.InputConv3D(in_channels, encoder_layer_channels[0], 4, 4),
        )
        for i in range(encoder_num_layers):
            self.encoder.append(
                spadop.SwinLayer(
                    encoder_layer_channels[i], encoder_layer_depths[i], encoder_layer_num_heads[i], 4,
                    last_norm=True, grad_ckpt=grad_ckpt,
                )
            )
            if i + 1 < len(encoder_layer_channels):
                self.encoder.append(
                    spadop.Conv3d(encoder_layer_channels[i], encoder_layer_channels[i + 1], 2, 2),
                )
        self.encoder.extend([
            ChannelLast(),
            nn.Linear(encoder_layer_channels[-1], encoder_output_channels),
            nn.LayerNorm(encoder_output_channels),
            nn.GELU(),
        ])

        self.decoder = nn.Sequential(
            nn.Linear(self.quantize.embedding_dim, decoder_layer_channels[-1]),
            nn.LayerNorm(encoder_layer_channels[-1]),
            nn.GELU(),
            ChannelFirst(),
        )
        for i in reversed(range(decoder_num_layers)):
            self.decoder.append(
                spadop.SwinLayer(
                    decoder_layer_channels[i], decoder_layer_depths[i], decoder_layer_num_heads[i], 4,
                    last_norm=True, grad_ckpt=grad_ckpt,
                )
            )
            if i > 0:
                self.decoder.append(
                    spadop.TransposedConv3d(decoder_layer_channels[i], decoder_layer_channels[i - 1], 2, 2),
                )
        self.decoder.append(
            # there can be something like OutputTransposedConv3d, but unnecessary
            spadop.TransposedConv3d(
                decoder_layer_channels[0], in_channels, 4, 4,
            )
        )

    def encode(self, x: spadop.SpatialTensor) -> spadop.SpatialTensor:
        return self.encoder(x)

    def decode(self, z_q: spadop.SpatialTensor) -> spadop.SpatialTensor:
        return self.decoder(z_q)
