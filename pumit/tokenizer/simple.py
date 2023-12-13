from collections.abc import Sequence

from torch import nn

from luolib.types import call_partial, partial_t, tuple3_t
from luolib.models import spadop
from luolib.models.layers import LayerNormNd

from .base import VQVisualTokenizer

class SimpleVQVT(VQVisualTokenizer):
    def __init__(
        self,
        in_channels: int,
        start_stride: int,
        downsample_layer_channels: Sequence[int],
        upsample_layer_channels: Sequence[int],
        encoder_act: partial_t[nn.Module] = nn.GELU,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert start_stride > 1
        self.encoder = nn.Sequential()
        self._stride = 1
        for i in range(len(downsample_layer_channels)):
            stride = start_stride if i == 0 else 2
            self._stride *= stride
            self.encoder.extend([
                spadop.Conv3d(
                    in_channels if i == 0 else downsample_layer_channels[i - 1],
                    downsample_layer_channels[i],
                    kernel_size=stride,
                    stride=stride,
                ),
                LayerNormNd(downsample_layer_channels[i]),
                call_partial(encoder_act),
            ])
        self.encoder.extend([
            spadop.Conv3d(
                downsample_layer_channels[-1],
                downsample_layer_channels[-1],
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, downsample_layer_channels[-1]),
            nn.LeakyReLU(inplace=True),
            spadop.Conv3d(
                downsample_layer_channels[-1],
                self.quantize.proj.in_features,
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, self.quantize.proj.in_features),
            nn.LeakyReLU(inplace=True),
        ])

        output_stride = start_stride << len(downsample_layer_channels) - 1 >> len(upsample_layer_channels) - 1
        self.decoder = nn.Sequential(
            spadop.Conv3d(self.quantize.embedding_dim, upsample_layer_channels[-1], kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, upsample_layer_channels[-1]),
            nn.LeakyReLU(inplace=True),
            spadop.Conv3d(
                self.quantize.embedding_dim, upsample_layer_channels[-1], kernel_size=3, stride=1, padding=1
            ),
            nn.GroupNorm(8, upsample_layer_channels[-1]),
            nn.LeakyReLU(inplace=True),
            *[
                spadop.AdaptiveTransposedConvUpsample(upsample_layer_channels[i + 1], upsample_layer_channels[i], 2)
                for i in reversed(range(len(upsample_layer_channels) - 1))
            ],
            spadop.TransposedConv3d(upsample_layer_channels[0], in_channels, kernel_size=output_stride, stride=output_stride),
        )

    def encode(self, x: spadop.SpatialTensor) -> spadop.SpatialTensor:
        return self.encoder(x)

    def decode(self, z_q: spadop.SpatialTensor) -> spadop.SpatialTensor:
        return self.decoder(z_q)
