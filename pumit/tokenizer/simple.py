from collections.abc import Sequence

from torch import nn

from mylib.models.layers import LayerNormNd
from mylib.types import call_partial, partial_t, tuple3_t

from pumit import sac
from .base import VQTokenizer

class SimpleVQTokenizer(VQTokenizer):
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
                sac.InflatableConv3d(
                    in_channels if i == 0 else downsample_layer_channels[i - 1],
                    downsample_layer_channels[i],
                    kernel_size=stride,
                    stride=stride,
                ),
                LayerNormNd(downsample_layer_channels[i]),
                call_partial(encoder_act),
            ])
        self.encoder.extend([
            sac.InflatableConv3d(
                downsample_layer_channels[-1],
                downsample_layer_channels[-1],
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(8, downsample_layer_channels[-1]),
            nn.LeakyReLU(inplace=True),
            sac.InflatableConv3d(
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
            sac.InflatableConv3d(self.quantize.embedding_dim, upsample_layer_channels[-1], kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, upsample_layer_channels[-1]),
            nn.LeakyReLU(inplace=True),
            sac.InflatableConv3d(
                self.quantize.embedding_dim, upsample_layer_channels[-1], kernel_size=3, stride=1, padding=1
            ),
            nn.GroupNorm(8, upsample_layer_channels[-1]),
            nn.LeakyReLU(inplace=True),
            *[
                sac.AdaptiveTransposedConvUpsample(upsample_layer_channels[i + 1], upsample_layer_channels[i], 2)
                for i in reversed(range(len(upsample_layer_channels) - 1))
            ],
            sac.InflatableTransposedConv3d(upsample_layer_channels[0], in_channels, kernel_size=output_stride, stride=output_stride),
        )

    @property
    def stride(self) -> tuple3_t[int]:
        return (self._stride, ) * 3

    def encode(self, x: sac.SpatialTensor) -> sac.SpatialTensor:
        return self.encoder(x)

    def decode(self, z_q: sac.SpatialTensor) -> sac.SpatialTensor:
        return self.decoder(z_q)
