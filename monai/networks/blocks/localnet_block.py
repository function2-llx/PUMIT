# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from monai.networks.blocks import Convolution
from monai.networks.layers import same_padding
from monai.networks.layers.factories import Conv, Norm, Pool


def get_conv_block(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    kernel_size: Sequence[int] | int = 3,
    act: tuple | str | None = "RELU",
    norm: tuple | str | None = "BATCH",
) -> nn.Module:
    padding = same_padding(kernel_size)
    mod: nn.Module = Convolution(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        act=act,
        norm=norm,
        bias=False,
        conv_only=False,
        padding=padding,
    )
    return mod


def get_conv_layer(
    spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Sequence[int] | int = 3
) -> nn.Module:
    padding = same_padding(kernel_size)
    mod: nn.Module = Convolution(
        spatial_dims, in_channels, out_channels, kernel_size=kernel_size, bias=False, conv_only=True, padding=padding
    )
    return mod


def get_deconv_block(spatial_dims: int, in_channels: int, out_channels: int) -> nn.Module:
    mod: nn.Module = Convolution(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=2,
        act="RELU",
        norm="BATCH",
        bias=False,
        is_transposed=True,
        padding=1,
        output_padding=1,
    )
    return mod


class ResidualBlock(nn.Module):
    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Sequence[int] | int
    ) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError(
                f"expecting in_channels == out_channels, " f"got in_channels={in_channels}, out_channels={out_channels}"
            )
        self.conv_block = get_conv_block(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.conv = get_conv_layer(
            spatial_dims=spatial_dims, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.norm = Norm[Norm.BATCH, spatial_dims](out_channels)
        self.relu = nn.ReLU()

    def forward(self, x) -> torch.Tensor:
        out: torch.Tensor = self.relu(self.norm(self.conv(self.conv_block(x))) + x)
        return out


class LocalNetResidualBlock(nn.Module):
    def __init__(self, spatial_dims: int, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if in_channels != out_channels:
            raise ValueError(
                f"expecting in_channels == out_channels, " f"got in_channels={in_channels}, out_channels={out_channels}"
            )
        self.conv_layer = get_conv_layer(spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels)
        self.norm = Norm[Norm.BATCH, spatial_dims](out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, mid) -> torch.Tensor:
        out: torch.Tensor = self.relu(self.norm(self.conv_layer(x)) + mid)
        return out


class LocalNetDownSampleBlock(nn.Module):
    """
    A down-sample module that can be used for LocalNet, based on:
    `Weakly-supervised convolutional neural networks for multimodal image registration
    <https://doi.org/10.1016/j.media.2018.07.002>`_.
    `Label-driven weakly-supervised learning for multimodal deformable image registration
    <https://arxiv.org/abs/1711.01666>`_.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, kernel_size: Sequence[int] | int
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
        Raises:
            NotImplementedError: when ``kernel_size`` is even
        """
        super().__init__()
        self.conv_block = get_conv_block(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.residual_block = ResidualBlock(
            spatial_dims=spatial_dims, in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size
        )
        self.max_pool = Pool[Pool.MAX, spatial_dims](kernel_size=2)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Halves the spatial dimensions.
        A tuple of (x, mid) is returned:

            -  x is the downsample result, in shape (batch, ``out_channels``, insize_1 / 2, insize_2 / 2, [insize_3 / 2]),
            -  mid is the mid-level feature, in shape (batch, ``out_channels``, insize_1, insize_2, [insize_3])

        Args:
            x: Tensor in shape (batch, ``in_channels``, insize_1, insize_2, [insize_3])

        Raises:
            ValueError: when input spatial dimensions are not even.
        """
        for i in x.shape[2:]:
            if i % 2 != 0:
                raise ValueError("expecting x spatial dimensions be even, " f"got x of shape {x.shape}")
        x = self.conv_block(x)
        mid = self.residual_block(x)
        x = self.max_pool(mid)
        return x, mid


class LocalNetUpSampleBlock(nn.Module):
    """
    An up-sample module that can be used for LocalNet, based on:
    `Weakly-supervised convolutional neural networks for multimodal image registration
    <https://doi.org/10.1016/j.media.2018.07.002>`_.
    `Label-driven weakly-supervised learning for multimodal deformable image registration
    <https://arxiv.org/abs/1711.01666>`_.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        mode: str = "nearest",
        align_corners: bool | None = None,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            mode: interpolation mode of the additive upsampling, default to 'nearest'.
            align_corners: whether to align corners for the additive upsampling, default to None.
        Raises:
            ValueError: when ``in_channels != 2 * out_channels``
        """
        super().__init__()
        self.deconv_block = get_deconv_block(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels
        )
        self.conv_block = get_conv_block(spatial_dims=spatial_dims, in_channels=out_channels, out_channels=out_channels)
        self.residual_block = LocalNetResidualBlock(
            spatial_dims=spatial_dims, in_channels=out_channels, out_channels=out_channels
        )
        if in_channels / out_channels != 2:
            raise ValueError(
                f"expecting in_channels == 2 * out_channels, "
                f"got in_channels={in_channels}, out_channels={out_channels}"
            )
        self.out_channels = out_channels
        self.mode = mode
        self.align_corners = align_corners

    def additive_upsampling(self, x, mid) -> torch.Tensor:
        x = F.interpolate(x, mid.shape[2:], mode=self.mode, align_corners=self.align_corners)
        # [(batch, out_channels, ...), (batch, out_channels, ...)]
        x = x.split(split_size=int(self.out_channels), dim=1)
        # (batch, out_channels, ...)
        out: torch.Tensor = torch.sum(torch.stack(x, dim=-1), dim=-1)
        return out

    def forward(self, x, mid) -> torch.Tensor:
        """
        Halves the channel and doubles the spatial dimensions.

        Args:
            x: feature to be up-sampled, in shape (batch, ``in_channels``, insize_1, insize_2, [insize_3])
            mid: mid-level feature saved during down-sampling,
                in shape (batch, ``out_channels``, midsize_1, midsize_2, [midsize_3])

        Raises:
            ValueError: when ``midsize != insize * 2``
        """
        for i, j in zip(x.shape[2:], mid.shape[2:]):
            if j != 2 * i:
                raise ValueError(
                    "expecting mid spatial dimensions be exactly the double of x spatial dimensions, "
                    f"got x of shape {x.shape}, mid of shape {mid.shape}"
                )
        h0 = self.deconv_block(x) + self.additive_upsampling(x, mid)
        r1 = h0 + mid
        r2 = self.conv_block(h0)
        out: torch.Tensor = self.residual_block(r2, r1)
        return out


class LocalNetFeatureExtractorBlock(nn.Module):
    """
    A feature-extraction module that can be used for LocalNet, based on:
    `Weakly-supervised convolutional neural networks for multimodal image registration
    <https://doi.org/10.1016/j.media.2018.07.002>`_.
    `Label-driven weakly-supervised learning for multimodal deformable image registration
    <https://arxiv.org/abs/1711.01666>`_.

    Adapted from:
        DeepReg (https://github.com/DeepRegNet/DeepReg)
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: tuple | str | None = "RELU",
        initializer: str = "kaiming_uniform",
    ) -> None:
        """
        Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        act: activation type and arguments. Defaults to ReLU.
        kernel_initializer: kernel initializer. Defaults to None.
        """
        super().__init__()
        self.conv_block = get_conv_block(
            spatial_dims=spatial_dims, in_channels=in_channels, out_channels=out_channels, act=act, norm=None
        )
        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        for m in self.conv_block.modules():
            if isinstance(m, conv_type):
                if initializer == "kaiming_uniform":
                    nn.init.kaiming_normal_(torch.as_tensor(m.weight))
                elif initializer == "zeros":
                    nn.init.zeros_(torch.as_tensor(m.weight))
                else:
                    raise ValueError(
                        f"initializer {initializer} is not supported, " "currently supporting kaiming_uniform and zeros"
                    )

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: Tensor in shape (batch, ``in_channels``, insize_1, insize_2, [insize_3])
        """
        out: torch.Tensor = self.conv_block(x)
        return out
