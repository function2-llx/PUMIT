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

import warnings
from collections.abc import Sequence
from pydoc import locate
from typing import Any

import torch
from torch import nn

from monai.networks.blocks import BaseEncoder, UpSample
from monai.networks.layers.factories import Conv
from monai.networks.layers.utils import get_act_layer
from monai.networks.nets import EfficientNetEncoder
from monai.networks.nets.basic_unet import UpCat
from monai.utils import InterpolateMode, optional_import

__all__ = ["FlexibleUNet", "FlexUNet", "FLEXUNET_BACKBONE", "FlexUNetEncoderRegister"]


class FlexUNetEncoderRegister:
    """
    A register to regist backbones for the flexible unet. All backbones can be found in
    register_dict. Please notice each output of backbone must be 2x downsample in spatial
    dimension of last output. For example, if given a 512x256 2D image and a backbone with
    4 outputs. Then spatial size of each encoder output should be 256x128, 128x64, 64x32
    and 32x16.
    """

    def __init__(self):
        self.register_dict = {}

    def register_class(self, name: type[Any] | str):
        """
        Register a given class to the encoder dict. Please notice that input class must be a
        subclass of BaseEncoder.
        """
        if isinstance(name, str):
            tmp_name, has_built_in = optional_import("monai.networks.nets", name=f"{name}")  # search built-in
            if not has_built_in:
                tmp_name = locate(f"{name}")  # search dotted path
            name = tmp_name
            if not isinstance(name, type):
                raise ValueError(f"Cannot find {name} class.")

        if not issubclass(name, BaseEncoder):
            warnings.warn(
                f"{name} would better be derived from monai.networks.blocks.BaseEncoder "
                "or implement all interfaces specified by it."
            )

        name_string_list = name.get_encoder_names()
        feature_number_list = name.num_outputs()
        feature_channel_list = name.num_channels_per_output()
        parameter_list = name.get_encoder_parameters()

        assert len(name_string_list) == len(feature_number_list) == len(feature_channel_list) == len(parameter_list)
        for cnt, name_string in enumerate(name_string_list):
            cur_dict = {
                "type": name,
                "feature_number": feature_number_list[cnt],
                "feature_channel": feature_channel_list[cnt],
                "parameter": parameter_list[cnt],
            }
            self.register_dict[name_string] = cur_dict


FLEXUNET_BACKBONE = FlexUNetEncoderRegister()
FLEXUNET_BACKBONE.register_class(EfficientNetEncoder)


class UNetDecoder(nn.Module):
    """
    UNet Decoder.
    This class refers to `segmentation_models.pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        encoder_channels: number of output channels for all feature maps in encoder.
            `len(encoder_channels)` should be no less than 2.
        decoder_channels: number of output channels for all feature maps in decoder.
            `len(decoder_channels)` should equal to `len(encoder_channels) - 1`.
        act: activation type and arguments.
        norm: feature normalization type and arguments.
        dropout: dropout ratio.
        bias: whether to have a bias term in convolution blocks in this decoder.
        upsample: upsampling mode, available options are
            ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        pre_conv: a conv block applied before upsampling.
            Only used in the "nontrainable" or "pixelshuffle" mode.
        interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
            Only used in the "nontrainable" mode.
        align_corners: set the align_corners parameter for upsample. Defaults to True.
            Only used in the "nontrainable" mode.
        is_pad: whether to pad upsampling features to fit the encoder spatial dims.

    """

    def __init__(
        self,
        spatial_dims: int,
        encoder_channels: Sequence[int],
        decoder_channels: Sequence[int],
        act: str | tuple,
        norm: str | tuple,
        dropout: float | tuple,
        bias: bool,
        upsample: str,
        pre_conv: str | None,
        interp_mode: str,
        align_corners: bool | None,
        is_pad: bool,
    ):
        super().__init__()
        if len(encoder_channels) < 2:
            raise ValueError("the length of `encoder_channels` should be no less than 2.")
        if len(decoder_channels) != len(encoder_channels) - 1:
            raise ValueError("`len(decoder_channels)` should equal to `len(encoder_channels) - 1`.")

        in_channels = [encoder_channels[-1]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:-1][::-1]) + [0]
        halves = [True] * (len(skip_channels) - 1)
        halves.append(False)
        blocks = []
        for in_chn, skip_chn, out_chn, halve in zip(in_channels, skip_channels, decoder_channels, halves):
            blocks.append(
                UpCat(
                    spatial_dims=spatial_dims,
                    in_chns=in_chn,
                    cat_chns=skip_chn,
                    out_chns=out_chn,
                    act=act,
                    norm=norm,
                    dropout=dropout,
                    bias=bias,
                    upsample=upsample,
                    pre_conv=pre_conv,
                    interp_mode=interp_mode,
                    align_corners=align_corners,
                    halves=halve,
                    is_pad=is_pad,
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, features: list[torch.Tensor], skip_connect: int = 4):
        skips = features[:-1][::-1]
        features = features[1:][::-1]

        x = features[0]
        for i, block in enumerate(self.blocks):
            if i < skip_connect:
                skip = skips[i]
            else:
                skip = None
            x = block(x, skip)

        return x


class SegmentationHead(nn.Sequential):
    """
    Segmentation head.
    This class refers to `segmentation_models.pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_.

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels for the block.
        out_channels: number of output channels for the block.
        kernel_size: kernel size for the conv layer.
        act: activation type and arguments.
        scale_factor: multiplier for spatial size. Has to match input size if it is a tuple.

    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        act: tuple | str | None = None,
        scale_factor: float = 1.0,
    ):
        conv_layer = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        up_layer: nn.Module = nn.Identity()
        if scale_factor > 1.0:
            up_layer = UpSample(
                spatial_dims=spatial_dims,
                scale_factor=scale_factor,
                mode="nontrainable",
                pre_conv=None,
                interp_mode=InterpolateMode.LINEAR,
            )
        if act is not None:
            act_layer = get_act_layer(act)
        else:
            act_layer = nn.Identity()
        super().__init__(conv_layer, up_layer, act_layer)


class FlexibleUNet(nn.Module):
    """
    A flexible implementation of UNet-like encoder-decoder architecture.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        backbone: str,
        pretrained: bool = False,
        decoder_channels: tuple = (256, 128, 64, 32, 16),
        spatial_dims: int = 2,
        norm: str | tuple = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: str | tuple = ("relu", {"inplace": True}),
        dropout: float | tuple = 0.0,
        decoder_bias: bool = False,
        upsample: str = "nontrainable",
        pre_conv: str = "default",
        interp_mode: str = "nearest",
        is_pad: bool = True,
    ) -> None:
        """
        A flexible implement of UNet, in which the backbone/encoder can be replaced with
        any efficient network. Currently the input must have a 2 or 3 spatial dimension
        and the spatial size of each dimension must be a multiple of 32 if is_pad parameter
        is False.
        Please notice each output of backbone must be 2x downsample in spatial dimension
        of last output. For example, if given a 512x256 2D image and a backbone with 4 outputs.
        Spatial size of each encoder output should be 256x128, 128x64, 64x32 and 32x16.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            backbone: name of backbones to initialize, only support efficientnet right now,
                can be from [efficientnet-b0,..., efficientnet-b8, efficientnet-l2].
            pretrained: whether to initialize pretrained ImageNet weights, only available
                for spatial_dims=2 and batch norm is used, default to False.
            decoder_channels: number of output channels for all feature maps in decoder.
                `len(decoder_channels)` should equal to `len(encoder_channels) - 1`,default
                to (256, 128, 64, 32, 16).
            spatial_dims: number of spatial dimensions, default to 2.
            norm: normalization type and arguments, default to ("batch", {"eps": 1e-3,
                "momentum": 0.1}).
            act: activation type and arguments, default to ("relu", {"inplace": True}).
            dropout: dropout ratio, default to 0.0.
            decoder_bias: whether to have a bias term in decoder's convolution blocks.
            upsample: upsampling mode, available options are``"deconv"``, ``"pixelshuffle"``,
                ``"nontrainable"``.
            pre_conv:a conv block applied before upsampling. Only used in the "nontrainable" or
                "pixelshuffle" mode, default to `default`.
            interp_mode: {``"nearest"``, ``"linear"``, ``"bilinear"``, ``"bicubic"``, ``"trilinear"``}
                Only used in the "nontrainable" mode.
            is_pad: whether to pad upsampling features to fit features from encoder. Default to True.
                If this parameter is set to "True", the spatial dim of network input can be arbitrary
                size, which is not supported by TensorRT. Otherwise, it must be a multiple of 32.
        """
        super().__init__()

        if backbone not in FLEXUNET_BACKBONE.register_dict:
            raise ValueError(
                f"invalid model_name {backbone} found, must be one of {FLEXUNET_BACKBONE.register_dict.keys()}."
            )

        if spatial_dims not in (2, 3):
            raise ValueError("spatial_dims can only be 2 or 3.")

        encoder = FLEXUNET_BACKBONE.register_dict[backbone]
        self.backbone = backbone
        self.spatial_dims = spatial_dims
        encoder_parameters = encoder["parameter"]
        if not (
            ("spatial_dims" in encoder_parameters)
            and ("in_channels" in encoder_parameters)
            and ("pretrained" in encoder_parameters)
        ):
            raise ValueError("The backbone init method must have spatial_dims, in_channels and pretrained parameters.")
        encoder_feature_num = encoder["feature_number"]
        if encoder_feature_num > 5:
            raise ValueError("Flexible unet can only accept no more than 5 encoder feature maps.")

        decoder_channels = decoder_channels[:encoder_feature_num]
        self.skip_connect = encoder_feature_num - 1
        encoder_parameters.update({"spatial_dims": spatial_dims, "in_channels": in_channels, "pretrained": pretrained})
        encoder_channels = tuple([in_channels] + list(encoder["feature_channel"]))
        encoder_type = encoder["type"]
        self.encoder = encoder_type(**encoder_parameters)

        self.decoder = UNetDecoder(
            spatial_dims=spatial_dims,
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=decoder_bias,
            upsample=upsample,
            interp_mode=interp_mode,
            pre_conv=pre_conv,
            align_corners=None,
            is_pad=is_pad,
        )
        self.segmentation_head = SegmentationHead(
            spatial_dims=spatial_dims,
            in_channels=decoder_channels[-1],
            out_channels=out_channels,
            kernel_size=3,
            act=None,
        )

    def forward(self, inputs: torch.Tensor):
        """
        Do a typical encoder-decoder-header inference.

        Args:
            inputs: input should have spatially N dimensions ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N])``,
                N is defined by `dimensions`.

        Returns:
            A torch Tensor of "raw" predictions in shape ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N])``.

        """
        x = inputs
        enc_out = self.encoder(x)
        decoder_out = self.decoder(enc_out, self.skip_connect)
        x_seg = self.segmentation_head(decoder_out)

        return x_seg


FlexUNet = FlexibleUNet
