# sac stands for spatially adaptive convolution

from collections.abc import Iterable, Sequence
from typing import Literal

import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from mylib.types import param3_t, tuple3_t
from monai.utils import InterpolateMode, ensure_tuple_rep

from pumt.transforms import RGB_TO_GRAY_WEIGHT

class SpatialTensor(torch.Tensor):
    # gradient checkpointing will not work for this class
    # https://github.com/pytorch/pytorch/issues/105644

    @staticmethod
    def __new__(cls, x, aniso_d: int, num_downsamples: int = 0, *args, **kwargs):
        return torch.as_tensor(x, *args, **kwargs).as_subclass(SpatialTensor)

    def __init__(self, _x, aniso_d: int, num_downsamples: int = 0, *_args, **_kwargs):
        super().__init__()
        self.aniso_d = aniso_d
        self.num_downsamples = num_downsamples

    def __repr__(self, *args, **kwargs):
        aniso_d = getattr(self, 'aniso_d', 'missing')
        num_downsamples = getattr(self, 'num_downsamples', 'missing')
        return f'shape={self.shape}, aniso_d={aniso_d}, num_downsamples={num_downsamples}\n{super().__repr__()}'

    @property
    def num_pending_hw_downsamples(self):
        return max(self.aniso_d - self.num_downsamples, 0)

    @property
    def can_downsample_d(self) -> bool:
        return self.num_pending_hw_downsamples == 0

    @property
    def num_remained_d_upsamples(self) -> int:
        return max(self.num_downsamples - self.aniso_d, 0)

    @property
    def can_upsample_d(self) -> bool:
        return self.num_remained_d_upsamples > 0

    @classmethod
    def find_meta_ref_iter(cls, iterable: Iterable):
        for x in iterable:
            if (ret := cls.find_meta_ref(x)) is not None:
                return ret
        return None

    @classmethod
    def find_meta_ref(cls, obj):
        match obj:
            case SpatialTensor():
                return obj
            case tuple() | list():
                return cls.find_meta_ref_iter(obj)
            case dict():
                return cls.find_meta_ref_iter(obj.values())
            case _:
                return None

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        ret = super().__torch_function__(func, types, args, kwargs)
        if isinstance(ret, Sequence):
            unpack = False
        else:
            unpack = True
            ret = [ret]
        if any(isinstance(x, SpatialTensor) for x in ret) and (
            (meta_ref := cls.find_meta_ref(args)) is not None
            or (meta_ref := cls.find_meta_ref(kwargs)) is not None
        ):
            meta_ref: SpatialTensor
            for x in ret:
                if isinstance(x, SpatialTensor):
                    x.aniso_d = meta_ref.aniso_d
                    x.num_downsamples = meta_ref.num_downsamples
        if unpack:
            ret = ret[0]
        return ret

    def as_tensor(self):
        return self.as_subclass(torch.Tensor)

class InflatableConv3d(nn.Conv3d):
    def __init__(self, *args, adaptive: bool = True, d_inflation: Literal['average', 'center'] = 'average', **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive = adaptive
        if adaptive:
            assert self.stride[0] == self.stride[1] == self.stride[2], 'only support isotropic stride'
            assert self.stride[0] & self.stride[0] - 1 == 0, 'only support power of 2'
            self.num_downsamples = self.stride[0].bit_length() - 1
            assert self.padding_mode == 'zeros'
        self.inflation = d_inflation
        assert d_inflation in ['average', 'center']

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        weight_key = f'{prefix}weight'
        if (weight := state_dict.get(weight_key)) is not None and weight.ndim + 1 == self.weight.ndim:
            d = self.kernel_size[0]
            match self.inflation:
                case 'average':
                    weight = einops.repeat(weight / d, 'co ci ... -> co ci d ...', d=d)
                case 'center':
                    new_weight = weight.new_zeros(*weight.shape[:2], d, *weight.shape[2:])
                    if d & 1:
                        new_weight[:, :, d >> 1] = weight
                    else:
                        new_weight[:, :, [d - 1 >> 1, d >> 1]] = weight[:, :, None] / 2
                    weight = new_weight
                case _:
                    raise ValueError
            state_dict[weight_key] = weight
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        if self.adaptive:
            x: SpatialTensor
            stride = list(self.stride)
            padding = list(self.padding)
            stride[0] = max(self.stride[0] >> x.num_pending_hw_downsamples, 1)
            if stride[0] == self.stride[0] and x.num_pending_hw_downsamples == 0:
                weight = self.weight
            else:
                padding[0] = 0
                if stride[0] == 1:
                    assert self.stride[0] == self.kernel_size[0] or self.kernel_size[0] == 3, "don't do this or teach me how to do this /kl"
                    weight = self.weight.sum(dim=2, keepdim=True)
                else:
                    assert self.stride[0] == self.kernel_size[0], "don't do this or teach me how to do this /kl"
                    weight = einops.reduce(
                        self.weight,
                        'co ci (dr dc) ... -> co ci dr ...',
                        'sum',
                        dr=stride[0],
                    )
            x: SpatialTensor = nnf.conv3d(x, weight, self.bias, stride, padding, self.dilation, self.groups)
            x.num_downsamples += self.num_downsamples
            return x
        else:
            return super().forward(x)

class InflatableInputConv3d(InflatableConv3d):
    def __init__(self, *args, force: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.force = force

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.get(weight_key := f'{prefix}weight')) is not None \
        and (weight.shape[1] != (ci := self.weight.shape[1]) or self.force):
            state_dict[weight_key] = einops.repeat(
                einops.reduce(weight, 'co ci ... -> co ...', 'sum') / ci,
                'co ... -> co ci ...', ci=ci,
            )
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class InflatableOutputConv3d(InflatableConv3d):
    def __init__(self, *args, force: bool = False, c_inflation: Literal['RGB_L', 'average'] = 'RGB_L', **kwargs):
        super().__init__(*args, **kwargs)
        self.force = force
        assert c_inflation in ['RGB_L', 'average']
        self.c_inflation = c_inflation

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.get(weight_key := f'{prefix}weight')) is not None \
        and (weight.shape[0] != (co := self.weight.shape[0]) or self.force):
            match self.c_inflation:
                case 'average':
                    weight = einops.repeat(
                        einops.reduce(weight, 'co ci ... -> ci ...', 'sum') / co,
                        'ci ... -> co ci ...', co=co,
                    )
                case 'RGB_L':
                    assert weight.shape[0] == 3 and self.out_channels == 1
                    # RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
                    weight = einops.einsum(
                        weight.new_tensor(RGB_TO_GRAY_WEIGHT), weight,
                        'c, c ... -> ...'
                    )[None]

            state_dict[weight_key] = weight

        if (bias := state_dict.get(bias_key := f'{prefix}bias')) is not None \
        and (bias.shape[0] != (co := self.bias.shape[0]) or self.force):
            match self.c_inflation:
                case 'average':
                    bias = einops.repeat(
                        einops.reduce(bias, 'co -> ', 'sum') / co,
                        ' -> co', co=co,
                    )
                case 'RGB_L':
                    bias = torch.dot(bias.new_tensor(RGB_TO_GRAY_WEIGHT), bias)[None]
            state_dict[bias_key] = bias
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class AdaptiveInterpolationDownsample(nn.Module):
    def __init__(self, mode: InterpolateMode = InterpolateMode.AREA, antialias: bool = False):
        super().__init__()
        self.mode = mode
        self.antialias = antialias

    def forward(self, x: SpatialTensor):
        x = nnf.interpolate(
            x,
            scale_factor=(0.5 if x.can_downsample_d else 1, 0.5, 0.5),
            mode=self.mode,
            antialias=self.antialias,
        )
        x.num_downsamples += 1
        return x

class AdaptiveConvDownsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: param3_t[int] = 2,
        bias: bool = True,
        conv_t: type[InflatableConv3d] = InflatableConv3d,
        d_inflation: Literal['average', 'center'] = 'average',
    ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.conv = conv_t(
            in_channels, out_channels, kernel_size,
            stride=2,
            padding=tuple(k - 1 >> 1 for k in ensure_tuple_rep(kernel_size, 3)),
            bias=bias,
            d_inflation=d_inflation,
        )

    def forward(self, x: SpatialTensor):
        return self.conv(x)

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.pop(f'{prefix}weight', None)) is not None:
            state_dict[f'{prefix}conv.weight'] = weight
            if (bias := state_dict.pop(f'{prefix}bias', None)) is not None:
                state_dict[f'{prefix}conv.bias'] = bias
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class AdaptiveInterpolationUpsample(nn.Module):
    def __init__(self, mode: InterpolateMode = InterpolateMode.TRILINEAR):
        super().__init__()
        self.mode = mode

    def upsample(self, x: SpatialTensor) -> SpatialTensor:
        return nnf.interpolate(x, scale_factor=(2. if x.can_upsample_d else 1., 2., 2.), mode=self.mode)

    def forward(self, x: SpatialTensor):
        x = self.upsample(x)
        x.num_downsamples -= 1
        return x

class AdaptiveInterpolationUpsampleWithPostConv(AdaptiveInterpolationUpsample):
    # following VQGAN's implementation
    def __init__(self, in_channels: int, out_channels: int | None = None, mode: InterpolateMode = InterpolateMode.NEAREST_EXACT):
        super().__init__(mode)
        if out_channels is None:
            out_channels = in_channels
        self.conv = InflatableConv3d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: SpatialTensor):
        x = super().forward(x)
        x = self.conv(x)
        return x

class InflatableTransposedConv3d(nn.ConvTranspose3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for stride in self.stride:
            assert stride & stride - 1 == 0, 'only support power of 2'
        assert self.stride[1] == self.stride[2], 'only support stride_h == stride_w'
        self.num_upsamples = self.stride[1].bit_length() - 1
        assert self.kernel_size == self.stride
        assert self.padding == (0, 0, 0)
        assert self.output_padding == (0, 0, 0)
        assert self.padding_mode == 'zeros'

    def forward(self, x: SpatialTensor, output_size=None):
        assert output_size is None
        stride = list(self.stride)
        stride[0] = min(1 << x.num_remained_d_upsamples, self.stride[0])
        if stride[0] == self.stride[0]:
            weight = self.weight
        else:
            weight = einops.reduce(
                self.weight,
                'co ci (dr dc) ... -> co ci dr ...',
                'sum',
                dr=stride[0],
            )
        x: SpatialTensor = nnf.conv_transpose3d(x, weight, self.bias, stride, self.padding, self.output_padding, self.groups, self.dilation)
        x.num_downsamples -= self.num_upsamples
        return x

class AdaptiveTransposedConvUpsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.transposed_conv = InflatableTransposedConv3d(in_channels, out_channels, kernel_size=stride, stride=stride)
        self.conv = nn.Sequential(
            InflatableConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: SpatialTensor):
        x = self.transposed_conv(x)
        return self.conv(x)

def resample(x: torch.Tensor, shape: tuple3_t[int]):
    downsample_shape = tuple(np.minimum(x.shape[2:], shape))
    if downsample_shape != x.shape[2:]:
        x = nnf.interpolate(x, downsample_shape, mode='area')
    if shape != x.shape[2:]:
        x = nnf.interpolate(x, shape, mode='trilinear')
    return x
