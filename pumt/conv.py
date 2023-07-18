import abc
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import einops
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.types import param3_t
from monai.utils import InterpolateMode, ensure_tuple_rep

class SpatialTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x, aniso_d: int, *args, **kwargs):
        return torch.as_tensor(x, *args, **kwargs).as_subclass(SpatialTensor)

    def __init__(self, _x, aniso_d: int, *_args, **_kwargs):
        super().__init__()
        self.aniso_d = aniso_d
        self.num_downsamples = 0

    def __repr__(self, *args, **kwargs):
        aniso_d = getattr(self, 'aniso_d', 'missing')
        num_downsamples = getattr(self, 'num_downsamples', 'missing')
        return f'shape={self.shape}, aniso_d={aniso_d}, num_downsamples={num_downsamples}\n{super().__repr__()}'

    @property
    def downsample_ready(self) -> bool:
        return self.aniso_d <= self.num_downsamples

    @property
    def upsample_ready(self) -> bool:
        return self.aniso_d < self.num_downsamples

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
        if isinstance(ret, SpatialTensor) and (
            (meta_ref := cls.find_meta_ref(args)) is not None
            or (meta_ref := cls.find_meta_ref(kwargs)) is not None
        ):
            ret.aniso_d = meta_ref.aniso_d
            ret.num_downsamples = meta_ref.num_downsamples
        return ret

class InflatableConv3d(nn.Conv3d):
    def __init__(self, *args, d_inflation: Literal['average', 'center'] = 'average', **kwargs):
        super().__init__(*args, **kwargs)
        assert self.stride[0] in (1, 2)
        assert self.padding_mode == 'zeros'
        assert d_inflation in ['average', 'center']
        self.inflation = d_inflation

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

    def forward(self, x: SpatialTensor):
        stride = list(self.stride)
        padding = list(self.padding)
        if x.aniso_d > x.num_downsamples:
            stride[0] = 1
            padding[0] = 0
            weight = self.weight.sum(dim=2, keepdim=True)
        else:
            weight = self.weight
        return nnf.conv3d(x, weight, self.bias, stride, padding, self.dilation, self.groups)

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

# RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
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
                    weight = einops.einsum(
                        weight.new_tensor([0.299, 0.587, 0.114]), weight,
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
                    bias = torch.dot(bias.new_tensor([0.299, 0.587, 0.114]), bias)[None]
            state_dict[bias_key] = bias
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class AdaptiveDownsample(nn.Module, abc.ABC):
    @abc.abstractmethod
    def downsample(self, x: SpatialTensor) -> SpatialTensor:
        pass

    def forward(self, x: SpatialTensor):
        x = self.downsample(x)
        x.num_downsamples += 1
        return x

class AdaptiveInterpolationDownsample(AdaptiveDownsample):
    def __init__(self, mode: InterpolateMode = InterpolateMode.AREA, antialias: bool = False):
        super().__init__()
        self.mode = mode
        self.antialias = antialias

    def downsample(self, x: SpatialTensor):
        return nnf.interpolate(
            x,
            scale_factor=(0.5 if x.downsample_ready else 1, 0.5, 0.5),
            mode=self.mode,
            antialias=self.antialias,
        )

class AdaptiveConvDownsample(AdaptiveDownsample):
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

    def downsample(self, x: SpatialTensor):
        return self.conv(x)

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.pop(f'{prefix}weight', None)) is not None:
            state_dict[f'{prefix}conv.weight'] = weight
            if (bias := state_dict.pop(f'{prefix}bias', None)) is not None:
                state_dict[f'{prefix}conv.bias'] = bias
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

class AdaptiveUpsample(nn.Module):
    def __init__(self, mode: InterpolateMode = InterpolateMode.TRILINEAR):
        super().__init__()
        self.mode = mode

    def upsample(self, x: SpatialTensor) -> SpatialTensor:
        return nnf.interpolate(x, scale_factor=(2. if x.upsample_ready else 1., 2., 2.), mode=self.mode)

    def forward(self, x: SpatialTensor):
        x = self.upsample(x)
        x.num_downsamples -= 1
        return x

class AdaptiveUpsampleWithPostConv(AdaptiveUpsample):
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
