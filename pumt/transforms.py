import math
from collections.abc import Hashable, Mapping

import numpy as np

from luolib.types import RangeTuple
from luolib.utils import DataKey
from monai import transforms as mt
from monai.data import MetaTensor

class RandAffineCropD(mt.Randomizable, mt.LazyTransform):
    def __init__(
        self,
        tz: int,
        tx: RangeTuple,
        rotate_p: float,
        scale_x_p: float,
        scale_x: RangeTuple,
        scale_z_p: float,
        scale_z: RangeTuple,
        stride: int,
        lazy: bool = False,
    ):
        self.tz = tz
        self.tx = tx
        self.rotate_p = rotate_p
        self.scale_z_p = scale_z_p
        self.scale_z = scale_z
        self.scale_x_p = scale_x_p
        self.scale_x = scale_x
        self.stride = stride
        mt.LazyTransform.__init__(self, lazy)

    def __call__(self, data: Mapping, lazy: bool | None = None):
        data = dict(data)
        img: MetaTensor = data[DataKey.IMG]
        origin_size_x = min(img.shape[2:])
        spacing_z = img.pixdim[0]
        spacing_x = min(img.pixdim[1:])
        aniso_z = spacing_z > 3 * spacing_x
        # di: how much to downsample along axis i
        dx = self.stride
        # ti: n. tokens along axis i
        tx_max = min(self.tx.max, math.ceil(origin_size_x / dx))
        # don't use smaller patch for 2D images
        tx_min = self.tx.min if img.shape[1] > 1 and tx_max >= self.tx.min else tx_max
        tx = self.R.randint(tx_min, tx_max + 1)
        size_x = tx * dx
        if self.R.uniform() < self.scale_x_p:
            scale_x = self.R.uniform(
                self.scale_x.min * origin_size_x / size_x,
                min(origin_size_x / size_x, self.scale_x.max),
            )
        else:
            scale_x = 1.
        if not aniso_z and self.R.uniform() < self.scale_z_p:
            scale_z = self.R.uniform(*self.scale_z)
        else:
            scale_z = 1.
        # ratio of spacing z / spacing x
        rz = np.clip(spacing_z * scale_z / (spacing_x * scale_x), 1, dx)
        dz = dx >> (int(rz).bit_length() - 1)
        tz = min(self.tz, tx, math.ceil(img.shape[1] / (scale_z * dz)))
        sample_size = (tz * dz, size_x, size_x)
        # following nnU-Net
        rotate_x_range = 0. if aniso_z else np.pi / 6
        rotate_range = (np.pi / 2, rotate_x_range, rotate_x_range)
        cropper = mt.Compose(
            [
                mt.SpatialPad(sample_size, value=data['min']),
                mt.RandSpatialCrop(sample_size, random_center=True, random_size=False),
                mt.RandAffine(1., rotate_range, self.rotate_p),
                mt.Affine(scale_params=(scale_z, scale_x, scale_x), image_only=True),
            ],
            lazy=self.lazy if lazy is None else lazy,
            apply_pending=False,
        )
        data[DataKey.IMG] = cropper(img)
        return data

class CenterScaleCropD(mt.LazyTransform):
    def __init__(
        self,
        tz: int,
        tx: int,
        scale_x: float,
        stride: int,
        lazy: bool = False,
    ):
        self.tz = tz
        self.tx = tx
        self.scale_x = scale_x
        self.stride = stride
        mt.LazyTransform.__init__(self, lazy)

    def __call__(self, data: Mapping, lazy: bool | None = None):
        data = dict(data)
        img: MetaTensor = data[DataKey.IMG]
        origin_size_x = min(img.shape[2:])
        dx = self.stride
        tx = min(self.tx, math.ceil(origin_size_x / dx))
        size_x = dx * tx
        scale_x = min(self.scale_x, origin_size_x / size_x)
        rz = np.clip(img.pixdim[0] / (min(img.pixdim[1:]) * scale_x), 1, dx)
        dz = dx >> int(rz).bit_length() - 1
        tz = min(self.tz, tx, math.ceil(img.shape[1] / dz))
        sample_size = (tz * dz, size_x, size_x)
        cropper = mt.Compose(
            [
                mt.SpatialPad(sample_size, value=data['min']),
                mt.CenterSpatialCrop(sample_size),
                mt.Affine(scale_params=(1., scale_x, scale_x), image_only=True),
            ],
            lazy=self.lazy if lazy is None else lazy,
            apply_pending=False,
        )
        data[DataKey.IMG] = cropper(img)
        return data

class NormalizeIntensityD(mt.Transform):
    def __init__(self, b_min: float, b_max: float):
        self.b_min = b_min
        self.b_max = b_max

    def __call__(self, data: Mapping):
        data = dict(data)
        img: MetaTensor = data[DataKey.IMG]
        modality: str = data['modality']
        if modality.startswith('RGB') or modality.startswith('gray'):
            normalizer = mt.ScaleIntensityRange(0., 255., self.b_min, self.b_max)
        else:
            normalizer = mt.ScaleIntensityRange(data['p0.5'], data['p99.5'], self.b_min, self.b_max, clip=True)
        data[DataKey.IMG] = normalizer(img)
        return data

class PrepareInputD(mt.Transform):
    def __init__(self, img_key: Hashable = DataKey.IMG, spacing_key: Hashable = DataKey.SPACING):
        self.img_key = img_key
        self.spacing_key = spacing_key

    def __call__(self, data: Mapping):
        img: MetaTensor = data[self.img_key]
        spacing = img.pixdim
        # make x- and y-axis always be downsampled
        img_tensor = img.as_tensor
        spacing[0] = max(spacing[0], min(spacing[1:]))
        return {
            self.img_key: img_tensor * 2 - 1,
            self.spacing_key: spacing,
        }
