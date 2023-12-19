import math
from typing import Hashable, Mapping

import numpy as np

from monai import transforms as mt
from monai.data import MetaTensor
from monai.utils import ImageMetaKey
from pumit.transforms import ensure_rgb

class UpdateSpacingD(mt.Transform):
    def __call__(self, data: Mapping[Hashable, ...]):
        data = dict(data)
        img: MetaTensor = data['img']
        spacing = data['spacing']
        img.affine = np.diag([*spacing, 1])
        return data

class RandAffineCropD(mt.Randomizable, mt.LazyTransform):
    def __init__(self, rotate_p: float, isotropic_th: float, lazy: bool = False):
        self.rotate_p = rotate_p
        self.isotropic_th = isotropic_th
        mt.LazyTransform.__init__(self, lazy)

    def __call__(self, data: Mapping[Hashable, ...], lazy: bool | None = None):
        data = dict(data)
        trans_info = data['_trans']
        img: MetaTensor = data['img']
        # following nnU-Net
        rotate_x_range = 0. if (img.pixdim[0] > self.isotropic_th * min(img.pixdim[1:])) else np.pi / 6
        rotate_range = (np.pi / 2, rotate_x_range, rotate_x_range)
        sample_size = trans_info['size']
        cropper = mt.Compose(
            [
                mt.SpatialPad(sample_size),
                mt.RandSpatialCrop(sample_size, random_center=True, random_size=False),
                mt.RandAffine(self.rotate_p, rotate_range),
                mt.Affine(scale_params=trans_info['scale'], image_only=True),
            ],
            lazy=self.lazy if lazy is None else lazy,
            apply_pending=False,
        )
        data['img'] = cropper(img)
        return data

class CenterScaleCropD(mt.LazyTransform):
    def __init__(self, tz: int, tx: int, scale_x: float, stride: int, lazy: bool = False):
        self.tz = tz
        self.tx = tx
        self.scale_x = scale_x
        self.stride = stride
        mt.LazyTransform.__init__(self, lazy)

    def __call__(self, data: Mapping[Hashable, ...], lazy: bool | None = None):
        data = dict(data)
        img: MetaTensor = data['img']
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
                mt.SpatialPad(sample_size),
                mt.CenterSpatialCrop(sample_size),
                mt.Affine(scale_params=(1., scale_x, scale_x), image_only=True),
            ],
            lazy=self.lazy if lazy is None else lazy,
            apply_pending=False,
        )
        data['img'] = cropper(img)
        return data

class AsSpatialTensorD(mt.Transform):
    def __call__(self, data: Mapping[Hashable, ...]):
        img: MetaTensor = data['img']
        return ensure_rgb(img.as_tensor()), data['_trans']['aniso_d'], img.meta[ImageMetaKey.FILENAME_OR_OBJ]

class AdaptivePadD(mt.Transform):
    def __call__(self, data: Mapping):
        data = dict(data)
        img: MetaTensor = data['img']
        aniso_d = max(int(img.pixdim[0] / min(img.pixdim[1:])).bit_length() - 1, 0)
        modality = data['modality']
        d = max(64 >> aniso_d, 1)
        trans = mt.Compose(
            [
                mt.SpatialPad((d, -1, -1)),
                mt.CenterSpatialCrop((d, -1, -1)),
                mt.DivisiblePad((-1, 16, 16)),
            ],
            lazy=True,
        )
        img = trans(img)
        return ensure_rgb(img.as_tensor()), aniso_d, modality, img.meta[ImageMetaKey.FILENAME_OR_OBJ]
