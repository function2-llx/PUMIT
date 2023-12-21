from pathlib import Path
from typing import Hashable, Mapping

import torch

from monai import transforms as mt
from monai.data import MetaTensor
from monai.utils import ImageMetaKey
from pumit.transforms import ensure_rgb

class InputTransformD(mt.Transform):
    def __call__(self, data: Mapping[Hashable, ...]) -> tuple[torch.Tensor, bool, int, Path]:
        """the image data returned here has range of [0, 1]"""
        img: MetaTensor = data['img']
        img_t, not_rgb = ensure_rgb(img.as_tensor())
        return img_t, not_rgb, data['_trans']['aniso_d'], img.meta[ImageMetaKey.FILENAME_OR_OBJ]

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
