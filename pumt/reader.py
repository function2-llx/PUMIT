from typing import Any
from collections.abc import Sequence

import numpy as np

from monai.config import PathLike
from monai.data import ImageReader, is_supported_format
from monai.utils import ensure_tuple, MetaKeys

class PUMTReader(ImageReader):
    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        return is_supported_format(filename, ['npz'])

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs) -> Sequence[Any] | Any:
        img_ = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        for name in filenames:
            img = np.load(name)
            img_.append(img)
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        array = img['array']
        affine = img['affine']
        return array, {
            MetaKeys.AFFINE: affine,
            MetaKeys.ORIGINAL_AFFINE: affine,
            MetaKeys.SPATIAL_SHAPE: array.shape[1:],
            MetaKeys.ORIGINAL_CHANNEL_DIM: 0,
        }
