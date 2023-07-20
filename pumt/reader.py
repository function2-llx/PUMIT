from typing import Any
from collections.abc import Sequence

import numpy as np
import torch.distributed

from monai.config import PathLike
from monai.data import ImageReader, is_supported_format
from monai import transforms as mt
from monai.utils import ensure_tuple, MetaKeys

def get_time_ms():
    from time import monotonic_ns
    return monotonic_ns() / 1e6

class PUMTReader(mt.Randomizable, ImageReader):
    def __init__(self, max_slices: int | None = None):
        self.max_slices = max_slices

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        return True

    def read(self, filepaths: Sequence[PathLike] | PathLike, **kwargs) -> Sequence[Any] | Any:
        img_ = []

        filepaths: Sequence[PathLike] = ensure_tuple(filepaths)
        for filepath in filepaths:
            img: np.memmap = np.load(filepath, 'r')
            if self.max_slices is not None and (r := img.shape[1] - self.max_slices) > 0:
                start_slice = self.R.randint(r)
                img = img[:, start_slice:start_slice + self.max_slices]
            img_.append(img)
        return img_ if len(filepaths) > 1 else img_[0]

    def get_data(self, img: np.memmap) -> tuple[np.ndarray, dict]:
        return img, {
            MetaKeys.AFFINE: np.eye(4),
            MetaKeys.ORIGINAL_AFFINE: np.eye(4),
            MetaKeys.SPATIAL_SHAPE: img.shape[1:],
            MetaKeys.ORIGINAL_CHANNEL_DIM: 0,
        }
