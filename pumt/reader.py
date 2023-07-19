from typing import Any
from collections.abc import Sequence

import numpy as np
import torch.distributed

from monai.config import PathLike
from monai.data import ImageReader, is_supported_format
from monai.utils import ensure_tuple, MetaKeys

def get_time_ms():
    from time import monotonic_ns
    return monotonic_ns() / 1e6

class PUMTReader(ImageReader):
    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        return is_supported_format(filename, ['npz'])

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs) -> Sequence[Any] | Any:
        img_ = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        for name in filenames:
            img = np.load(name)
            img.name = name
            img_.append(img)
        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        t = get_time_ms()
        try:
            array = img['array']
            affine = img['affine']
            if get_time_ms() - t > 5000:
                print(img.name)
        except Exception as e:
            rank = torch.distributed.get_rank()
            print(f'\n\n[rank {rank}] read failed1:', img.name)
            print(f'\n\n[rank {rank}] read failed2:', img.name)
            print(f'\n\n[rank {rank}] read failed3:', img.name)
            raise e
        return array, {
            MetaKeys.AFFINE: affine,
            MetaKeys.ORIGINAL_AFFINE: affine,
            MetaKeys.SPATIAL_SHAPE: array.shape[1:],
            MetaKeys.ORIGINAL_CHANNEL_DIM: 0,
        }
