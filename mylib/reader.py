from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from monai.config import PathLike
from monai.data import ImageReader, is_supported_format, NumpyReader
from monai.utils import ensure_tuple


class PyTorchReader(ImageReader):
    def __init__(self, channel_dim: str | int | None = None, **kwargs):
        self.channel_dim = channel_dim
        self.kwargs = kwargs

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        suffixes = ['.pt']
        return is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike, **kwargs) -> Sequence[Any] | Any:
        img_: list[np.ndarray] = []

        filenames: Sequence[PathLike] = ensure_tuple(data)
        kwargs_ = self.kwargs.copy()
        kwargs_.update(kwargs)
        for name in filenames:
            img = torch.load(name, 'cpu', **kwargs_).numpy()
            img_.append(img)

        return img_ if len(img_) > 1 else img_[0]

    def get_data(self, img) -> tuple[np.ndarray, dict]:
        return NumpyReader.get_data(self, img)
