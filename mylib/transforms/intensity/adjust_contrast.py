from typing import Hashable, Mapping

import torch

from monai import transforms as mt
from monai.config import KeysCollection

from mylib.types import tuple2_t

class RandAdjustContrastD(mt.RandomizableTransform, mt.MapTransform):
    def __init__(self, keys: KeysCollection, contrast_range: tuple2_t[float], prob: float, allow_missing: bool = False):
        mt.RandomizableTransform.__init__(self, prob)
        mt.MapTransform.__init__(self, keys, allow_missing)
        self.contrast_range = contrast_range

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        self.randomize(None)
        if not self._do_transform:
            return data
        factor = self.R.uniform(*self.contrast_range)
        d = dict(data)
        sample_x = d[self.first_key(d)]
        spatial_dims = sample_x.ndim - 1
        reduce_dims = tuple(range(1, spatial_dims + 1))
        for key in self.key_iterator(d):
            x = d[key]
            mean = x.mean(dim=reduce_dims, keepdim=True)
            x.mul_(factor).add_(mean, alpha=1 - factor).clamp_(0, 1)
        return d
