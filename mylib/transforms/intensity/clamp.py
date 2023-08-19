from collections.abc import Hashable, Mapping

import torch

from monai import transforms as mt
from monai.config import KeysCollection

class ClampIntensityD(mt.MapTransform):
    def __init__(self, keys: KeysCollection, min_v: float = 0., max_v: float = 1., allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.min_v = min_v
        self.max_v = max_v

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        data = dict(data)
        for k in self.key_iterator(data):
            data[k] = data[k].clamp(self.min_v, self.max_v)
        return data
