from typing import Hashable, Mapping

import einops
import numpy as np
import torch
from torch.nn import functional as nnf

from monai import transforms as mt
from monai.config import KeysCollection

class SimulateLowResolutionD(mt.RandomizableTransform, mt.MapTransform):
    def __init__(self, keys: KeysCollection, zoom_range: tuple[float, float], prob: float, dummy_dim: int | None = None, allow_missing: bool = False):
        mt.RandomizableTransform.__init__(self, prob)
        mt.MapTransform.__init__(self, keys, allow_missing)
        self.zoom_range = zoom_range
        self.dummy_dim = dummy_dim

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        self.randomize(None)
        if not self._do_transform:
            return data
        d = dict(data)
        zoom_factor = self.R.uniform(*self.zoom_range)
        for key in self.key_iterator(d):
            x = d[key]
            spatial_shape = np.array(x.shape[1:])
            if self.dummy_dim is not None:
                dummy_size = spatial_shape[self.dummy_dim]
                spatial_shape = np.delete(spatial_shape, self.dummy_dim)
                x = x.movedim(self.dummy_dim + 1, 1)
                x = einops.rearrange(x, 'c d ... -> (c d) ...')

            downsample_shape = (spatial_shape * zoom_factor).astype(np.int16)
            x = x[None]
            x = nnf.interpolate(x, tuple(downsample_shape), mode='nearest-exact')
            # no tricubic at PyTorch 2.0, use linear interpolation for both 2D & 3D
            x = nnf.interpolate(x, tuple(spatial_shape), mode='bilinear' if len(spatial_shape) == 2 else 'trilinear')
            x = x[0]
            if self.dummy_dim is not None:
                x = einops.rearrange(x, '(c d) ... -> c d ...', d=dummy_size)
                x = x.movedim(1, self.dummy_dim + 1)
            d[key] = x
        return d
