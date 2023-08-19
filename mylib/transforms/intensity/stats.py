from collections.abc import Hashable, Mapping

from monai import transforms as mt
from monai.data import MetaTensor

class CleverStats(mt.Transform):
    def __call__(self, x: MetaTensor):
        mean = x.new_empty((x.shape[0], ))
        std = x.new_empty((x.shape[0], ))
        for i, v in enumerate(x):
            v = v[v > 0]
            mean[i] = v.mean()
            std[i] = v.std()
        x.meta['mean'] = mean
        x.meta['std'] = std
        return x

class CleverStatsD(mt.MapTransform):
    def __call__(self, data: Mapping[Hashable, ...]):
        stats = CleverStats()
        data = dict(data)
        for key in self.key_iterator(data):
            data[key] = stats(data[key])
        return data
