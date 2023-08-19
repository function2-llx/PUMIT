from collections.abc import Hashable, Mapping

from monai import transforms as mt

class CreateForegroundMaskD(mt.Transform):
    def __init__(self, ref_key: Hashable, fg_mask_key: Hashable):
        self.ref_key = ref_key
        self.fg_mask_key = fg_mask_key

    def __call__(self, data: Mapping[Hashable, ...]):
        data = dict(data)
        data[self.fg_mask_key] = data[self.ref_key] > 0
        return data
