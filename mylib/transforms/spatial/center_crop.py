from collections.abc import Hashable, Sequence, Mapping

import torch

from monai import transforms as monai_t
from monai.config import KeysCollection

class SpatialCropWithSpecifiedCenterD(monai_t.MapTransform):
    def __init__(self, keys: KeysCollection, center_key: Hashable, roi_size: Sequence[int], allow_missing_keys: bool = False):
        super().__init__(keys, allow_missing_keys)
        self.center_key = center_key
        self.roi_size = roi_size

    def __call__(self, data: Mapping[Hashable, torch.Tensor]):
        center: Sequence[int] = data[self.center_key]
        cropper = monai_t.SpatialCropD(self.keys, center, self.roi_size, allow_missing_keys=self.allow_missing_keys)
        return cropper(data)
