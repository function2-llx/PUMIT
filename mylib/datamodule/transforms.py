from collections.abc import Hashable

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from monai import transforms as monai_t

from umei.utils import DataKey

class ImageNetNormalizeMixin:
    @property
    def img_keys(self) -> list[Hashable]:
        return [DataKey.IMG]

    def intensity_normalize_transform(self, _stage):
        return [
            monai_t.ScaleIntensityRangeD(self.img_keys, 0, 255),
            monai_t.NormalizeIntensityD(
                self.img_keys,
                subtrahend=IMAGENET_DEFAULT_MEAN,
                divisor=IMAGENET_DEFAULT_STD,
                channel_wise=True,
            ),
        ]
