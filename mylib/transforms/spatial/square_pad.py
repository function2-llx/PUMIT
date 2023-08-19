import monai
from monai.config import KeysCollection, NdarrayOrTensor

class SpatialSquarePad(monai.transforms.SpatialPad):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(-1, **kwargs)

    def __call__(self, data: NdarrayOrTensor, **kwargs):
        size = max(data.shape[1:3])
        self.spatial_size = [size, size, -1]
        return super().__call__(data, **kwargs)

# FIXME: set padder
class SpatialSquarePadD(monai.transforms.SpatialPadD):
    def __init__(
        self,
        keys: KeysCollection,
        **kwargs,
    ) -> None:
        super().__init__(keys, -1, **kwargs)
