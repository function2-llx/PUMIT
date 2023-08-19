import numpy as np

from monai import transforms as mt

__all__ = [
    'RandomizableLoadImage',
    'RandomizableLoadImageD',
]

class RandomizableLoadImage(mt.Randomizable, mt.LoadImage):
    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
        for reader in self.readers:
            if isinstance(reader, mt.Randomizable):
                reader.set_random_state(seed, state)
        return self

class RandomizableLoadImageD(mt.Randomizable, mt.LoadImageD):
    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None):
        RandomizableLoadImage.set_random_state(self._loader, seed, state)
