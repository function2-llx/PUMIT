from typing import Any, Hashable, Mapping, Sequence

import numpy as np
import torch

from monai import transforms as mt
from monai.config import DtypeLike, NdarrayOrTensor
from monai.transforms import Randomizable

from mylib.types import tuple2_t

class SpatialRangeGenerator(mt.Randomizable):
    def __init__(
        self,
        rand_range: Sequence[tuple2_t[float]] | tuple2_t[float],
        prob: Sequence[float] | float = 1.,
        default: float = 0.,
        repeat: int | None = None,  # number of times to repeat when all dimensions share transform
        dtype: DtypeLike = np.float32,
    ):
        super().__init__()
        self.rand_range = rand_range = np.array(rand_range)
        self.prob = prob = np.array(prob)
        self.default = default
        self.dtype = dtype
        if rand_range.ndim == 1:
            assert repeat is not None
            self.repeat = repeat
            # shared transform
            assert prob.ndim == 0
        else:
            # independent transform
            self.spatial_dims = rand_range.shape[0]
            if prob.ndim > 0:
                # independent prob
                assert prob.shape[0] == self.spatial_dims

    def randomize(self, *_, **__):
        match self.rand_range.ndim, self.prob.ndim:
            case 1, 0:
                # shared transform & prob
                if self.R.uniform() >= self.prob:
                    return None
                return np.repeat(self.R.uniform(*self.rand_range), self.repeat)
            case _, 0:
                # independent transform, shared prob
                if self.R.uniform() >= self.prob:
                    return None
                return np.array([self.R.uniform(*r) for r in self.rand_range])
            case _, _:
                # independent transform, independent prob
                do_transform = self.R.uniform(size=self.spatial_dims) < self.prob
                if np.any(do_transform):
                    return np.array([
                        self.R.uniform(*r) if do
                        else self.default
                        for r, do in zip(self.rand_range, do_transform)
                    ])
                else:
                    return None

    def __call__(self, *args, **kwargs):
        ret = self.randomize()
        if ret is not None:
            ret = ret.astype(self.dtype)
        return ret

class RandSpatialCenterGeneratorD(mt.Randomizable):
    def __init__(
        self,
        ref_key: str,
        roi_size: Sequence[int] | int,
        max_roi_size: Sequence[int] | int | None = None,
        random_center: bool = True,
        random_size: bool = False,
    ):
        self.ref_key = ref_key
        self.dummy_rand_cropper = mt.RandSpatialCrop(roi_size, max_roi_size, random_center, random_size)

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Randomizable:
        super().set_random_state(seed, state)
        self.dummy_rand_cropper.set_random_state(seed, state)
        return self

    def randomize(self, spatial_size: Sequence[int]):
        self.dummy_rand_cropper.randomize(spatial_size)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> tuple[int, ...]:
        spatial_size = data[self.ref_key].shape[1:]
        self.randomize(spatial_size)
        if self.dummy_rand_cropper.random_center:
            slices = self.dummy_rand_cropper._slices
        else:
            slices = mt.CenterSpatialCrop(self.dummy_rand_cropper._size).compute_slices(spatial_size)
        return tuple(
            s.start + s.stop >> 1
            for s in slices
        )

class RandCenterGeneratorByLabelClassesD(mt.Randomizable):
    def __init__(
        self,
        label_key: str,
        roi_size: Sequence[int] | int,
        ratios: list[float | int] | None = None,
        num_classes: int | None = None,
        image_key: str | None = None,
        image_threshold: float = 0.0,
        indices_key: str | None = None,
        allow_smaller: bool = False,
        warn: bool = False,
    ) -> None:
        self.label_key = label_key
        self.image_key = image_key
        self.indices_key = indices_key
        self.dummy_rand_cropper = mt.RandCropByLabelClasses(
            roi_size,
            ratios,
            num_classes=num_classes,
            image_threshold=image_threshold,
            allow_smaller=allow_smaller,
            warn=warn,
        )

    def set_random_state(self, seed: int | None = None, state: np.random.RandomState | None = None) -> Randomizable:
        super().set_random_state(seed, state)
        self.dummy_rand_cropper.set_random_state(seed, state)
        return self

    def randomize(
        self, label: torch.Tensor, indices: list[NdarrayOrTensor] | None = None, image: torch.Tensor | None = None
    ) -> None:
        self.dummy_rand_cropper.randomize(label=label, indices=indices, image=image)

    def __call__(self, data: Mapping[Hashable, Any]):
        d = dict(data)
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        indices = d.pop(self.indices_key, None) if self.indices_key is not None else None
        self.randomize(label, indices, image)

        return self.dummy_rand_cropper.centers[0]

class FilterInstanceD(mt.Transform):
    def __init__(self, class_key: Hashable, mask_key: Hashable):
        self.class_key = class_key
        self.mask_key = mask_key

    def __call__(self, data: Mapping[Hashable, Any]):
        d = dict(data)
        class_label: torch.Tensor = d[self.class_key]
        mask_label: torch.Tensor = d[self.mask_key]

        filter_idx = mask_label.any(dim=-1)
        filter_idx = filter_idx.view(filter_idx.shape[0], -1).any(dim=1)
        d[self.class_key] = class_label[filter_idx]
        d[self.mask_key] = mask_label[filter_idx]
        return d
