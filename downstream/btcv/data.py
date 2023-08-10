from collections.abc import Hashable
import itertools as it
import os
from pathlib import Path
import sys

from lightning import LightningDataModule
import numpy as np
from torch.utils.data import RandomSampler

from luolib import transforms as lt
from luolib.types import tuple3_t
from monai import transforms as mt
from monai.data import CacheDataset, DataLoader, Dataset, load_decathlon_datalist
from monai.utils import GridSampleMode, GridSamplePadMode

DATASET_ROOT = Path('downstream/data/BTCV')

class InputTransformD(mt.Transform):
    def __init__(self, as_tensor: bool = False):
        self.as_tensor = as_tensor

    def __call__(self, data: dict[Hashable, ...]):
        data = dict(data)
        img, label = data['image'], data['label']
        if self.as_tensor:
            img = img.as_tensor()
            label = label.as_tensor()
        return img, label

class BTCVDataModule(LightningDataModule):
    def __init__(
        self, *,
        num_fg_classes: int = 13,
        sample_size: tuple3_t[int],
        spacing: tuple3_t[float],
        num_workers: int = os.cpu_count() >> 2,
        num_cache_workers: int | None = None,
        train_batch_size: int,
        num_train_batches: int,
        cache_num: int = sys.maxsize,
    ):
        super().__init__()
        self.num_fg_classes = num_fg_classes
        self.sample_size = sample_size
        self.spacing = spacing
        self.num_workers = num_workers
        self.num_cache_workers = num_workers if num_cache_workers is None else num_cache_workers
        self.train_batch_size = train_batch_size
        self.num_train_batches = num_train_batches
        self.cache_num = cache_num

    @property
    def train_transform(self):
        indices_postfix = '_cls_indices'
        indices_key = f'label{indices_postfix}'
        return mt.Compose(
            [
                mt.LoadImageD(['image', 'label'], image_only=True, ensure_channel_first=True),
                mt.ScaleIntensityRangeD(
                    'image',
                    -175, 250, 0., 1.,
                    clip=True,
                ),
                mt.OrientationD(['image', 'label'], 'SAR'),
                mt.SpacingD(
                    ['image', 'label'],
                    self.spacing,
                    mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
                    padding_mode=GridSamplePadMode.ZEROS,
                ),
                mt.SpatialPadD(['image', 'label'], self.sample_size),
                mt.ClassesToIndicesD('label', indices_postfix, self.num_fg_classes + 1),
                mt.OneOf(
                    [
                        mt.RandSpatialCropD(['image', 'label'], self.sample_size, random_center=True, random_size=False),
                        mt.RandCropByLabelClassesD(
                            ['image', 'label'], 'label',
                            self.sample_size,
                            [0, *it.repeat(1, self.num_fg_classes)],
                            num_classes=self.num_fg_classes + 1,
                            indices_key=indices_key,
                        ),
                    ],
                    (2, 1),
                    apply_pending=False,
                ),
                mt.RandAffineD(
                    ['image', 'label'], self.sample_size, 1.,
                    rotate_range=(np.pi / 2, 0, 0), rotate_prob=0.2,
                    scale_range=[(-0.3, 0.4), 0., 0.], isotropic_scale=True, scale_prob=0.2,
                    mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
                ),
                mt.RandFlipD(['image', 'label'], 0.5, 0),
                mt.RandFlipD(['image', 'label'], 0.5, 1),
                mt.RandFlipD(['image', 'label'], 0.5, 2),
                mt.RandGaussianNoiseD('image', 0.1),
                mt.RandGaussianSmoothD('image', (0.8, 1.2), (0.8, 1.2), (0.8, 1.2), prob=0.1),
                mt.RandScaleIntensityD('image', 0.25, 0.15),
                lt.RandAdjustContrastD('image', (0.75, 1.25), 0.15),
                lt.SimulateLowResolutionD('image', (0.5, 1.), 0.15, 0),
                mt.OneOf(
                    [
                        mt.Identity(),
                        lt.RandGammaCorrectionD('image', 1., (0.7, 1.5), False),
                        lt.RandGammaCorrectionD('image', 1., (0.7, 1.5), True),
                    ],
                    weights=(9, 3, 1),
                ),
                InputTransformD(as_tensor=True),
            ],
            lazy=True,
            overrides={
                # https://github.com/Project-MONAI/MONAI/issues/6850
                'image': {
                    'mode': GridSampleMode.BILINEAR,
                    'padding_mode': GridSamplePadMode.ZEROS,
                },
                'label': {
                    'mode': GridSampleMode.NEAREST,
                    'padding_mode': GridSamplePadMode.ZEROS,
                },
            }
        )

    def train_dataloader(self):
        data = load_decathlon_datalist(DATASET_ROOT / 'smit.json', data_list_key='training')
        dataset = CacheDataset(data, self.train_transform, cache_num=self.cache_num, num_workers=self.num_cache_workers)
        return DataLoader(
            dataset,
            self.num_workers,
            sampler=RandomSampler(dataset, num_samples=self.num_train_batches * self.train_batch_size),
            batch_size=self.train_batch_size,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    @property
    def val_transform(self):
        return mt.Compose(
            [
                mt.LoadImageD(['image', 'label'], image_only=True, ensure_channel_first=True),
                mt.ScaleIntensityRangePercentilesD(
                    'image',
                    0.5, 99.5, 0., 1.,
                    clip=True, channel_wise=True,
                ),
                lt.CleverStatsD('image'),
                mt.OrientationD(['image', 'label'], 'SAR'),
                mt.SpacingD('image', self.spacing, mode=GridSampleMode.BILINEAR, padding_mode=GridSamplePadMode.ZEROS),
                InputTransformD(as_tensor=True),
            ],
            lazy=True,
            overrides={
                # https://github.com/Project-MONAI/MONAI/issues/6850
                'image': {
                    'mode': GridSampleMode.BILINEAR,
                    'padding_mode': GridSamplePadMode.ZEROS,
                },
                'label': {
                    'mode': GridSampleMode.NEAREST,
                    'padding_mode': GridSamplePadMode.ZEROS,
                },
            }
        )

    def val_dataloader(self):
        data = load_decathlon_datalist(DATASET_ROOT / 'smit.json', data_list_key='validation')
        return DataLoader(CacheDataset(data, self.val_transform), self.num_workers, pin_memory=True)
