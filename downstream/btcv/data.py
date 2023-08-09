from collections.abc import Hashable
from pathlib import Path

import einops
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
    def __call__(self, data: dict[Hashable, ...]):
        data = dict(data)
        img = data['image']
        mean, std = img.meta['mean'], img.meta['std']
        mean = einops.rearrange(mean, 'c -> c 1 1 1')
        std = einops.rearrange(std, 'c -> c 1 1 1')
        if (label := data.get('label')) is not None:
            return img, mean, std, label.as_tensor()
        return img, mean, std

class BTCVDataModule(LightningDataModule):
    def __init__(
        self, *,
        num_fg_classes: int = 13,
        sample_size: tuple3_t[int],
        spacing: tuple3_t[float],
        num_workers: int,
        num_cache_workers: int | None = None,
        train_batch_size: int,
        num_train_batches: int,
    ):
        super().__init__()
        self.num_fg_classes = num_fg_classes
        self.sample_size = sample_size
        self.spacing = spacing
        self.num_workers = num_workers
        self.num_cache_workers = num_workers if num_cache_workers is None else num_cache_workers
        self.train_batch_size = train_batch_size
        self.num_train_batches = num_train_batches

    @property
    def train_transform(self):
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
                mt.SpacingD(
                    ['image', 'label'], self.spacing,
                    mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
                    padding_mode=GridSamplePadMode.ZEROS,
                ),
                mt.SpatialPadD(['image', 'label'], self.sample_size),
                mt.RandSpatialCropD(['image', 'label'], self.sample_size, random_center=True, random_size=False),
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
                InputTransformD(),
            ],
            lazy=True,
        )

    def train_dataloader(self):
        data = load_decathlon_datalist(DATASET_ROOT / 'smit.json', data_list_key='training')
        dataset = CacheDataset(data, self.train_transform, num_workers=self.num_cache_workers)
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
                InputTransformD(),
            ],
            lazy=True,
        )

    def val_dataloader(self):
        data = load_decathlon_datalist(DATASET_ROOT / 'smit.json', data_list_key='validation')
        return DataLoader(CacheDataset(data, self.val_transform), self.num_workers, pin_memory=True)
