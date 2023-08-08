from pathlib import Path
import itertools as it

from PIL import Image
from lightning import LightningDataModule
import numpy as np
from torch.utils.data import RandomSampler

from luolib.types import tuple3_t
from luolib import transforms as lt
from monai import transforms as mt
from monai.config import PathLike
from monai.data import CacheDataset, DataLoader
from monai.utils import GridSampleMode

mr_mapping = {
    0: 0,
    63: 1,
    126: 2,
    189: 3,
    252: 4,
}

def read_label(label_dir: PathLike):
    label_dir = Path(label_dir)
    png_files = sorted(label_dir.glob('*.png'))
    img = np.stack(
        [np.array(Image.open(png_path)) for png_path in png_files],
        axis=-1,
    )
    img = np.vectorize(mr_mapping.get)(img)
    img = np.flip(np.rot90(img), 0)
    return img.astype(np.uint8)

class CHAOSDataModule(LightningDataModule):
    def __init__(
        self, *,
        num_fg_classes: int = 4,
        patch_size: tuple3_t[int],
        spacing: tuple3_t[float],
        num_workers: int,
        num_cache_workers: int | None = None,
        train_batch_size: int,
        num_train_batches: int,
    ):
        super().__init__()
        self.num_fg_classes = num_fg_classes
        self.sample_size = patch_size
        self.spacing = spacing
        self.num_workers = num_workers
        self.num_cache_workers = num_workers if num_cache_workers is None else num_workers
        self.train_batch_size = train_batch_size
        self.num_train_batches = num_train_batches

    @property
    def train_transform(self):
        class_indices_postfix = '_cls_indices'
        fg_mask_key = 'fg_mask'
        return mt.Compose(
            [
                mt.LoadImageD(['image', 'label'], image_only=True, ensure_channel_first=True),
                mt.OrientationD(['image', 'label'], 'SAR'),
                mt.ScaleIntensityRangePercentilesD('image', 0.5, 99.5, 0., 1., clip=True, channel_wise=True),
                lt.CreateForegroundMaskD('image', fg_mask_key),
                mt.IntensityStatsD('image', ['mean', 'std'], 'stat', fg_mask_key, True),
                mt.ClassesToIndicesD('label', class_indices_postfix, self.num_fg_classes + 1),
                mt.SpacingD(
                    ['image', 'label'], self.spacing,
                    mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
                ),
                mt.OneOf(
                    [
                        mt.RandSpatialCropD(['image', 'label'], self.sample_size, random_center=True, random_size=False),
                        mt.RandCropByLabelClassesD(
                            ['image', 'label'], 'label',
                            self.sample_size,
                            [0, *it.repeat(1 / self.num_fg_classes, self.num_fg_classes)],
                            self.num_fg_classes + 1,
                            indices_key=f'label{class_indices_postfix}',
                        ),
                    ],
                    weights=(2, 1),
                ),
                mt.RandAffineD(
                    ['image', 'label'], self.sample_size, 1.,
                    rotate_range=(np.pi / 2, 0, 0), rotate_prob=0.2,
                    scale_range=(-0.3, 0.4), isotropic_scale=True, scale_prob=0.2,
                    mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST],
                ),
                mt.RandFlipD(['image', 'label'], 0.5, 0),
                mt.RandFlipD(['image', 'label'], 0.5, 1),
                mt.RandFlipD(['image', 'label'], 0.5, 2),
                mt.RandGaussianNoiseD('image', 0.1),
                mt.RandGaussianSmoothD('image', (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), prob=0.1),
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
                mt.SelectItemsD(['image', 'label']),
            ],
            lazy=True,
        )

    def train_dataloader(self):
        data = [
            {
                'image': data_dir / 'image.nii.gz',
                'label': data_dir / 'label.nii.gz',
            }
            for data_dir in Path('downstream/data/CHAOS/Train').glob('*/*')
        ]
        dataset = CacheDataset(data, self.train_transform, num_workers=self.num_cache_workers)
        return DataLoader(
            dataset,
            self.num_workers,
            sampler=RandomSampler(dataset, num_samples=self.num_train_batches * self.train_batch_size),
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )
