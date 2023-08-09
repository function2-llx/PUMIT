from collections.abc import Hashable
from pathlib import Path

from PIL import Image
import einops
from lightning import LightningDataModule
import numpy as np
from torch.utils.data import RandomSampler

from luolib import transforms as lt
from luolib.types import tuple3_t
from monai import transforms as mt
from monai.config import PathLike
from monai.data import CacheDataset, DataLoader, Dataset
from monai.utils import GridSampleMode, GridSamplePadMode

DATASET_ROOT = Path('downstream/data/CHAOS')

mr_mapping = {
    0: 0,
    63: 1,
    126: 2,
    189: 3,
    252: 4,
}

inv_mr_mapping = {
    v: k
    for k, v in mr_mapping.items()
}

def convert_label(label: np.ndarray):
    label = np.vectorize(mr_mapping.get)(label)
    return np.flip(np.rot90(label), 0)

def convert_pred(pred: np.ndarray):
    pred = np.vectorize(inv_mr_mapping.get, otypes=[np.uint8])(pred)
    return np.rot90(np.flip(pred, 0), 3)

def read_label(label_dir: PathLike):
    label_dir = Path(label_dir)
    png_files = sorted(label_dir.glob('*.png'))
    label = np.stack(
        [np.array(Image.open(png_path)) for png_path in png_files],
        axis=-1,
    )
    label = convert_label(label)
    return label.astype(np.uint8)

def save_pred(pred: np.ndarray, save_dir: PathLike, dicom_ref_dir: PathLike):
    save_dir = Path(save_dir)
    dicom_ref_dir = Path(dicom_ref_dir)
    dcm_files = sorted(dicom_ref_dir.glob('*.dcm'))
    pred = convert_pred(pred)
    for i in range(pred.shape[2]):
        Image.fromarray(pred[..., i]).save(save_dir / f'{dcm_files[i].stem}.png')

class InputTransformD(mt.Transform):
    def __call__(self, data: dict[Hashable, ...]):
        data = dict(data)
        img = data['image']
        mean, std = img.meta['mean'], img.meta['std']
        if img.shape[0] == 1:
            img = einops.repeat(img, '1 ... -> c ...', c=2)
            mean = einops.repeat(mean, '1 -> c', c=2)
            std = einops.repeat(std, '1 -> c', c=2)
        mean = einops.rearrange(mean, 'c -> c 1 1 1')
        std = einops.rearrange(std, 'c -> c 1 1 1')
        if (label := data.get('label')) is not None:
            return img.as_tensor(), mean, std, label.as_tensor()
        return img, mean, std

class CHAOSDataModule(LightningDataModule):
    def __init__(
        self, *,
        num_fg_classes: int = 4,
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
        self.num_cache_workers = num_workers if num_cache_workers is None else num_workers
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
                    scale_range=[(-0.3, 0.1), 0., 0.], isotropic_scale=True, scale_prob=0.2,
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
        data = [
            {
                'image': data_dir / 'image.nii.gz',
                'label': data_dir / 'label.nii.gz',
            }
            for data_dir in (DATASET_ROOT / 'Train').glob('*/*')
        ]
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
    def predict_transform(self):
        return mt.Compose(
            [
                mt.LoadImageD('image', image_only=True, ensure_channel_first=True),
                mt.ScaleIntensityRangePercentilesD(
                    'image',
                    0.5, 99.5, 0., 1.,
                    clip=True, channel_wise=True,
                ),
                lt.CleverStatsD('image'),
                mt.OrientationD('image', 'SAR'),
                mt.SpacingD('image', self.spacing, mode=GridSampleMode.BILINEAR, padding_mode=GridSamplePadMode.ZEROS),
                InputTransformD(),
            ],
            lazy=True,
        )

    def predict_dataloader(self):
        data = [
            {'image': data_dir / 'image.nii.gz'}
            for data_dir in (DATASET_ROOT / 'Test').glob('*/*')
        ]
        dataset = Dataset(data, self.predict_transform)
        return DataLoader(dataset, self.num_workers, pin_memory=True)
