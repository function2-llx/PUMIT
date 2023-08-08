from pathlib import Path
import itertools as it

from PIL import Image
from lightning import LightningDataModule
import numpy as np

from luolib.types import tuple3_t
from monai import transforms as mt
from monai.config import PathLike
from monai.data import CacheDataset
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
    def __init__(self, *, num_fg_classes: int = 4, patch_size: tuple3_t[int], spacing: tuple3_t[float]):
        super().__init__()
        self.num_fg_classes = num_fg_classes
        self.sample_size = patch_size
        self.spacing = spacing

    @property
    def train_transform(self):
        class_indices_postfix = '_cls_indices'
        return mt.Compose(
            [
                mt.LoadImageD(['image', 'label'], image_only=True, ensure_channel_first=True),
                mt.OrientationD(['image', 'label'], 'SAR'),
                mt.ScaleIntensityRangePercentilesD('image', 0.5, 99.5, 0., 1., clip=True, channel_wise=True),
                mt.NormalizeIntensityD('image'),
                mt.ClassesToIndicesD('label', class_indices_postfix, self.num_fg_classes + 1),
                mt.SpacingD(
                    ['image', 'label'], self.spacing,
                    mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST],

                ),
                mt.OneOf([
                    mt.RandSpatialCropD(['image', 'label'], self.sample_size, random_center=True, random_size=False),
                    mt.RandCropByLabelClassesD(
                        ['image', 'label'], 'label',
                        self.sample_size,
                        [0, *it.repeat(1 / self.num_fg_classes, self.num_fg_classes)],
                        self.num_fg_classes + 1,
                        indices_key=f'label{class_indices_postfix}',
                    ),
                ]),
                mt.RandAffineD(),
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

        dataset = CacheDataset(data)