from collections.abc import Callable
from dataclasses import dataclass

import cytoolz
import pandas as pd
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import WeightedRandomSampler

from luolib.types import RangeTuple
from luolib.utils import DataKey
from monai.data import DataLoader, Dataset
from monai import transforms as mt

from pumt.reader import PUMTReader
from pumt.transforms import KeepSpacingD, RandAffineCropD, CenterScaleCropD, NormalizeIntensityD

DATA_ROOT = Path('datasets-PUMT')

@dataclass(kw_only=True)
class DataLoaderConf:
    train_batch_size: int
    val_batch_size: int = 16
    num_train_steps: int
    num_workers: int

@dataclass
class TransformConf:
    train_num_tokens: int = 1024
    val_num_tokens: int = 4096
    rotate_p: float = 0.3
    scale_p: float = 0.3
    scale_x_p: float = 0.5
    train_scale_x: RangeTuple[float] = (0.75, 2)
    val_scale_x: float = 1.5
    scale_z_p: float = 0.3
    scale_z: RangeTuple[float] = (0.8, 1.25)
    train_tx: RangeTuple[int] = (12, 16)
    val_tx: int = 16
    stride: int = 16

class TokenizerDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_weights: dict[str, float],
        dl_conf: DataLoaderConf,
        trans_conf: TransformConf
    ):
        super().__init__()
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        for dataset_dir in DATA_ROOT.iterdir():
            dataset_name = dataset_dir.name
            dataset_weight = dataset_weights.get(dataset_name, 1.)
            meta = pd.read_csv(dataset_dir / 'images-meta.csv', index_col='key')
            meta['weight'] *= dataset_weight
            meta[DataKey.IMG] = meta.index.map(lambda key: dataset_dir / 'data' / f'{key}.npz')
            for modality in meta['modality'].unique():
                sub_meta = meta[meta['modality'] == modality]
                val_sample = sub_meta.sample(1, weights=sub_meta['weight'])
                self.train_data = pd.concat([self.train_data, sub_meta.drop(index=val_sample.index)])
                self.val_data = pd.concat([self.val_data, val_sample])
        self.dl_conf = dl_conf
        self.trans_conf = trans_conf

    def train_transform(self) -> Callable:
        trans_conf = self.trans_conf
        return mt.Compose(
            [
                mt.LoadImageD(DataKey.IMG, PUMTReader, image_only=True),
                RandAffineCropD(
                    trans_conf.train_num_tokens,
                    trans_conf.rotate_p,
                    trans_conf.scale_x_p,
                    trans_conf.train_scale_x,
                    trans_conf.scale_z_p,
                    trans_conf.scale_z,
                    trans_conf.train_tx,
                    trans_conf.stride,
                ),
                NormalizeIntensityD(),
                KeepSpacingD(),
            ],
            lazy=True,
        )

    def train_dataloader(self):
        conf = self.dl_conf
        return DataLoader(
            Dataset(self.train_data.to_dict('records'), self.train_transform()),
            conf.num_workers,
            batch_size=conf.train_batch_size,
            sampler=WeightedRandomSampler(
                self.train_data['weight'].to_numpy(),
                conf.num_train_steps * conf.train_batch_size,
            ),
            collate_fn=cytoolz.identity,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_transform(self) -> Callable:
        trans_conf = self.trans_conf
        return mt.Compose(
            [
                mt.LoadImageD(DataKey.IMG, PUMTReader, image_only=True),
                CenterScaleCropD(
                    trans_conf.val_num_tokens,
                    trans_conf.val_scale_x,
                    trans_conf.val_tx,
                    trans_conf.stride,
                ),
                NormalizeIntensityD(),
                KeepSpacingD(),
            ],
            lazy=True,
        )

    def val_dataloader(self):
        conf = self.dl_conf
        return DataLoader(
            Dataset(self.val_data.to_dict('records'), self.val_transform()),
            conf.num_workers,
            batch_size=conf.val_batch_size,
            collate_fn=cytoolz.identity,
        )
