from collections.abc import Callable
from dataclasses import dataclass, field

import cytoolz
import math
import numpy as np
import pandas as pd
from pathlib import Path
from lightning import LightningDataModule
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from luolib.types import RangeTuple
from luolib.utils import DataKey
from monai.data import DataLoader, Dataset
from monai import transforms as mt
from monai.transforms import apply_transform

from pumt.reader import PUMTReader
from pumt.transforms import PrepareInputD, RandAffineCropD, CenterScaleCropD, NormalizeIntensityD

DATA_ROOT = Path('datasets-PUMT')

@dataclass(kw_only=True)
class DataLoaderConf:
    train_batch_size: int
    val_batch_size: int = 4
    num_train_steps: int
    num_workers: int = 8

@dataclass(kw_only=True)
class TransformConf:
    train_tz: int
    val_tz: int = 4
    train_tx: int = 8
    val_tx: int = 12
    rotate_p: float = 0.3
    scale_z_p: float = 0.3
    scale_z: RangeTuple = field(default_factory=lambda: RangeTuple(0.8, 1.25))
    scale_x_p: float = 0.5
    train_scale_x: RangeTuple
    val_scale_x: float = 1.5
    stride: int = 16

class TokenizerBatchSampler(Sampler[list[int]]):
    def __init__(self, meta: list[dict], batch_size: int, num_batches: int, trans_conf: TransformConf, buffer_size: int = 4096):
        super().__init__(meta)
        self.meta = meta
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.trans_conf = trans_conf
        self.buffer_size = buffer_size
        self.weight = torch.tensor([x['weight'] for x in meta])

    def __iter__(self):
        remain_batches = self.num_batches
        trans_conf = self.trans_conf
        while remain_batches > 0:
            bucket = {}
            for i in self.weight.multinomial(self.buffer_size).tolist():
                meta = self.meta[i]
                shape = np.array([meta[f'shape-{i}'] for i in range(3)])
                origin_size_x = min(shape[1:])
                spacing = np.array([meta[f'space-{i}'] for i in range(3)])
                spacing_z = spacing[0]
                spacing_x = min(spacing[1:])
                # di: how much to downsample along axis i
                dx = trans_conf.stride
                # ti: n. tokens along axis i
                tx = trans_conf.train_tx
                size_x = tx * dx
                if np.random.uniform() < trans_conf.scale_x_p:
                    scale_x = np.random.uniform(
                        trans_conf.train_scale_x.min * origin_size_x / size_x,
                        min(origin_size_x / size_x, trans_conf.train_scale_x.max),
                    )
                else:
                    scale_x = 1.
                if spacing_z <= 3 * spacing_x and np.random.uniform() < trans_conf.scale_z_p:
                    scale_z = np.random.uniform(*trans_conf.scale_z)
                else:
                    scale_z = 1.
                # ratio of spacing z / spacing x
                rz = np.clip(spacing_z * scale_z / (spacing_x * scale_x), 1, dx)
                aniso_d = int(rz).bit_length() - 1
                meta['scale'] = (scale_z, scale_x, scale_x)
                dz = dx >> aniso_d
                if shape[0] == 1:
                    tz = 1
                else:
                    tz = trans_conf.train_tz
                meta['sample size'] = (tz * dz, size_x, size_x)
                bucket.setdefault(aniso_d, []).append(i)
                if len(batch := bucket[aniso_d]) == self.batch_size:
                    yield batch
                    remain_batches -= 1
                    batch.clear()

class DataFrameDataset(Dataset):
    data: pd.DataFrame

    def _transform(self, index: int):
        data_i = self.data.iloc[index].to_dict()
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i

class TokenizerDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_weights: dict[str, float],
        dl_conf: DataLoaderConf,
        trans_conf: TransformConf,
    ):
        super().__init__()
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        for dataset_dir in DATA_ROOT.iterdir():
            dataset_name = dataset_dir.name
            dataset_weight = dataset_weights.get(dataset_name, 1.)
            meta = pd.read_csv(dataset_dir / 'images-meta.csv', dtype={'key': 'string'}).set_index('key')
            meta['weight'] *= dataset_weight
            meta[DataKey.IMG] = meta.index.map(lambda key: dataset_dir / 'data' / f'{key}.npz')
            for modality in meta['modality'].unique():
                sub_meta = meta[meta['modality'] == modality]
                val_sample = sub_meta.sample(1, weights=sub_meta['weight'])
                self.train_data = pd.concat([self.train_data, sub_meta.drop(index=val_sample.index)])
                self.val_data = pd.concat([self.val_data, val_sample])
        self.dl_conf = dl_conf
        self.trans_conf = trans_conf
        self.world_size = None

    def train_transform(self) -> Callable:
        trans_conf = self.trans_conf
        return mt.Compose(
            [
                mt.LoadImageD(DataKey.IMG, PUMTReader, image_only=True),
                NormalizeIntensityD(0., 1.),
                RandAffineCropD(
                    trans_conf.train_tz,
                    trans_conf.train_tx,
                    trans_conf.rotate_p,
                    trans_conf.scale_x_p,
                    trans_conf.train_scale_x,
                    trans_conf.scale_z_p,
                    trans_conf.scale_z,
                    trans_conf.stride,
                ),
                PrepareInputD(),
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
                conf.num_train_steps * conf.train_batch_size * self.world_size,
            ),
            # collate_fn=cytoolz.identity,
            pin_memory=True,
            persistent_workers=conf.num_workers > 0,
        )

    def val_transform(self) -> Callable:
        trans_conf = self.trans_conf
        return mt.Compose(
            [
                mt.LoadImageD(DataKey.IMG, PUMTReader, image_only=True),
                NormalizeIntensityD(0., 1.),
                CenterScaleCropD(
                    trans_conf.val_tz,
                    trans_conf.val_tx,
                    trans_conf.val_scale_x,
                    trans_conf.stride,
                ),
                PrepareInputD(),
            ],
            lazy=True,
        )

    def val_dataloader(self):
        conf = self.dl_conf
        return DataLoader(
            Dataset(self.val_data.to_dict('records'), self.val_transform()),
            conf.num_workers,
            batch_size=conf.val_batch_size,
            # collate_fn=cytoolz.identity,
        )
