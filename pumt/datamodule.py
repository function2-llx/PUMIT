from collections.abc import Callable
from dataclasses import dataclass

from lightning import LightningDataModule
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.types import Device
from torch.utils.data import Dataset as TorchDataset, Sampler

from luolib import transforms as lt
from luolib.types import tuple2_t
from luolib.utils import DataKey
from monai.config import PathLike
from monai.data import Dataset as MONAIDataset, DataLoader
from monai import transforms as mt

from pumt.reader import PUMTReader
from pumt.transforms import AsSpatialTensorD, RandAffineCropD, CenterScaleCropD, UpdateSpacingD, ensure_rgb

DATA_ROOT = Path('processed-data')

@dataclass(kw_only=True)
class DataLoaderConf:
    train_batch_size: int
    val_batch_size: int = 1  # or help me write another distributed batch sampler for validation
    num_train_batches: int
    num_workers: int = 8

@dataclass(kw_only=True)
class TransformConf:
    train_tz: int
    val_tz: int = 4
    train_tx: int
    val_tx: int = 12
    rotate_p: float = 0.3
    scale_z: tuple2_t[float] = (0.8, 1.25)
    scale_z_p: float = 0.3
    train_scale_x: tuple2_t[float]
    scale_x_p: float = 0.5
    val_scale_x: float = 1.5
    flip_p: float = 0.5
    scale_intensity: float = 0.25
    scale_intensity_p: float = 0.15
    shift_intensity: float = 0.1
    shift_intensity_p: float = 0.
    adjust_contrast: tuple2_t[float] = (0.75, 1.25)
    adjust_contrast_p: float = 0.15
    gamma: tuple2_t[float] = (0.7, 1.5)
    gamma_p: float = 0.3
    stride: int = 16

class PUMTDistributedBatchSampler(Sampler[list[tuple[int, dict]]]):
    def __init__(
        self,
        data: list[dict],
        num_batches: int,
        num_skip_batches: int,
        trans_conf: TransformConf,
        num_replicas: int,
        rank: int,
        random_state: np.random.RandomState,
        batch_size: int | None = None,
        weight: torch.Tensor | None = None,
        buffer_size: int = 16384,
    ):
        super().__init__(data)
        self.data = data
        self.num_batches = num_batches
        self.num_skip_batches = num_skip_batches
        self.trans_conf = trans_conf
        self.num_replicas = num_replicas
        self.rank = rank
        self.R = random_state
        self.batch_size = batch_size
        self.weight = weight
        self.buffer_size = buffer_size

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        remain_batches = self.num_batches
        trans_conf = self.trans_conf
        bucket = {}
        next_rank = 0
        num_skipped_batches = 0
        while remain_batches > 0:
            for i in self.weight.multinomial(self.buffer_size).tolist():
                data = self.data[i]
                shape = np.array([data[f'shape-{i}'] for i in range(3)])
                origin_size_x = min(shape[1:])
                spacing = np.array([data[f'space-{i}'] for i in range(3)])
                spacing_z = spacing[0]
                spacing_x = min(spacing[1:])
                # di: how much to downsample along axis i
                dx = trans_conf.stride
                # ti: n. tokens along axis i
                tx = trans_conf.train_tx
                size_x = tx * dx
                if self.R.uniform() < trans_conf.scale_x_p:
                    scale_x = self.R.uniform(
                        trans_conf.train_scale_x[0],
                        min(origin_size_x / size_x, trans_conf.train_scale_x[1]),
                    )
                else:
                    scale_x = 1.
                if spacing_z <= 3 * spacing_x and self.R.uniform() < trans_conf.scale_z_p:
                    scale_z = self.R.uniform(*trans_conf.scale_z)
                else:
                    scale_z = 1.
                # ratio of spacing z / spacing x
                rz = np.clip(spacing_z * scale_z / (spacing_x * scale_x), 1, dx << 1)
                aniso_d = int(rz).bit_length() - 1
                dz = max(dx >> aniso_d, 1)
                tz = 1 if (1 << aniso_d > dx) else trans_conf.train_tz
                trans_info = {
                    'aniso_d': aniso_d,
                    'scale': (scale_z, scale_x, scale_x),
                    'size': (tz * dz, size_x, size_x),
                }
                bucket.setdefault(aniso_d, []).append((i, trans_info))
                if len(batch := bucket[aniso_d]) == self.batch_size:
                    if next_rank == self.rank:
                        if num_skipped_batches >= self.num_skip_batches:
                            yield batch
                        else:
                            num_skipped_batches += 1
                        remain_batches -= 1
                        if remain_batches == 0:
                            break
                    next_rank = (next_rank + 1) % self.num_replicas
                    bucket.pop(aniso_d)

class PUMTDataset(TorchDataset):
    def __init__(self, data: list[dict], transform: Callable):
        self.data = data
        self.transform = transform

    def __getitem__(self, item: tuple[int, dict]):
        index, trans = item
        data = dict(self.data[index])
        data['_trans'] = trans
        return mt.apply_transform(self.transform, data)

class PUMTDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_weights: dict[str, float],
        dl_conf: DataLoaderConf,
        trans_conf: TransformConf,
        seed: int | None = 42,
        device: Device = None,
    ):
        super().__init__()
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        self.R = np.random.RandomState(seed)
        dataset_names = []
        for dataset_dir in DATA_ROOT.iterdir():
            dataset_name = dataset_dir.name
            if (dataset_dir / 'images-meta.csv').exists():
                dataset_names.append(dataset_name)
            else:
                print(f'skip {dataset_name}')
        # deterministic
        dataset_names = sorted(dataset_names)
        for dataset_name in dataset_names:
            dataset_dir = DATA_ROOT / dataset_name
            dataset_weight = dataset_weights.get(dataset_name, 1.)
            if not (meta_path := dataset_dir / 'images-meta.csv').exists():
                print(f'skip {dataset_name}')
                continue
            meta = pd.read_csv(meta_path, dtype={'key': 'string'})
            meta['weight'] *= dataset_weight
            meta[DataKey.IMG] = meta['key'].map(lambda key: dataset_dir / 'data' / f'{key}.npy')
            for modality in meta['modality'].unique():
                if pd.isna(modality):
                    print(f"{dataset_name} missing {pd.isna(meta['modality']).sum()}")
                else:
                    sub_meta = meta[meta['modality'] == modality]
                    val_sample = sub_meta.sample(1, weights=sub_meta['weight'], random_state=self.R)
                    self.train_data = pd.concat([self.train_data, sub_meta.drop(index=val_sample.index)])
                    self.val_data = pd.concat([self.val_data, val_sample])
        self.dl_conf = dl_conf
        self.trans_conf = trans_conf
        self.device = device

    def setup_ddp(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size

    def setup(self, stage: str) -> None:
        self.setup_ddp(self.trainer.global_rank, self.trainer.world_size)

    def train_transform(self) -> Callable:
        # TODO: also enable skipping the transform deterministically?
        trans_conf = self.trans_conf
        return mt.Compose(
            [
                lt.RandomizableLoadImageD(DataKey.IMG, PUMTReader(int(1.5 * trans_conf.train_tz * trans_conf.stride)), image_only=True),
                mt.ToDeviceD(DataKey.IMG, self.device),
                UpdateSpacingD(),
                RandAffineCropD(trans_conf.rotate_p),
                *[
                    mt.RandFlipD(DataKey.IMG, prob=trans_conf.flip_p, spatial_axis=i)
                    for i in range(3)
                ],
                mt.RandScaleIntensityD(DataKey.IMG, trans_conf.scale_intensity, trans_conf.scale_intensity_p),
                mt.RandShiftIntensityD(DataKey.IMG, trans_conf.shift_intensity, prob=trans_conf.shift_intensity_p),
                lt.ClampIntensityD(DataKey.IMG),
                lt.RandAdjustContrastD(DataKey.IMG, trans_conf.adjust_contrast, trans_conf.adjust_contrast_p),
                mt.OneOf([
                    lt.RandGammaCorrectionD(DataKey.IMG, trans_conf.gamma_p, trans_conf.gamma, False),
                    lt.RandGammaCorrectionD(DataKey.IMG, trans_conf.gamma_p, trans_conf.gamma, True),
                ]),
                AsSpatialTensorD(),
            ],
            lazy=True,
        )

    @staticmethod
    def collate_fn(batch: list[tuple[torch.Tensor, int, PathLike]]):
        tensor_list, [aniso_d, *aniso_d_list], paths = zip(*batch)
        assert (np.array(aniso_d_list) == aniso_d).all()
        return torch.stack(tensor_list), aniso_d, paths

    def train_dataloader(self, num_skip_batches: int = 0):
        conf = self.dl_conf
        data = self.train_data.to_dict('records')
        weight = torch.from_numpy(self.train_data['weight'].to_numpy())
        return DataLoader(
            PUMTDataset(data, self.train_transform()),
            conf.num_workers,
            batch_sampler=PUMTDistributedBatchSampler(
                data, conf.num_train_batches, num_skip_batches, self.trans_conf,
                self.world_size, self.rank, self.R, conf.train_batch_size, weight,
            ),
            prefetch_factor=8 if conf.num_workers > 0 else None,
            collate_fn=self.collate_fn,
            pin_memory=True,
            persistent_workers=conf.num_workers > 0,
        )

    def val_transform(self) -> Callable:
        trans_conf = self.trans_conf
        return mt.Compose(
            [
                mt.LoadImageD(DataKey.IMG, PUMTReader, image_only=True),
                CenterScaleCropD(
                    trans_conf.val_tz,
                    trans_conf.val_tx,
                    trans_conf.val_scale_x,
                    trans_conf.stride,
                ),
                AsSpatialTensorD(),
            ],
            lazy=True,
        )

    def val_dataloader(self):
        conf = self.dl_conf
        data = self.val_data.to_dict('records')
        return DataLoader(
            MONAIDataset(data, self.val_transform()),
            conf.num_workers,
            batch_size=1,
            collate_fn=self.collate_fn,
        )
