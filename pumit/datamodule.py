from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from lightning import LightningDataModule
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset as TorchDataset, Sampler

from luolib import transforms as lt
from luolib.types import tuple2_t
from luolib.utils import DataKey
from monai.config import PathLike
from monai.data import Dataset as MONAIDataset, DataLoader
from monai import transforms as mt

from pumit.reader import PUMITReader
from pumit.transforms import AdaptivePadD, AsSpatialTensorD, RandAffineCropD, UpdateSpacingD

DATA_ROOT = Path('processed-data')

@dataclass(kw_only=True)
class DataLoaderConf:
    train_batch_size: int | None = None
    val_batch_size: int = 1  # default=1, or help me write another distributed batch sampler for validation
    num_train_batches: int | None = None
    num_workers: int = 8

@dataclass(kw_only=True)
class TransformConf:
    train_tz: int
    val_tz: int = 4
    train_token_xy: int
    val_tx: int = 12
    rotate_p: float = 0.3
    scale_z: tuple2_t[float] = (0.8, 1.25)
    scale_z_p: float = 0.3
    train_scale_xy: tuple2_t[float]
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
    isotropic_th: float = 3.

class PUMITDistributedBatchSampler(Sampler[list[tuple[int, dict]]]):
    """put samples with the same DA into a batch"""
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
        super().__init__()
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
        bucket = {}
        next_rank = 0
        num_skipped_batches = 0
        while remain_batches > 0:
            for i in self.weight.multinomial(self.buffer_size).tolist():
                data = self.data[i]
                aniso_d, trans_info = self.gen_trans_info(data)
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

    def gen_trans_info(self, data: dict):
        trans_conf = self.trans_conf
        shape = np.array(data['shape'])
        origin_size_xy = shape[1:].min().item()
        spacing = data['spacing']
        spacing_z = spacing[0]
        spacing_xy = min(spacing[1:])

        # down_i: how much to downsample along axis i
        down_xy = trans_conf.stride
        # token_i: number of tokens along axis i
        token_xy = trans_conf.train_token_xy
        size_xy = token_xy * down_xy
        if self.R.uniform() < trans_conf.scale_x_p:
            scale_xy = self.R.uniform(
                trans_conf.train_scale_xy[0],
                min(origin_size_xy / size_xy, trans_conf.train_scale_xy[1]),
            )
        else:
            scale_xy = 1.

        if spacing_z <= trans_conf.isotropic_th * spacing_xy and self.R.uniform() < trans_conf.scale_z_p:
            scale_z = self.R.uniform(*trans_conf.scale_z)
        else:
            scale_z = 1.
        # ratio of spacing z / spacing x after scaling
        rz = np.clip(spacing_z * scale_z / (spacing_xy * scale_xy), 1, down_xy << 1)
        aniso_d = int(rz).bit_length() - 1
        down_z = max(down_xy >> aniso_d, 1)
        token_z = 1 if (1 << aniso_d > down_xy) else trans_conf.train_tz

        trans_info = {
            'aniso_d': aniso_d,
            'scale': np.array((scale_z, scale_xy, scale_xy)),
            'size': np.array((token_z * down_z, size_xy, size_xy)),
        }
        return aniso_d, trans_info

class PUMITDataset(TorchDataset):
    def __init__(self, data: list[dict], transform: Callable):
        self.data = data
        self.transform = transform

    def __getitem__(self, item: tuple[int, dict]):
        index, trans_info = item
        data = dict(self.data[index])
        data['_trans'] = trans_info
        return mt.apply_transform(self.transform, data)

class PUMITDataModule(LightningDataModule):
    def __init__(
        self,
        dataset_weights: dict[str, float],
        dl_conf: DataLoaderConf,
        trans_conf: TransformConf,
        seed: int | None = 42,
        device: Literal['cpu', 'cuda'] = 'cpu',
    ):
        super().__init__()
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        self.R = np.random.RandomState(seed)
        dataset_names = []
        for dataset_dir in DATA_ROOT.iterdir():
            dataset_name = dataset_dir.name
            if (dataset_dir / 'meta.pkl').exists():
                dataset_names.append(dataset_name)
            else:
                print(f'skip {dataset_name}')
        # deterministic
        dataset_names = sorted(dataset_names)
        for dataset_name in dataset_names:
            dataset_dir = DATA_ROOT / dataset_name
            dataset_weight = dataset_weights.get(dataset_name, 1.)
            meta: pd.DataFrame = pd.read_pickle(dataset_dir / 'meta.pkl')
            meta['weight'] *= dataset_weight
            meta['img'] = meta['key'].map(lambda key: dataset_dir / 'data' / f'{key}.npy')
            for modality in meta['modality'].unique():
                sub_meta = meta[meta['modality'] == modality]
                val_sample = sub_meta.sample(1, weights=sub_meta['weight'], random_state=self.R)
                self.train_data = pd.concat([self.train_data, sub_meta.drop(index=val_sample.index)])
                self.val_data = pd.concat([self.val_data, val_sample])
        self.dl_conf = dl_conf
        self.trans_conf = trans_conf
        self.device = torch.device(device)

    def setup_ddp(self, local_rank: int, global_rank: int, world_size: int):
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        if self.device.type == 'cuda':
            self.device = torch.device(local_rank)

    def setup(self, stage: str) -> None:
        self.setup_ddp(self.trainer.local_rank, self.trainer.global_rank, self.trainer.world_size)

    def train_transform(self) -> Callable:
        # TODO: also enable skipping the transform deterministically?
        # note: intensity range is [0, 1] before and after transform
        trans_conf = self.trans_conf
        return mt.Compose(
            [
                lt.RandomizableLoadImageD(
                    DataKey.IMG,
                    PUMITReader(int(1.5 * trans_conf.train_tz * trans_conf.stride)),
                    image_only=True,
                ),
                mt.ToDeviceD(DataKey.IMG, self.device),
                UpdateSpacingD(),
                RandAffineCropD(trans_conf.rotate_p, trans_conf.isotropic_th),
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
        tensor_list, [aniso_d, *aniso_d_list], *info = zip(*batch)
        assert (np.array(aniso_d_list) == aniso_d).all()
        return torch.stack(tensor_list), aniso_d, *info

    def train_dataloader(self, num_skip_batches: int = 0):
        conf = self.dl_conf
        data = self.train_data.to_dict('records')
        weight = torch.from_numpy(self.train_data['weight'].to_numpy())
        return DataLoader(
            PUMITDataset(data, self.train_transform()),
            conf.num_workers,
            batch_sampler=PUMITDistributedBatchSampler(
                data, conf.num_train_batches, num_skip_batches, self.trans_conf,
                self.world_size, self.global_rank, self.R, conf.train_batch_size, weight,
            ),
            prefetch_factor=8 if conf.num_workers > 0 else None,
            collate_fn=self.collate_fn,
            pin_memory=self.device.type == 'cpu',
            persistent_workers=conf.num_workers > 0,
        )

    def val_transform(self) -> Callable:
        return mt.Compose([
            mt.LoadImageD(DataKey.IMG, PUMITReader, image_only=True),
            UpdateSpacingD(),
            mt.CropForegroundD(DataKey.IMG, DataKey.IMG),
            AdaptivePadD(),
        ])

    def val_dataloader(self):
        conf = self.dl_conf
        data = self.val_data.to_dict('records')
        return DataLoader(
            MONAIDataset(data, self.val_transform()),
            conf.num_workers,
            batch_size=1,
            collate_fn=self.collate_fn,
        )
