from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from lightning import LightningDataModule
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch.utils.data import Dataset as TorchDataset, Sampler
from tqdm import tqdm

from luolib import transforms as lt
from luolib.types import tuple2_t
from monai.config import PathLike
from monai.data import Dataset as MONAIDataset, DataLoader
from monai import transforms as mt

from pumit.reader import PUMITReader
from pumit.transforms import AdaptivePadD, InputTransformD, PUMITLoader

DATA_ROOT = Path('processed-data')

@dataclass(kw_only=True)
class DataLoaderConf:
    train_batch_size: int | None = None
    val_batch_size: int = 1  # default=1, or help me write another distributed batch sampler for validation
    num_train_batches: int | None = None
    num_workers: int = 8

@dataclass(kw_only=True)
class TransformConf:
    # spatial
    base_size_z: int = 128
    size_xy: int = 128
    rotate_p: float = 0.25
    scale_z: tuple2_t[float] = (3 / 4, 4 / 3)
    scale_z_p: float = 0.25
    scale_xy: tuple2_t[float] = (0.75, 2)
    scale_xy_p: float = 0.5
    # intensity
    scale_intensity: float = 0.25
    scale_intensity_p: float = 0.15
    shift_intensity: float = 0.1
    shift_intensity_p: float = 0.15

    @dataclass
    class AdjustContrast:
        prob: float = 0.15
        range: tuple2_t[float] = (0.75, 1.25)
        preserve_intensity_range: bool = True
    adjust_contrast: AdjustContrast = field(default_factory=AdjustContrast)

    @dataclass
    class GammaCorrection:
        prob: float = 0.3
        range: tuple2_t[float] = (0.7, 1.5)
        prob_invert: float = 0.25
    gamma_correction: GammaCorrection = field(default_factory=GammaCorrection)

def gen_trans_info(data: dict, trans_conf: TransformConf, R: np.random.RandomState) -> dict:
    shape = np.array(data['shape'])
    origin_size_xy = shape[1:].min().item()
    spacing = data['spacing']
    spacing_z = spacing[0]
    spacing_xy = spacing[1:].min()

    # down_i: how much to downsample along axis i
    if R.uniform() < trans_conf.scale_xy_p:
        scale_xy = R.uniform(
            trans_conf.scale_xy[0],
            min(origin_size_xy / trans_conf.size_xy, trans_conf.scale_xy[1]),
        )
    else:
        scale_xy = 1.
    size_xy = trans_conf.size_xy

    if spacing_z <= 3 * spacing_xy and R.uniform() < trans_conf.scale_z_p:
        scale_z = R.uniform(*trans_conf.scale_z)
    else:
        scale_z = 1.
    # ratio of spacing z / spacing x after scaling
    ratio = np.clip(spacing_z * scale_z / (spacing_xy * scale_xy), 1, 1 << 6)
    aniso_d = int(ratio).bit_length() - 1
    if aniso_d == 6:
        # extremely anisotropic data, then treat it as 2D
        size_z = 1
    else:
        # make sure base_size_z is not smaller than 32
        size_z = trans_conf.base_size_z >> aniso_d

    return {
        'aniso_d': aniso_d,
        'scale': np.array((scale_z, scale_xy, scale_xy)),
        'size': np.array((size_z, size_xy, size_xy)),
    }

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
        trans_conf = self.trans_conf
        while remain_batches > 0:
            for i in self.weight.multinomial(self.buffer_size, replacement=True).tolist():
                data = self.data[i]
                trans_info = gen_trans_info(data, trans_conf, self.R)
                bucket.setdefault(aniso_d := trans_info['aniso_d'], []).append((i, trans_info))
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
        # dataset_weights: dict[str, float],
        dl_conf: DataLoaderConf,
        trans_conf: TransformConf,
        seed: int | None = 42,
        device: Literal['cpu', 'cuda'] = 'cpu',
    ):
        super().__init__()
        self.train_data = pd.DataFrame()
        self.val_data = pd.DataFrame()
        self.R = np.random.RandomState(seed)
        # dataset_names = []
        dataset_info = {}
        for dataset_dir in DATA_ROOT.iterdir():
            dataset_name = dataset_dir.name
            if (meta_path := dataset_dir / 'meta.pkl').exists():
                meta = pd.read_pickle(meta_path)
                is_2d = (meta['shape'].map(lambda shape: shape[0]).to_numpy() == 1).all()
                dataset_info[dataset_name] = {
                    'dims': 2 if is_2d else 3,
                    # take sqrt to balance between datasets
                    'weights': (weights := meta['weight'].sum()),
                    'sqrt-weights': weights.sqrt(),
                }
            else:
                print(f'skip {dataset_name}')
        # sort to be deterministic
        dataset_info = pd.DataFrame.from_dict(dataset_info, orient='index').sort_index()
        weights_2d = dataset_info[dataset_info['dims'] == 2]['sqrt-weights'].sum()
        weights_3d = dataset_info[dataset_info['dims'] == 3]['sqrt-weights'].sum()
        # make the ratio of 2D:3D=1:3
        weights_ratio = 3 * weights_2d / weights_3d
        for dataset_name in dataset_info.index:
            dataset_dir = DATA_ROOT / dataset_name
            meta: pd.DataFrame = pd.read_pickle(dataset_dir / 'meta.pkl')
            dataset_weight_scale = 1 / dataset_info[dataset_name]['weights']
            if dataset_info[dataset_name]['dims'] == 3:
                dataset_weight_scale *= weights_ratio
            dataset_info['scale'] = dataset_weight_scale
            meta['weight'] *= dataset_weight_scale
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
        conf = self.trans_conf
        return mt.Compose(
            [
                PUMITLoader(conf.rotate_p, self.device),
                mt.RandScaleIntensityD('img', conf.scale_intensity, prob=conf.scale_intensity_p, channel_wise=True),
                mt.RandShiftIntensityD('img', conf.shift_intensity, prob=conf.shift_intensity_p, channel_wise=True),
                lt.RandDictWrapper(
                    'img',
                    lt.RandAdjustContrast(
                        conf.adjust_contrast.prob,
                        conf.adjust_contrast.range,
                        conf.adjust_contrast.preserve_intensity_range,
                    ),
                ),
                lt.ClampIntensityD('img'),
                lt.RandDictWrapper(
                    'img',
                    lt.RandGammaCorrection(
                        conf.gamma_correction.prob,
                        conf.gamma_correction.range,
                        conf.gamma_correction.prob_invert,
                        False,
                        False,
                    ),
                ),
                InputTransformD(),
            ],
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
            mt.LoadImageD('img', PUMITReader, image_only=True),
            mt.CropForegroundD('img', 'img'),
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
