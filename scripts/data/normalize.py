from collections.abc import Hashable, Mapping
import itertools as it
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch.cuda
from tqdm.contrib.concurrent import process_map

from luolib.utils import DataKey
from monai import transforms as mt
from monai.data import MetaTensor
from monai.transforms import ToTensorD
from pumt.reader import PUMTReader

class NormalizeIntensityD(mt.Transform):
    def __call__(self, data: Mapping[Hashable, ...]):
        data = dict(data)
        img: MetaTensor = data[DataKey.IMG]
        modality: str = data['modality']

        if modality.startswith('RGB') or modality.startswith('gray'):
            normalizer = mt.ScaleIntensityRange(0., 255., 0., 1.)
        else:
            normalizer = mt.ScaleIntensityRange(data['p0.5'], data['p99.5'], 0., 1., clip=True)
        data[DataKey.IMG] = normalizer(img)
        return data

src_dir = Path('datasets-PUMT')
save_dir = Path('PUMT-normalized')

loader = mt.LoadImageD(DataKey.IMG, PUMTReader, image_only=True)
normalizer = NormalizeIntensityD()

def process(meta: dict, dataset_name: str, device_id: int):
    data = loader(meta)
    converter = ToTensorD(DataKey.IMG, device=f'cuda:{device_id}')
    data = converter(data)
    data = normalizer(data)
    img: MetaTensor = data[DataKey.IMG]
    np.savez(save_dir / dataset_name / 'data' / f'{meta["key"]}.npz', array=img.cpu().numpy(), affine=img.affine)

def main():
    for dataset_dir in src_dir.iterdir():
        dataset_name = dataset_dir.name
        meta = pd.read_csv(dataset_dir / 'images-meta.csv', dtype={'key': 'string'})
        meta[DataKey.IMG] = meta['key'].map(lambda key: dataset_dir / 'data' / f'{key}.npz')
        (save_dir / dataset_name / 'data').mkdir(parents=True)
        shutil.copy2(dataset_dir / 'images-meta.csv', save_dir / dataset_name / 'images-meta.csv')
        process_map(
            process,
            meta.to_dict('records'),
            it.repeat(dataset_name),
            it.cycle(range(torch.cuda.device_count())),
            desc=dataset_name,
            max_workers=8,
            ncols=80,
        )

if __name__ == '__main__':
    main()
