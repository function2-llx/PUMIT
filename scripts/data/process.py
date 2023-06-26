from abc import abstractmethod, ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import itertools as it
from pathlib import Path

import cytoolz
import numpy as np
import pandas as pd
import torch
from tqdm.contrib.concurrent import process_map

from monai import transforms as mt
from monai.data import MetaTensor, PydicomReader

DATASETS_ROOT = Path('datasets')
PROCESSED_ROOT = Path('datasets-PUMT')
PROCESSED_ROOT.mkdir(exist_ok=True, parents=True)

@dataclass
class ImageFile:
    key: str
    modality: str | list[str]
    path: Path
    weight: float = 1
    loader: Callable[[Path], MetaTensor] | None = None

class DatasetProcessor(ABC):
    name: str

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name

    @property
    def output_root(self):
        return PROCESSED_ROOT / self.name

    @abstractmethod
    def get_image_files(self) -> Sequence[ImageFile]:
        pass

    # default loader for all image files
    def get_loader(self) -> Callable[[Path], MetaTensor]:
        raise NotImplementedError

    @abstractmethod
    def get_cropper(self) -> Callable[[MetaTensor], MetaTensor]:
        pass

    def process(self, ncols: int = 80, max_workers: int = 16, **kwargs):
        files = self.get_image_files()
        if len(files) == 0:
            return
        if (images_meta_path := self.output_root / 'images-meta.xlsx').exists():
            images_meta = pd.read_excel(images_meta_path, index_col='key')
            remained_mask = images_meta['modality'].isna()
            remained_keys = images_meta.index[remained_mask]
            old_meta = images_meta[~remained_mask]
            files = [file for file in files if file.key in remained_keys]
            print('continue!')
        else:
            old_meta = pd.DataFrame()
            (self.output_root / 'data').mkdir(parents=True)
        if len(files) == 0:
            remained_meta = pd.DataFrame()
        else:
            results = process_map(self.process_file, files, max_workers=max_workers, ncols=ncols, **kwargs)
            remained_meta = pd.DataFrame.from_records(cytoolz.concat(results), index='key')
        pd.concat([old_meta, remained_meta]).to_excel(images_meta_path, freeze_panes=(1, 1))

    def process_file(
        self,
        file: ImageFile,
        cropper: Callable[[MetaTensor], MetaTensor] | None = None,
    ):
        if file.loader is None:
            loader = self.get_loader()
        else:
            loader = file.loader
        try:
            data = loader(file.path)
        except Exception:
            print(file.path)
            return [{'key': file.key, 'origin': file.path}]

        if cropper is None:
            cropper = self.get_cropper()
        ret = self.process_file_data(file, data, cropper)
        for x in ret:
            x['origin'] = file.path
        return ret

    def process_file_data(self, file: ImageFile, data: MetaTensor, cropper: Callable[[MetaTensor], MetaTensor]) -> list[dict]:
        if isinstance(file.modality, str):
            return [self.process_image(data, file.key, file.modality, cropper, file.weight)]
        else:
            assert isinstance(file.modality, list) and len(file.modality) == data.shape[0]
            if len(file.modality) == 1:
                return [self.process_image(data, file.key, file.modality[0], cropper, file.weight)]
            return [
                self.process_image(data[i:i + 1], f'{file.key}-m{i}', modality, cropper, file.weight)
                for i, modality in enumerate(file.modality)
            ]

    def process_image(self, img: MetaTensor, key: str, modality: str | dict, cropper: Callable[[MetaTensor], MetaTensor], weight: float) -> dict:
        cropped = cropper(img)
        save_path = self.output_root / 'data' / f'{key}.npz'
        np.savez(save_path, array=cropped.numpy(), affine=cropped.affine)
        cropped_f = cropped.float()
        return {
            'key': key,
            'modality': modality,
            **{
                f'shape-{i}': s
                for i, s in enumerate(cropped.shape[1:])
            },
            **{
                f'space-{i}': s.item()
                for i, s in enumerate(cropped.pixdim)
            },
            **{
                f'shape-origin-{i}': s
                for i, s in enumerate(img.shape[1:])
            },
            'mean': cropped_f.mean().item(),
            'median': cropped_f.median().item(),
            'std': cropped_f.std().item(),
            'max': cropped_f.max().item(),
            'min': cropped_f.min().item(),
            'p0.5': mt.percentile(cropped_f, 0.5).item(),
            'p99.5': mt.percentile(cropped_f, 99.5).item(),
            'weight': weight,
        }

class Default3DLoaderMixin:
    reader = None

    def get_loader(self) -> Callable[[Path], MetaTensor]:
        return mt.Compose([
            mt.LoadImage(self.reader, image_only=True, dtype=None, ensure_channel_first=True),
            mt.Orientation('RAS'),
            MetaTensor.contiguous,
        ])

class NaturalImageLoaderMixin:
    dummy_dim: int = 1  # it seems that most 2D medical images are taken in the coronal plane
    assert_gray_scale: bool = False

    def __init__(self, dummy_dim: int | None = None):
        if dummy_dim is not None:
            self.dummy_dim = dummy_dim

    def check_and_adapt_to_3d(self, img: MetaTensor):
        if img.shape[0] == 4:
            assert (img[3] == 255).all()
            img = img[:3]
        if self.assert_gray_scale and img.shape[0] != 1:
            assert (img[0] == img[1]).all() and (img[0] == img[2]).all()
            img = img[0:1]
        img = img.unsqueeze(self.dummy_dim + 1)
        img.affine[self.dummy_dim, self.dummy_dim] = 1e8
        return img

    def get_loader(self) -> Callable[[Path], MetaTensor]:
        return mt.Compose([
            mt.LoadImage(image_only=True, dtype=None, ensure_channel_first=True),
            self.check_and_adapt_to_3d,
        ])

    def __call__(self, path: Path):
        return self.get_loader()(path)

class ValueBasedCropper(ABC):
    @abstractmethod
    def get_crop_value(self, img: MetaTensor):
        pass

    def get_cropper(self):
        def select_fn(img: MetaTensor):
            v = torch.as_tensor(self.get_crop_value(img))
            if v.numel() != 1:
                assert v.ndim == 1 and v.shape[0] == img.shape[0]
                for _ in range(img.ndim - 1):
                    v = v[..., None]
            return (img > v).all(dim=0, keepdim=True)

        return mt.CropForeground(select_fn)

class PercentileCropperMixin(ValueBasedCropper):
    min_p: float = 0.5
    exclude_min: bool = True

    def get_crop_value(self, img: MetaTensor):
        ret = torch.empty(img.shape[0])
        for i, c in enumerate(img.float()):
            if self.exclude_min:
                min_v = c.min()
                ret[i] = mt.percentile(c[c > min_v], self.min_p)
            else:
                ret[i] = mt.percentile(c, self.min_p)
            
        return ret

class ConstantCropperMixin(ValueBasedCropper):
    v: float | int = 0
    
    def get_crop_value(self, _img):
        return self.v

# a function f(n) that n*f(n)→1 (n→1), n*f(n)→2 (n→∞)
def adaptive_weight(n: int):
    return (2 * n - 1) / n ** 2

class ACDCProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'ACDC'

    def get_image_files(self):
        return [
            ImageFile(key := patient_dir.name, modality='cine MRI', path=patient_dir / f'{key}_4d.nii.gz')
            for split in ['training', 'testing']
            for patient_dir in (self.dataset_root / split).iterdir() if patient_dir.is_dir()
        ]

    def process_file_data(self, file: ImageFile, data: MetaTensor, cropper: Callable[[MetaTensor], MetaTensor]) -> list[dict]:
        t = data.shape[0]
        # cine MRI results in very similar scans, empirically adjust the weight
        weight = adaptive_weight(t) * file.weight
        return [
            self.process_image(data[i:i + 1], f'{file.key}-{i}', 'MRI', cropper, weight)
            for i in range(t)
        ]

class AMOS22Processor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'AMOS22'

    def get_image_files(self):
        suffix = '.nii.gz'
        return [
            ImageFile(
                key := path.name[:-len(suffix)],
                # `readme.md` says "id numbers less than 500 belong to CT data", but it seems to include 500.
                modality='CT' if int(key.rsplit('_', 1)[1]) <= 500 else 'MRI',
                path=path,
            )
            for path in (self.dataset_root / 'amos22').glob('images*/*.nii.gz')
        ]

class BrainPTM2021Processor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'BrainPTM-2021'

    def process_file_data(self, file: ImageFile, data: MetaTensor, cropper: Callable[[MetaTensor], MetaTensor]) -> list[dict]:
        if file.modality != 'MRI/DWI':
            return super().process_file_data(file, data, cropper)
        d_weight = adaptive_weight(data.shape[0] - 1) * file.weight
        return [
            self.process_image(data[0:1], f'{file.key}-0', file.modality, cropper, weight=file.weight),
            *[
                self.process_image(data[i:i + 1], f'{file.key}-{i}', file.modality, cropper, d_weight)
                for i in range(1, data.shape[0])
            ]
        ]

    def get_image_files(self):
        ret = []
        for case_dir in self.dataset_root.glob('case_*'):
            key = case_dir.name
            ret.append(ImageFile(f'{key}-T1', 'MRI/T1', case_dir / 'T1.nii.gz'))
            ret.append(ImageFile(f'{key}-DWI', 'MRI/DWI', case_dir / 'Diffusion.nii.gz'))
        return ret

class BraTS2023SegmentationProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    def get_image_files(self) -> Sequence[ImageFile]:
        modality_map = {
            't1c': 'MRI/T1c',
            't1n': 'MRI/T1',
            't2f': 'MRI/T2-FLAIR',
            't2w': 'MRI/T2',
        }
        ret = []
        for subject_dir in self.dataset_root.glob('*/*/'):
            key = subject_dir.name
            for modality_suffix, modality in modality_map.items():
                ret.append(ImageFile(f'{key}-{modality_suffix}', modality, subject_dir / f'{key}-{modality_suffix}.nii.gz'))
        return ret

class BraTS2023GLIProcessor(BraTS2023SegmentationProcessor):
    name = 'BraTS2023/BraTS-GLI'

class BraTS2023MENProcessor(BraTS2023SegmentationProcessor):
    name = 'BraTS2023/BraTS-MEN'

class BraTS2023METProcessor(BraTS2023SegmentationProcessor):
    name = 'BraTS2023/BraTS-MET'

class BraTS2023PEDProcessor(BraTS2023SegmentationProcessor):
    name = 'BraTS2023/BraTS-PED'

class BraTS2023SSAProcessor(BraTS2023SegmentationProcessor):
    name = 'BraTS2023/BraTS-SSA'

class BCVProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    def get_image_files(self) -> Sequence[ImageFile]:
        import re
        pattern = re.compile(r'\d+')
        ret = []
        for split in ['Training', 'Testing']:
            for path in (self.dataset_root / 'RawData' / split / 'img').glob('*.nii.gz'):
                key = pattern.search(path.name).group()
                ret.append(ImageFile(key, 'CT', path))
        return ret

class BCVAbdomenProcessor(BCVProcessor):
    name = 'BCV/Abdomen'

class BCVCervixProcessor(BCVProcessor):
    name = 'BCV/Cervix'

class CGMHPelvisProcessor(NaturalImageLoaderMixin, ConstantCropperMixin, DatasetProcessor):
    name = 'CGMH Pelvis'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.stem, 'RGB/XR', path)
            for path in (self.dataset_root / 'CGMH_PelvisSegment' / 'Image').glob('*.png')
        ]

class ChákṣuProcessor(NaturalImageLoaderMixin, ConstantCropperMixin, DatasetProcessor):
    name = 'Chákṣu'
    min_p = 0
    exclude_min = False

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(f'{image_dir.name}.{path.stem}', 'RGB/fundus', path)
            for image_dir in self.dataset_root.glob('*/1.0_Original_Fundus_Images/*') if image_dir.is_dir()
            for path in image_dir.iterdir() if path.suffix.lower() in ['.png', '.jpg']
        ]

class CheXpertProcessor(ConstantCropperMixin, DatasetProcessor):
    name = 'CheXpert'

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'chexpertchestxrays-u20210408'
    
    def get_image_files(self) -> Sequence[ImageFile]:
        ret = []
        dummy_dim_mapping = {
            'frontal': 1,
            'lateral': 0,
        }
        for path in (self.dataset_root / 'CheXpert-v1.0').glob('*/*/*/*.jpg'):
            view = path.stem.split('_')[1]
            ret.append(ImageFile(
                '-'.join([*path.parts[-3:-1], path.stem]),
                'RGB/XR',
                path,
                loader=NaturalImageLoaderMixin(dummy_dim_mapping[view]),
            ))

        return ret

class CHAOSProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'CHAOS'

    def get_image_files(self) -> Sequence[ImageFile]:
        ret = []
        for split in ['Train', 'Test']:
            for case_dir in (self.dataset_root / f'{split}_Sets' / 'CT').iterdir():
                key = case_dir.name
                ret.append(ImageFile(f'{key}-CT', 'CT', case_dir / 'DICOM_anon'))
            for case_dir in (self.dataset_root / f'{split}_Sets' / 'MR').iterdir():
                for phase in ['In', 'Out']:
                    key = case_dir.name
                    ret.append(ImageFile(
                        key=f'{key}-MR-T1DUAL-{phase}',
                        modality=f'MRI/T1-dual/{phase.lower()}',
                        path=case_dir / 'T1DUAL' / 'DICOM_anon' / f'{phase}Phase',
                    ))
                ret.append(ImageFile(f'{key}-MR-T2SPIR', 'MRI/T2-SPIR', case_dir / 'T2SPIR' / 'DICOM_anon'))

        return ret

class CrossMoDA2022Processor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'crossMoDA2022'

    def get_image_files(self) -> Sequence[ImageFile]:
        suffix = '.nii.gz'
        modality_map = {
            'ceT1': 'MRI/T1c',
            'hrT2': 'MRI/T2',
        }
        ret = []
        for folder_name, modality_suffix in [
            ('training_source', 'ceT1'),
            ('training_target', 'hrT2'),
            ('validation', 'hrT2'),
        ]:
            for path in (self.dataset_root / folder_name).glob(f'*_{modality_suffix}{suffix}'):
                key = path.name[:-len(suffix)]
                modality = modality_map[modality_suffix]
                ret.append(ImageFile(f'{key}', modality, path))
        return ret

class CHASEDB1Processor(NaturalImageLoaderMixin, ConstantCropperMixin, DatasetProcessor):
    name = 'CHASE_DB1'
    min_p = 0
    exclude_min = False

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(key := f'{i:02d}{side}', 'RGB/fundus', self.dataset_root / f'Image_{key}.jpg')
            for i, side in it.product(range(1, 15), ('L', 'R'))
        ]

class CHUACProcessor(NaturalImageLoaderMixin, ConstantCropperMixin, DatasetProcessor):
    name = 'CHUAC'
    min_p = 0
    exclude_min = False

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(str(i), 'RGB/fundus', self.dataset_root / 'Original' / f'{i}.png')
            for i in range(1, 31)
        ]

class EPISURGProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'EPISURG'

    def get_image_files(self) -> Sequence[ImageFile]:
        ret = []
        for path in (self.dataset_root / 'subjects').rglob('*t1mri*'):
            key = '_'.join(path.parts[-3:-1])
            ret.append(ImageFile(key, 'MRI/T1', path))
        return ret

class FLARE22Processor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'FLARE22'

    def get_image_files(self):
        image_folders = [
            *(self.dataset_root / 'FLARE2022' / 'Training').glob('FLARE22_UnlabeledCase*'),
            self.dataset_root / 'FLARE2022' / 'Training' / 'FLARE22_LabeledCase50' / 'images',
            self.dataset_root / 'FLARE2022' / 'Validation'
        ]
        suffix = '.nii.gz'
        return [
            ImageFile(key=path.name[:-len(suffix)], modality='CT', path=path)
            for folder in image_folders for path in folder.glob(f'*{suffix}')
        ]

class HaNSegProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'HaN-Seg'

    def get_image_files(self) -> Sequence[ImageFile]:
        modality_map = {
            'CT': 'CT',
            'MR_T1': 'MRI/T1',
        }
        ret = []
        for case_dir in (self.dataset_root / 'HaN-Seg' / 'set_1').iterdir():
            if not case_dir.is_dir():
                continue
            key = case_dir.name
            for modality_suffix, modality in modality_map.items():
                ret.append(ImageFile(f'{key}-{modality_suffix}', modality, case_dir / f'{key}_IMG_{modality_suffix}.nrrd'))
        return ret

class HNSCCProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'HNSCC'
    reader = PydicomReader

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'HNSCC'

    def get_image_files(self) -> Sequence[ImageFile]:
        meta = pd.read_csv(self.dataset_root / 'metadata.csv')
        meta = meta.drop_duplicates(subset='Series UID', keep='last').set_index('Series UID')
        ret = []
        # these are strange series
        ignore_sids = {
            '1.3.6.1.4.1.14519.5.2.1.1706.8040.204861137296365813191219519295',
            '1.3.6.1.4.1.14519.5.2.1.1706.8040.126694397784217103845514279404',
            '1.3.6.1.4.1.14519.5.2.1.1706.8040.128464676098500286122976392857',
            '1.3.6.1.4.1.14519.5.2.1.1706.8040.167120054523886306987571801649',
            '1.3.6.1.4.1.14519.5.2.1.1706.8040.245936214162991561897245946126',
            '1.3.6.1.4.1.14519.5.2.1.1706.8040.275430449343962751502690968943',
        }
        for sid, (num, modality, path) in meta[['Number of Images', 'Modality', 'File Location']].iterrows():
            path = self.dataset_root / path
            if num <= 2 or modality not in ['CT', 'MR'] or 'loc' in path.name.lower():
                continue
            if sid in ignore_sids:
                continue
            if modality == 'MR':
                desc = meta.at[sid, 'Series Description']
                if desc == '3D XRT VOL':
                    continue
                if 'T1 C' in desc or 'POST' in desc:
                    modality = 'T1'
                elif 'T1' in desc:
                    modality = 'T1'
                elif 'T2' in desc:
                    modality = 'T2'
                else:
                    raise ValueError(desc)
                modality = f'MRI/{modality}'
            ret.append(ImageFile(sid, modality, path))
        return ret

def file_sha3(filepath: Path):
    sha3_hash = hashlib.sha3_256()

    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha3_hash.update(byte_block)

    return sha3_hash.hexdigest()

class IDRiDProcessor(NaturalImageLoaderMixin, ConstantCropperMixin, DatasetProcessor):
    name = 'IDRiD'

    def get_image_files(self) -> Sequence[ImageFile]:
        ret = []
        file_sha3s = set()
        for task in ['A. Segmentation', 'B. Disease Grading']:
            for path in (self.dataset_root / task / '1. Original Images').glob('*/*.jpg'):
                h = file_sha3(path)
                if h in file_sha3s:
                    continue
                file_sha3s.add(h)
                key = f'{task[0]}.{path.stem}'
                ret.append(ImageFile(key, 'RGB/fundus', path))
        return ret

class IXIProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'IXI'

    def get_image_files(self) -> Sequence[ImageFile]:
        suffix = '.nii.gz'
        ret = [
            ImageFile(path.name[:-len(suffix)], f'MRI/{modality}', path)
            for modality in ['T1', 'T2', 'PD', 'MRA']
            for path in (self.dataset_root / f'IXI-{modality}').glob(f'*{suffix}')
            # incomplete file
            if path.name != 'IXI371-IOP-0970-MRA.nii.gz'
        ]
        dti_groups: dict[str, list[tuple[int, Path]]] = {}
        for path in (self.dataset_root / 'IXI-DTI').glob(f'*{suffix}'):
            group, idx = path.name[:-len(suffix)].rsplit('-', 1)
            dti_groups.setdefault(group, []).append((int(idx), path))
        for group, items in dti_groups.items():
            all_idxes = [x[0] for x in items]
            if (min_idx := min(all_idxes)) != 0:
                print('missing zero:', group)
            if max(all_idxes) - min_idx + 1 != len(items):
                print('missing some diffusion:', group)
            d_weight = adaptive_weight(len(items) - 1)
            ret.extend([
                # not using `group` for key to keep the original formatting
                ImageFile(path.name[:-len(suffix)], 'MRI/DTI', path, weight=1 if i == 0 else d_weight)
                for i, path in items
            ])

        return ret

class KaggleRDCProcessor(NaturalImageLoaderMixin, ConstantCropperMixin, DatasetProcessor):
    name = 'Kaggle-RDC'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(f'{path.parent.name}-{path.stem}', 'RGB/fundus', path)
            for path in self.dataset_root.rglob('*.png')
        ]

class LIDCIDRIProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'LIDC-IDRI'
    reader = PydicomReader

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'TCIA_LIDC-IDRI_20200921'

    def get_image_files(self) -> Sequence[ImageFile]:
        meta = pd.read_csv(self.dataset_root / 'metadata.csv')
        meta = meta[meta['Modality'] == 'CT']
        return [
            ImageFile('_'.join(Path(path).parts[-3:-1]), 'CT', self.dataset_root / path)
            for path in meta['File Location']
        ]

class MRSpineSegProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'MRSpineSeg_Challenge_SMU'

    def get_image_files(self) -> Sequence[ImageFile]:
        suffix = '.nii.gz'
        return [
            ImageFile(path.name[:-len(suffix)], 'MRI/T2', path)
            for path in self.dataset_root.rglob(f'*{suffix}')
        ]

class MSDProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    def get_modality(self) -> list[str]:
        import json
        meta = json.loads(Path(self.dataset_root / 'dataset.json').read_bytes())
        modality = meta['modality']
        return [modality[str(i)] for i in range(len(modality))]

    def get_image_files(self) -> Sequence[ImageFile]:
        modality = self.get_modality()
        suffix = '.nii.gz'
        return [
            ImageFile(path.name[:-len(suffix)], modality, path)
            for path in self.dataset_root.glob(f'images*/*{suffix}')
        ]

class MSDBrainTumourProcessor(MSDProcessor):
    name = 'MSD/Task01_BrainTumour'

    def get_modality(self) -> list[str]:
        mapping = {
            'FLAIR': 'T2-FLAIR',
            'T1w': 'T1',
            't1gd': 'T1c',
            'T2w': 'T2'
        }
        return [f'MRI/{mapping[m]}' for m in super().get_modality()]

class MSDHeartProcessor(MSDProcessor):
    name = 'MSD/Task02_Heart'

class MSDLiverProcessor(MSDProcessor):
    name = 'MSD/Task03_Liver'

class MSDHippocampusProcessor(MSDProcessor):
    name = 'MSD/Task04_Hippocampus'

    def get_modality(self) -> list[str]:
        # according to https://arxiv.org/pdf/1902.09063.pdf Methods.Datasets.Task04_Hippocampus
        return ['MRI/T1']

class MSDProstateProcessor(MSDProcessor):
    name = 'MSD/Task05_Prostate'

    def get_modality(self) -> list[str]:
        return [f'MRI/{m}' for m in super().get_modality()]

class MSDLungProcessor(MSDProcessor):
    name = 'MSD/Task06_Lung'

class MSDPancreasProcessor(MSDProcessor):
    name = 'MSD/Task07_Pancreas'

class MSDHepaticVesselProcessor(MSDProcessor):
    name = 'MSD/Task08_HepaticVessel'

class MSDSpleenProcessor(MSDProcessor):
    name = 'MSD/Task09_Spleen'

class MSDColonProcessor(MSDProcessor):
    name = 'MSD/Task10_Colon'

class NCTCTProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'NCTCT'
    reader = PydicomReader

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'TCIA_CT_COLONOGRAPHY_06-22-2015'

    def get_image_files(self) -> Sequence[ImageFile]:
        meta = pd.read_csv(self.dataset_root / 'metadata.csv')
        meta = meta.drop_duplicates(subset='Series UID', keep='last').set_index('Series UID')
        return [
            ImageFile(sid, 'CT', self.dataset_root / path)
            for sid, (num, path) in meta[['Number of Images', 'File Location']].iterrows()
            if num >= 81
        ]

class NIHChestXRayProcessor(NaturalImageLoaderMixin, ConstantCropperMixin, DatasetProcessor):
    name = 'NIH Chest X-ray'
    assert_gray_scale = True

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'CXR8'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.stem, 'RGB/XR', path)
            for path in (self.dataset_root / 'images' / 'images').glob('*.png')
        ]

class PelviXNetProcessor(NaturalImageLoaderMixin, ConstantCropperMixin, DatasetProcessor):
    name = 'PelviXNet'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.name, 'RGB/XR', path)
            for path in (self.dataset_root / 'pxr-150' / 'images_all').glob('*.png')
        ]

class PICAIProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'PI-CAI'

    def get_image_files(self) -> Sequence[ImageFile]:
        mapping = {
            'adc': 'ADC',
            'cor': 'T2',
            't2w': 'T2',
            'sag': 'T2',
            'hbv': 'DWI',
        }
        return [
            ImageFile(key := path.stem, mapping[key.rsplit('_', 1)[1]], path)
            for path in (self.dataset_root / 'public_images').rglob('*.mha')
        ]

class Prostate158Processor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'Prostate158'

    def get_image_files(self) -> Sequence[ImageFile]:
        mapping = {
            'adc': 'ADC',
            'dwi': 'DWI',
            't2': 'T2',
        }
        return [
            ImageFile(f'{case_dir.name}-{modality_suffix}', f'MRI/{modality}', case_dir / f'{modality_suffix}.nii.gz')
            for case_dir in (self.dataset_root / 'prostate158_train' / 'train').iterdir()
            for modality_suffix, modality in mapping.items()
        ]

class PROSTATEMRIProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'PROSTATE-MRI'
    reader = PydicomReader

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'PROSTATE-MRI-5-18-2018-doiJNLP-dKJJAqnS'

    def get_image_files(self) -> Sequence[ImageFile]:
        mapping = {
            'DCE': 'T1c',
            'SShDWI': 'DWI',
            'dSShDWI': 'DWI',
        }
        ret = []
        for path in (self.dataset_root / 'PROSTATE-MRI').glob('*/*/*'):
            key_suffix, modality_string = path.name.split('-')[:2]
            modality = modality_string.split(' ', 1)[0]
            ret.append(ImageFile(
                f'{path.parent.parent.name}_{key_suffix}',
                f'MRI/{mapping.get(modality, modality)}',
                path,
            ))
        return ret

class RibFracProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'RibFrac'

    def get_image_files(self) -> Sequence[ImageFile]:
        suffix = '-image.nii.gz'
        return [
            ImageFile(path.name[:-len(suffix)], 'CT', path)
            for path in self.dataset_root.glob(f'*/*{suffix}')
        ]

class RSNA2020PEDProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'RSNA-2020-PED'
    reader = PydicomReader

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile('.'.join(path.parts[-2:]), 'CT', path)
            for path in self.dataset_root.glob('*/*/*')
        ]

class RSNA2022CSFDProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'RSNA-2022-CSFD'
    reader = PydicomReader

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.name, 'CT', path)
            for split in ['train', 'test']
            for path in (self.dataset_root / f'{split}_images').iterdir()
        ]

class STOICProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'STOIC'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.stem, 'CT', path)
            for path in (self.dataset_root / 'stoic2021-training' / 'data' / 'mha').glob('*.mha')
        ]

class TotalSegmentatorProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'TotalSegmentator'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(key, 'CT', path)
            for path in (self.dataset_root / 'Totalsegmentator_dataset').glob('*/ct.nii.gz')
            # this case seems to be corrupted
            # https://github.com/wasserth/TotalSegmentator/issues/24#issuecomment-1298390124
            if (key := path.parent.name) != 's0864'
        ]

class VerSeProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'VerSe'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.parent.name, 'CT', path)
            for path in self.dataset_root.glob('*/rawdata/*/*.nii.gz')
        ]

class VSSEGProcessor(Default3DLoaderMixin, PercentileCropperMixin, DatasetProcessor):
    name = 'Vestibular-Schwannoma-SEG'
    reader = PydicomReader

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'Vestibular-Schwannoma-SEG Feb 2021 manifest'

    def get_image_files(self) -> Sequence[ImageFile]:
        ret = []
        for path in (self.dataset_root / 'Vestibular-Schwannoma-SEG').glob('*/*/*'):
            if '-' not in path.name:
                continue
            seq_name = path.name.split('-')[1]
            if 't1' in seq_name:
                modality = 'T1'
            elif 't2' in seq_name:
                modality = 'T2'
            else:
                raise ValueError
            ret.append(ImageFile(f'{path.parent.parent.name}_{modality}', f'MRI/{modality}', path))
        return ret

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('datasets', nargs='*', type=str)
    parser.add_argument('--ie', action='store_true')
    args = parser.parse_args()
    for dataset in args.datasets:
        processor_cls: type[DatasetProcessor] | None = globals().get(f'{dataset}Processor', None)
        if processor_cls is None:
            print(f'no processor for {dataset}')
        else:
            processor = processor_cls()
            print(dataset)
            if args.ie:
                try:
                    processor.process()
                except Exception:
                    import traceback
                    print(traceback.format_exc())
            else:
                processor.process()

if __name__ == '__main__':
    main()
