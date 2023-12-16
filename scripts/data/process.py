import hashlib
import inspect
import itertools as it
from abc import abstractmethod, ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import cytoolz
import numpy as np
import pandas as pd
import torch
from torchvision import transforms as tvt

from luolib import transforms as lt
from luolib.types import tuple3_t
from luolib.utils import SavedSet, concat_drop_dup, get_cuda_device, process_map
from monai import transforms as mt
from monai.data import MetaTensor, PydicomReader

DATASETS_ROOT = Path('datasets')
PROCESSED_ROOT = Path('processed-data')
PROCESSED_ROOT.mkdir(exist_ok=True, parents=True)

MAX_SMALLER_EDGE = 512

@dataclass
class ImageFile:
    key: str
    modality: str | list[str]
    path: Path
    weight: float = 1

def is_natural_modality(modality: str) -> bool:
    return modality.startswith('RGB') or modality.startswith('gray')

class DatasetProcessor(ABC):
    name: str
    max_workers: int | None = None
    chunksize: int | None = None
    empty_cache: bool = False
    orientation: str | None = None
    """if orientation is None, will determine it from the spacing"""

    def update_multiprocessing(self, max_workers: int, chunksize: int, override: bool):
        if self.max_workers is None or override:
            self.max_workers = max_workers
        if self.chunksize is None or override:
            self.chunksize = chunksize

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name

    @property
    def output_name(self) -> str:
        return self.name

    @property
    def output_root(self):
        return PROCESSED_ROOT / self.output_name

    @abstractmethod
    def get_image_files(self) -> Sequence[ImageFile]:
        pass

    def get_loader(self, device: torch.device) -> Callable[[Path], MetaTensor]:
        """default loader for all image files"""
        raise NotImplementedError

    def process(self):
        files = self.get_image_files()
        assert len(files) == len(set(file.key for file in files)), 'duplicate keys?'
        assert len(files) > 0, 'no files?'
        processed = SavedSet(self.output_root / '.processed.txt')
        files = [file for file in files if file.key not in processed]
        if len(files) == 0:
            return
        print(f'{len(files)} files remaining')
        (self.output_root / 'data').mkdir(parents=True, exist_ok=True)
        results, status = zip(
            *process_map(
                self.process_file,
                files,
                max_workers=self.max_workers,
                chunksize=self.chunksize,
                ncols=80,
            ),
        )
        results = [*cytoolz.concat(results)]
        if len(results) == 0:
            return
        info, meta = zip(*results)
        info = pd.DataFrame.from_records(info, index='key')
        meta = pd.DataFrame.from_records(meta, index='key')
        info_path = self.output_root / 'info.csv'
        meta_path = self.output_root / 'meta.pkl'
        if info_path.exists():
            info = concat_drop_dup([pd.read_csv(info_path, index_col='key', dtype={'key': 'string'}), info])
            meta = concat_drop_dup([pd.read_pickle(meta_path), meta])
        info.to_csv(info_path)
        info.to_excel(info_path.with_suffix('.xlsx'), freeze_panes=(1, 1))
        meta.to_pickle(meta_path)
        processed.save_list([file.key for file, ok in zip(files, status) if ok])

    def process_file(self, file: ImageFile):
        """Load the data from file and process"""
        device = get_cuda_device()
        loader = self.get_loader(device)
        try:
            data = loader(file.path)
            ret = self.process_file_data(file, data)
            for info, meta in ret:
                info['origin'] = file.path
            return ret, True
        except Exception as e:
            print(file.path)
            print(e)
            return [], False
        finally:
            if self.empty_cache:
                torch.cuda.empty_cache()

    def process_file_data(self, file: ImageFile, data: MetaTensor) -> list[tuple[dict, dict]]:
        """Handle cases where a file contains various numbers of imaging modalities"""
        if isinstance(file.modality, str):
            return [self.process_image(data, file.key, file.modality, file.weight)]
        else:
            assert isinstance(file.modality, list) and len(file.modality) == data.shape[0]
            if len(file.modality) == 1:
                return [self.process_image(data, file.key, file.modality[0], file.weight)]
            return [
                self.process_image(data[i:i + 1], f'{file.key}-m{i}', modality, file.weight)
                for i, modality in enumerate(file.modality)
            ]

    def adjust_orientation(self, img: MetaTensor):
        if self.orientation is not None:
            return mt.Orientation(self.orientation)(img)
        # TODO: for isotropic image, perhaps create multiple instances, but it may also be handled in data augmentation
        if abs(img.pixdim[1] - img.pixdim[2]) > 1e-2:
            codes = ['RAS', 'ASR', 'SRA']
            diff = np.empty(len(codes))
            for i, code in enumerate(codes):
                orientation = mt.Orientation(code)
                img_o: MetaTensor = orientation(img)  # type: ignore
                diff[i] = abs(img_o.pixdim[1] - img_o.pixdim[2])
            code = codes[diff.argmin()]
            return mt.Orientation(code)(img)
        else:
            return img

    def normalize_image(self, img: MetaTensor, modality: str) -> tuple[MetaTensor, tuple3_t[int]]:
        """
        1. clip intensity for non-natural images
        2. crop foreground by
            - 0 for natural images
            - minimum for non-natural images
        3. scale intensity to [0, 1]
        Returns:
            (cropped & scaled image, original_shape)
        """
        img = img.float()
        # 1. clip intensity
        if is_natural_modality(modality):
            minv = 0
        else:
            minv = lt.quantile(img, 0.23 / 100)
            maxv = lt.quantile(img, 99.73 / 100)
            img[img < minv] = minv
            img[img > maxv] = maxv
        # 2. crop foreground
        original_shape = img.shape[1:]
        # make MONAI happy about the deprecated default
        cropper = mt.CropForeground(lambda x: x > minv, allow_smaller=True)
        cropped = cropper(img)
        # 3. scale intensity to [0, 1]
        if is_natural_modality(modality):
            minv = 0
            maxv = 255
        else:
            minv = cropped.min().item()
            maxv = cropped.max().item()
        return (cropped - minv) / (maxv - minv), original_shape

    def process_image(self, img: MetaTensor, key: str, modality: str | dict, weight: float) -> tuple[dict, dict]:
        """
        Returns:
            - human-readable information that will be stored in `info.csv`
            - metadata
        """
        img = self.adjust_orientation(img).contiguous()
        cropped, original_shape = self.normalize_image(img, modality)
        if (r := MAX_SMALLER_EDGE / min(cropped.shape[2:])) < 1:
            resizer = mt.Resize((-1, round(cropped.shape[2] * r), round(cropped.shape[3] * r)), anti_aliasing=True)
            cropped = resizer(cropped)
        spacing = cropped.pixdim.numpy()
        info = {
            'key': key,
            'modality': modality,
            **{
                f'shape-{i}': s
                for i, s in enumerate(cropped.shape[1:])
            },
            **{
                f'space-{i}': s.item()
                for i, s in enumerate(spacing)
            },
            **{
                f'shape-origin-{i}': s
                for i, s in enumerate(original_shape)
            },
            'weight': weight,
        }
        meta = {
            'key': key,
            'modality': modality,
            'spacing': spacing,
            'shape': tuple(cropped.shape[1:]),
            'weight': weight,
        }
        if not is_natural_modality(modality):
            meta['mean'] = cropped.mean(dim=(1, 2, 3)).item()
            meta['std'] = cropped.std(dim=(1, 2, 3)).item()

        np.save(self.output_root / 'data' / f'{key}.npy', cropped.cpu().numpy().astype(np.float16))
        return info, meta

class Default3DLoaderMixin:
    reader = None

    def get_loader(self, device: torch.device) -> Callable[[Path], MetaTensor]:
        return mt.Compose([
            mt.LoadImage(self.reader, image_only=True, dtype=None, ensure_channel_first=True),
            mt.ToDevice(device),
        ])

class NaturalImageLoaderMixin:
    assert_gray_scale: bool = False

    def __init__(self):
        self.resize = tvt.Resize(MAX_SMALLER_EDGE, antialias=True)
        self.crop = mt.CropForeground(allow_smaller=False)

    def check_and_adapt_to_3d(self, img: MetaTensor):
        if img.shape[0] == 4:
            assert (img[3] == 255).all()
            img = img[:3]
        if self.assert_gray_scale and img.shape[0] != 1:
            assert (img[0] == img[1]).all() and (img[0] == img[2]).all()
            img = img[0:1]
        img = img.unsqueeze(1)
        img.affine[0, 0] = 1e8
        return img

    def load(self, path: Path, device: torch.device):
        """this function will crop & resize natural images in advance"""
        from torchvision.io import read_image
        img = read_image(str(path)).to(device)
        img = self.crop(img)
        if min(img.shape[1:]) > MAX_SMALLER_EDGE:
            img = self.resize(img.div(255)).mul(255)
        img = MetaTensor(img.byte())
        return img

    def get_loader(self, device: torch.device) -> Callable[[Path], MetaTensor]:
        return mt.Compose([
            cytoolz.partial(self.load, device=device),
            self.check_and_adapt_to_3d,
        ])

def adaptive_weight(n: int):
    """a function f(n) that n*f(n)→1 (n→1), n*f(n)→2 (n→∞)"""
    return (2 * n - 1) / n ** 2

class ACDCProcessor(Default3DLoaderMixin, DatasetProcessor):
    name = 'ACDC'

    def get_image_files(self):
        return [
            ImageFile(key := patient_dir.name, modality='cine MRI', path=patient_dir / f'{key}_4d.nii.gz')
            for split in ['training', 'testing']
            for patient_dir in (self.dataset_root / split).iterdir() if patient_dir.is_dir()
        ]

    def process_file_data(self, file: ImageFile, data: MetaTensor) -> list[tuple[dict, dict]]:
        t = data.shape[0]
        # cine MRI results in very similar scans, empirically adjust the weight
        weight = adaptive_weight(t) * file.weight
        return [
            self.process_image(data[i:i + 1], f'{file.key}-{i}', 'MRI', weight)
            for i in range(t)
        ]

class AMOS22Processor(Default3DLoaderMixin, DatasetProcessor):
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

class BrainPTM2021Processor(Default3DLoaderMixin, DatasetProcessor):
    name = 'BrainPTM-2021'
    orientation = 'ASR'

    def process_file_data(self, file: ImageFile, data: MetaTensor) -> list[tuple[dict, dict]]:
        if file.modality != 'MRI/DWI':
            return super().process_file_data(file, data)
        d_weight = adaptive_weight(data.shape[0] - 1) * file.weight
        return [
            self.process_image(data[0:1], f'{file.key}-0', file.modality, weight=file.weight),
            *[
                self.process_image(data[i:i + 1], f'{file.key}-{i}', file.modality, d_weight)
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

class BraTS2023SegmentationProcessor(Default3DLoaderMixin, DatasetProcessor):
    orientation = 'SAR'

    @property
    def output_name(self):
        return f"BraTS2023-{self.name.split('-')[-1]}"

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

class BCVProcessor(Default3DLoaderMixin, DatasetProcessor):
    @property
    def output_name(self):
        return self.name.replace('/', '-')

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

class BUSIProcessor(NaturalImageLoaderMixin, DatasetProcessor):
    name = 'BUSI'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(key, 'RGB/US', path)
            for path in (self.dataset_root / 'Dataset_BUSI_with_GT').glob('*/*.png')
            if 'mask' not in (key := path.stem)
        ]

class CGMHPelvisProcessor(NaturalImageLoaderMixin, DatasetProcessor):
    name = 'CGMH Pelvis'
    assert_gray_scale = True

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.stem, 'gray/XR', path)
            for path in (self.dataset_root / 'CGMH_PelvisSegment' / 'Image').glob('*.png')
        ]

class ChákṣuProcessor(NaturalImageLoaderMixin, DatasetProcessor):
    name = 'Chákṣu'
    min_p = 0
    exclude_min = False

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(f'{image_dir.name}.{path.stem}', 'RGB/fundus', path)
            for image_dir in self.dataset_root.glob('*/1.0_Original_Fundus_Images/*') if image_dir.is_dir()
            for path in image_dir.iterdir() if path.suffix.lower() in ['.png', '.jpg']
        ]

class CheXpertProcessor(NaturalImageLoaderMixin, DatasetProcessor):
    name = 'CheXpert'
    assert_gray_scale: bool = True
    max_workers = 64
    chunksize = 10

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'chexpertchestxrays-u20210408' / 'CheXpert-v1.0'

    @staticmethod
    def get_patient_image_files(patient_dir: Path):
        return [
            ImageFile(
                '-'.join([*path.parts[-3:-1], path.stem]),
                'gray/XR',
                path,
            )
            for study_dir in patient_dir.iterdir()
            for path in study_dir.glob('*.jpg')
        ]

    def get_image_files(self) -> Sequence[ImageFile]:
        patient_dirs = []
        for split in ['train', 'valid', 'test']:
            patient_dirs.extend([*(self.dataset_root / split).iterdir()])
        ret: list[list[ImageFile]] = process_map(
            self.get_patient_image_files, patient_dirs,
            max_workers=self.max_workers, desc='get image files', ncols=80, chunksize=10,
        )

        return [*cytoolz.concat(ret)]

class CHAOSProcessor(Default3DLoaderMixin, DatasetProcessor):
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

class CrossMoDA2022Processor(Default3DLoaderMixin, DatasetProcessor):
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

class CHASEDB1Processor(NaturalImageLoaderMixin, DatasetProcessor):
    name = 'CHASE_DB1'
    min_p = 0
    exclude_min = False

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(key := f'{i:02d}{side}', 'RGB/fundus', self.dataset_root / f'Image_{key}.jpg')
            for i, side in it.product(range(1, 15), ('L', 'R'))
        ]

class CHUACProcessor(NaturalImageLoaderMixin, DatasetProcessor):
    name = 'CHUAC'
    min_p = 0
    exclude_min = False

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(str(i), 'RGB/fundus', self.dataset_root / 'Original' / f'{i}.png')
            for i in range(1, 31)
        ]

class EPISURGProcessor(Default3DLoaderMixin, DatasetProcessor):
    name = 'EPISURG'

    def get_image_files(self) -> Sequence[ImageFile]:
        ret = []
        for path in (self.dataset_root / 'subjects').rglob('*t1mri*'):
            key = '_'.join(path.parts[-3:-1])
            ret.append(ImageFile(key, 'MRI/T1', path))
        return ret

class FLARE22Processor(Default3DLoaderMixin, DatasetProcessor):
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

class HaNSegProcessor(Default3DLoaderMixin, DatasetProcessor):
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

class TCIAProcessor(DatasetProcessor, ABC):
    @cached_property
    def tcia_meta(self):
        meta = pd.read_csv(self.dataset_root / 'metadata.csv')
        return meta.drop_duplicates(subset='Series UID', keep='last').set_index('Series UID')

class HNSCCProcessor(Default3DLoaderMixin, TCIAProcessor):
    name = 'HNSCC'
    reader = PydicomReader

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'HNSCC'

    def get_image_files(self) -> Sequence[ImageFile]:
        meta = self.tcia_meta
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

class HC18Processor(NaturalImageLoaderMixin, DatasetProcessor):
    name = 'HC18'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(f'{split}-{path.stem}', 'RGB/US', path)
            for split in ['training', 'test']
            for path in (self.dataset_root / f'{split}_set').glob('*HC.png')
        ]

class IChallengeProcessor(NaturalImageLoaderMixin, DatasetProcessor, ABC):
    @property
    def output_name(self):
        return self.name.replace('/', '-')

class IChallengeADAMProcessor(IChallengeProcessor):
    name = 'iChallenge/ADAM'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.stem, 'RGB/fundus', path)
            for data_dir in [
                'Test/Test-image-400',
                'Train/Training-image-400',
                'Validation/image',
            ]
            for path in (self.dataset_root / data_dir).rglob('*.jpg')
        ]

class IChallengeGAMMAProcessor(IChallengeProcessor):
    name = 'iChallenge/GAMMA'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.stem, 'RGB/fundus', path)
            for path in self.dataset_root.glob('*/multi-modality_images/*/*.jpg')
        ]

class IChallengePALMProcessor(IChallengeProcessor):
    name = 'iChallenge/PALM'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.stem, 'RGB/fundus', path)
            for data_dir in [
                'Train/PALM-Training400',
                'Validation/Validation-400',
                'Test/PALM-Testing400-Images',
            ]
            for path in (self.dataset_root / data_dir).glob('*.jpg')
        ]

class IChallengeREFUGE2Processor(IChallengeProcessor):
    name = 'iChallenge/REFUGE2'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(f'R1-{path.stem}' if data_dir.startswith('Train') else path.stem, 'RGB/fundus', path)
            for data_dir in [
                'Train/REFUGE1-train/Training400',
                'Train/REFUGE1-val/REFUGE-Validation400',
                'Train/REFUGE1-test/Test400',
                'Validation/Images',
                'Test/refuge2-test',
            ]
            for path in (self.dataset_root / data_dir).rglob('*.jpg')
        ]

def file_sha3(filepath: Path):
    sha3_hash = hashlib.sha3_256()

    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha3_hash.update(byte_block)

    return sha3_hash.hexdigest()

class IDRiDProcessor(NaturalImageLoaderMixin, DatasetProcessor):
    name = 'IDRiD'

    def get_image_files(self) -> Sequence[ImageFile]:
        ret = []
        file_sha3s = set()
        for task, split in it.product(['A. Segmentation', 'B. Disease Grading'], ['a. Training', 'b. Testing']):
            for path in (self.dataset_root / task / '1. Original Images' / f'{split} Set').glob('*.jpg'):
                h = file_sha3(path)
                if h in file_sha3s:
                    continue
                file_sha3s.add(h)
                key = f'{task[0]}.{split[0]}.{path.stem}'
                ret.append(ImageFile(key, 'RGB/fundus', path))
        return ret

class IXIProcessor(Default3DLoaderMixin, DatasetProcessor):
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

class KaggleRDCProcessor(NaturalImageLoaderMixin, DatasetProcessor):
    name = 'Kaggle-RDC'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(f'{path.parent.name}-{path.stem}', 'RGB/fundus', path)
            for path in self.dataset_root.rglob('*.png')
        ]

class LIDCIDRIProcessor(Default3DLoaderMixin, DatasetProcessor):
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

class MRSpineSegProcessor(Default3DLoaderMixin, DatasetProcessor):
    name = 'MRSpineSeg_Challenge_SMU'

    def get_image_files(self) -> Sequence[ImageFile]:
        suffix = '.nii.gz'
        return [
            ImageFile(path.name[:-len(suffix)], 'MRI/T2', path)
            for path in self.dataset_root.glob(f'*/MR/*{suffix}')
        ]

class MSDProcessor(Default3DLoaderMixin, DatasetProcessor):
    @property
    def output_name(self):
        return self.name.replace('/', '-')

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

class NCTCTProcessor(Default3DLoaderMixin, TCIAProcessor):
    name = 'NCTCT'
    reader = PydicomReader

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'TCIA_CT_COLONOGRAPHY_06-22-2015'

    def get_image_files(self) -> Sequence[ImageFile]:
        meta = self.tcia_meta
        return [
            ImageFile(sid, 'CT', self.dataset_root / path)
            for sid, (num, path) in meta[['Number of Images', 'File Location']].iterrows()
            if num >= 81
        ]

class NIHChestXRayProcessor(NaturalImageLoaderMixin, DatasetProcessor):
    name = 'NIHChestX-ray'
    assert_gray_scale = True
    max_workers = 64
    chunksize = 10

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'CXR8'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.stem, 'gray/XR', path)
            for path in (self.dataset_root / 'images' / 'images').glob('*.png')
        ]

class PelviXNetProcessor(NaturalImageLoaderMixin, DatasetProcessor):
    name = 'PelviXNet'
    assert_gray_scale = True

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.stem, 'gray/XR', path)
            for path in (self.dataset_root / 'pxr-150' / 'images_all').glob('*.png')
        ]

class PICAIProcessor(Default3DLoaderMixin, DatasetProcessor):
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
            ImageFile(key := path.stem, f"MRI/{mapping[key.rsplit('_', 1)[1]]}", path)
            for path in (self.dataset_root / 'public_images').rglob('*.mha')
        ]

class Prostate158Processor(Default3DLoaderMixin, DatasetProcessor):
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

class ProstateDiagnosisProcessor(Default3DLoaderMixin, TCIAProcessor):
    name = 'PROSTATE-DIAGNOSIS'

    @property
    def dataset_root(self):
        return DATASETS_ROOT / self.name / 'TCIA_PROSTATE-DIAGNOSIS_06-22-2015'

    def get_image_files(self) -> Sequence[ImageFile]:
        meta = self.tcia_meta
        ignore = {'AX BLISSGAD8', 'AX BLISSGAD', 'dynTHRIVE'}
        ret = []
        for sid, (desc, path) in meta[['Series Description', 'File Location']].iterrows():
            if desc in ignore:
                continue
            if 'T2' in desc:
                modality = 'T2'
            elif 'GAD' in desc or desc == 'T1WTSEAXGd':
                modality = 'T1c'
            elif 'T1' in desc:
                modality = 'T1'
            else:
                raise ValueError
            ret.append(ImageFile(sid, f'MRI/{modality}', self.dataset_root / path))
        return ret

class ProstateMRIProcessor(Default3DLoaderMixin, DatasetProcessor):
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

class RibFracProcessor(Default3DLoaderMixin, DatasetProcessor):
    name = 'RibFrac'

    def get_image_files(self) -> Sequence[ImageFile]:
        suffix = '-image.nii.gz'
        return [
            ImageFile(path.name[:-len(suffix)], 'CT', path)
            for path in self.dataset_root.glob(f'*/*{suffix}')
        ]

class RSNA2020PEDProcessor(Default3DLoaderMixin, DatasetProcessor):
    name = 'RSNA-2020-PED'
    reader = PydicomReader

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile('.'.join(path.parts[-2:]), 'CT', path)
            for path in self.dataset_root.glob('*/*/*')
        ]

class RSNA2022CSFDProcessor(Default3DLoaderMixin, DatasetProcessor):
    name = 'RSNA-2022-CSFD'
    reader = PydicomReader

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.name, 'CT', path)
            for split in ['train', 'test']
            for path in (self.dataset_root / f'{split}_images').iterdir()
        ]

class STOICProcessor(Default3DLoaderMixin, DatasetProcessor):
    name = 'STOIC'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(path.stem, 'CT', path)
            for path in (self.dataset_root / 'stoic2021-training' / 'data' / 'mha').glob('*.mha')
        ]

class TotalSegmentatorProcessor(Default3DLoaderMixin, DatasetProcessor):
    name = 'TotalSegmentator'
    orientation = 'SRA'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(key, 'CT', path)
            for path in (self.dataset_root / 'Totalsegmentator_dataset').glob('*/ct.nii.gz')
            # this case seems to be corrupted
            # https://github.com/wasserth/TotalSegmentator/issues/24#issuecomment-1298390124
            if (key := path.parent.name) != 's0864'
        ]

class VerSeProcessor(Default3DLoaderMixin, DatasetProcessor):
    name = 'VerSe'

    def get_image_files(self) -> Sequence[ImageFile]:
        suffix = '.nii.gz'
        return [
            ImageFile(path.name[:-len(suffix)], 'CT', path)
            for path in self.dataset_root.glob(f'*/rawdata/*/*{suffix}')
        ]

class VSSEGProcessor(Default3DLoaderMixin, DatasetProcessor):
    name = 'Vestibular-Schwannoma-SEG'
    reader = PydicomReader
    orientation = 'SRA'

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
    from jsonargparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('datasets', nargs='*', type=str)
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--exclude', nargs='*', type=str, default=[])
    parser.add_argument('--ie', action='store_true', help='ignore exception')
    parser.add_argument('--max_workers', type=int, default=16)
    parser.add_argument('--chunksize', type=int, default=1)
    parser.add_argument('--override', action='store_true')
    args = parser.parse_args()
    if args.all:
        exclude = set(args.exclude + [MSDBrainTumourProcessor.__name__[:-len('Processor')]])
        datasets = [
            name for cls in globals().values()
            if inspect.isclass(cls) and issubclass(cls, DatasetProcessor) and hasattr(cls, 'name')
            and (name := cls.__name__[:-len('Processor')]) not in exclude
        ]
        print(datasets)
    else:
        datasets = args.datasets
    for dataset in datasets:
        processor_cls: type[DatasetProcessor] | None = globals().get(f'{dataset}Processor', None)
        if processor_cls is None:
            print(f'no processor for {dataset}')
        else:
            processor = processor_cls()
            processor.update_multiprocessing(args.max_workers, args.chunksize, args.override)
            print(dataset, f'max_workers: {processor.max_workers}, chunksize: {processor.chunksize}')
            try:
                processor.process()
            except Exception as e:
                if args.ie:
                    import traceback
                    print(traceback.format_exc())
                else:
                    raise e

if __name__ == '__main__':
    main()
