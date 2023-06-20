from abc import abstractmethod, ABC
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import itertools as it
from pathlib import Path

import cytoolz
import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from monai import transforms as mt
from monai.data import MetaTensor

DATASETS_ROOT = Path('datasets')
PROCESSED_ROOT = Path('datasets-PUMT')
PROCESSED_ROOT.mkdir(exist_ok=True, parents=True)

@dataclass
class ImageFile:
    key: str
    modality: str | list[str]
    path: Path
    weight: float = 1

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

    @abstractmethod
    def get_loader(self) -> Callable[[Path], MetaTensor]:
        pass

    @abstractmethod
    def get_cropper(self) -> Callable[[MetaTensor], MetaTensor]:
        pass

    def process(self, ncols: int = 80, max_workers: int = 16, **kwargs):
        files = self.get_image_files()
        if len(files) == 0:
            return
        (self.output_root / 'data').mkdir(parents=True)
        results = process_map(self.process_file, files, max_workers=max_workers, ncols=ncols, **kwargs)
        pd.DataFrame.from_records(
            cytoolz.concat(results), index='key'
        ).to_excel(self.output_root / 'images-meta.xlsx', freeze_panes=(1, 1))

    def process_file(
        self,
        file: ImageFile,
        loader: Callable[[Path], MetaTensor] | None = None,
        cropper: Callable[[MetaTensor], MetaTensor] | None = None,
    ):
        if loader is None:
            loader = self.get_loader()
        data = loader(file.path)
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
    def get_loader(self):
        return mt.Compose([
            mt.LoadImage(image_only=True, dtype=None, ensure_channel_first=True),
            mt.Orientation('RAS'),
            MetaTensor.contiguous,
        ])

class Default2DLoaderMixin:
    dummy_dim: int = 2
    assert_gray_scale: bool = False

    def adapt_to_3d(self, img: MetaTensor):
        if self.assert_gray_scale:
            from monai.utils import ImageMetaKey
            assert (img[0] == img[1]).all() and (img[0] == img[2]).all(), img.meta[ImageMetaKey.FILENAME_OR_OBJ]
            img = img[0:1]
        img = img.unsqueeze(self.dummy_dim + 1)
        img.affine[self.dummy_dim, self.dummy_dim] = 1e8
        return img

    def get_loader(self):
        return mt.Compose([
            mt.LoadImage(image_only=True, dtype=None, ensure_channel_first=True),
            self.adapt_to_3d,
        ])

class MinPercentileCropperMixin:
    min_p: float = 0.5

    def get_cropper(self):
        def select_fn(img: MetaTensor):
            v = mt.percentile(img.view(img.shape[0], -1).float(), self.min_p, dim=1)
            for _ in range(img.ndim - 1):
                v = v[..., None]
            return (img > v).all(dim=0, keepdim=True)

        return mt.CropForeground(select_fn)

class ACDCProcessor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
    name = 'ACDC'

    def get_image_files(self):
        return [
            ImageFile(key := patient_dir.name, modality='cine MRI', path=patient_dir / f'{key}_4d.nii.gz')
            for split in ['training', 'testing']
            for patient_dir in (self.dataset_root / split).iterdir() if patient_dir.is_dir()
        ]

    def process_file_data(self, key: str, data: MetaTensor, cropper: Callable[[MetaTensor], MetaTensor]) -> list[dict]:
        t = data.shape[0]
        return [
            # cine MRI results in very similar scans, empirically adjust the weight
            self.process_image(data[i:i + 1], f'{key}-{i}', 'MRI', cropper, weight=2 / t)
            for i in range(t)
        ]

class AMOS22Processor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
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

class BrainPTM2021Processor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
    name = 'BrainPTM-2021'

    def get_image_files(self):
        ret = []
        for case_dir in self.dataset_root.glob('case_*'):
            key = case_dir.name
            ret.append(ImageFile(f'{key}-T1', 'MRI/T1', case_dir / 'T1.nii.gz'))
            ret.append(ImageFile(f'{key}-DWI', 'MRI/DWI', case_dir / 'Diffusion.nii.gz'))
        return ret

class BraTS2023SegmentationProcessor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
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

class BCVProcessor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
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

class ChákṣuProcessor(Default2DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
    name = 'Chákṣu'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(f'{image_dir.name}.{path.stem}', 'RGB/fundus', path)
            for image_dir in self.dataset_root.glob('*/1.0_Original_Fundus_Images/*') if image_dir.is_dir()
            for path in image_dir.iterdir() if path.suffix.lower() in ['.png', '.jpg']
        ]

class CHAOSProcessor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
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

class CrossMoDA2022Processor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
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

class CHASEDB1Processor(Default2DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
    name = 'CHASE_DB1'
    min_p = 0.

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(key := f'{i:02d}{side}', 'RGB/fundus', self.dataset_root / f'Image_{key}.jpg')
            for i, side in it.product(range(1, 15), ('L', 'R'))
        ]

class CHUACProcessor(Default2DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
    name = 'CHUAC'
    min_p = 0

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(str(i), 'RGB/fundus', self.dataset_root / 'Original' / f'{i}.png')
            for i in range(1, 31)
        ]

class FLARE22Processor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
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

class HaNSegProcessor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
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

def file_sha3(filepath: Path):
    sha3_hash = hashlib.sha3_256()

    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha3_hash.update(byte_block)

    return sha3_hash.hexdigest()

class IDRiDProcessor(Default2DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
    name = 'IDRiD'
    min_p = 0

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

class IXIProcessor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
    name = 'IXI'

    def get_image_files(self) -> Sequence[ImageFile]:
        pass

class KaggleRDCProcessor(Default2DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
    name = 'kaggle-RDC'

    def get_image_files(self) -> Sequence[ImageFile]:
        return [
            ImageFile(f'{path.parent.name}-{path.stem}', 'RGB/fundus', path)
            for path in self.dataset_root.rglob('*.png')
        ]

class LIDCIDRIProcessor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
    name = 'LIDC-IDRI'

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

class MRSpineSegProcessor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
    name = 'MRSpineSeg_Challenge_SMU'

    def get_image_files(self) -> Sequence[ImageFile]:
        suffix = '.nii.gz'
        return [
            ImageFile(path.name[:-len(suffix)], 'MRI/T2', path)
            for path in self.dataset_root.rglob(f'*{suffix}')
        ]

class MSDProcessor(Default3DLoaderMixin, MinPercentileCropperMixin, DatasetProcessor):
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

def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('datasets', nargs='*', type=str)
    args = parser.parse_args()
    for dataset in args.datasets:
        processor_cls: type[DatasetProcessor] | None = globals().get(f'{dataset}Processor', None)
        if processor_cls is None:
            print(f'no processor for {dataset}')
        else:
            processor = processor_cls()
            print(dataset)
            try:
                processor.process()
            except Exception:
                pass

if __name__ == '__main__':
    main()
