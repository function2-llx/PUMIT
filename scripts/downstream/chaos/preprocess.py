from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from monai import transforms as mt
from monai.utils import ImageMetaKey

from downstream.chaos.data import read_label

dataset_dir = Path('datasets/CHAOS')
output_dir = Path('downstream/data/CHAOS')

def main():
    output_dir.mkdir(parents=True)
    for split in ['Train', 'Test']:
        for case_dir in (dataset_dir / f'{split}_Sets' / 'MR').iterdir():
            if not case_dir.is_dir():
                continue
            for seq in ['T1DUAL', 'T2SPIR']:
                src_dir = case_dir / seq / 'DICOM_anon'
                loader = mt.LoadImage('pydicomreader', True, ensure_channel_first=True)
                save_dir = output_dir / split / case_dir.name / seq
                saver = mt.SaveImage(save_dir, '', resample=False, separate_folder=False)
                if seq == 'T1DUAL':
                    img_in = loader(src_dir / 'InPhase')
                    affine = img_in.affine
                    img_out = loader(src_dir / 'OutPhase')
                    img = torch.cat((img_in, img_out), dim=0)
                elif seq == 'T2SPIR':
                    img = loader(src_dir)
                    affine = img.affine
                else:
                    raise ValueError
                img.meta[ImageMetaKey.FILENAME_OR_OBJ] = 'image'
                saver(img)
                if split == 'Test':
                    continue
                label = read_label(case_dir / seq / 'Ground')
                nib.save(
                    nib.Nifti1Image(label, affine=affine, dtype=np.int64),
                    save_dir / 'label.nii.gz',
                )

if __name__ == '__main__':
    main()
