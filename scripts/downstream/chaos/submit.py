# convert nnU-Net output to CHAOS submission format

from pathlib import Path
import shutil

import nibabel as nib
from jsonargparse import ArgumentParser
import numpy as np

from downstream.chaos.data import extract_template, save_pred

def main():
    parser = ArgumentParser()
    parser.add_argument('src', type=Path)
    parser.add_argument('dst', type=Path, default='CHAOS-submit')
    args = parser.parse_args()
    extract_template(args.dst)
    data_dir = Path('downstream/data/CHAOS')
    for split in ['Train', 'Test']:
        for case_dir in (data_dir / split).iterdir():
            for modality in ['T1DUAL', 'T2SPIR']:
                num = case_dir.name
                src_path = args.src / f'{modality[:2]}_{num}.nii.gz'
                pred = nib.load(src_path).get_fdata().astype(np.uint8)
                img_rel_path = f'MR/{num}/{modality}'
                save_pred(pred, args.dst / 'Task5' / img_rel_path / 'Results')
                pred[pred != 1] = 0
                save_pred(pred, args.dst / 'Task3' / img_rel_path / 'Results')
    shutil.make_archive(
        str(args.dst / 'submit'), 'zip', args.dst, '.',
        verbose=True,
    )

if __name__ == '__main__':
    main()
