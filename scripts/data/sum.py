from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('data_dir', type=Path)
    args = parser.parse_args()

    meta = pd.read_csv(args.data_dir / 'images-meta.csv', index_col='key')

    # brats
    # meta = meta[meta['modality'] == 'MRI/T2']

    # BrainPTM-2021
    # meta = meta[meta.index.str.endswith('DWI-0') | (meta['modality'] == 'MRI/T1')]

    # MSD-05 prostate
    # meta = meta[meta['modality'] == 'MRI/T2']

    # IXI
    # meta = meta[meta['modality'].isin(['MRI/T1', 'MRI/T2', 'MRI/MRA', 'MRI/PD']) | meta.index.str.endswith('DTI-01')]

    # Vestibular-Schwannoma-SEG
    # meta = meta[meta['modality'] == 'MRI/T1']

    print(len(meta.index))
    print(meta['shape-0'].sum())


if __name__ == '__main__':
    main()
