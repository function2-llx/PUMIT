from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('--move', action='store_true')
    args = parser.parse_args()
    r = []
    w = []
    for dataset_dir in Path('datasets-PUMT').iterdir():
        dataset_name = dataset_dir.name
        meta = pd.read_csv(dataset_dir / 'images-meta.csv', index_col='key')
        incomplete = meta[pd.isna(meta['modality'])]
        if not incomplete.empty:
            print('incomplete:', dataset_name)
            continue
        meta = meta[meta['modality'].str.startswith('CT') | meta['modality'].str.contains('MRI')]
        ani = meta[(meta['space-1'] - meta['space-2']).abs() > 1e-3]
        if ani.shape[0] > 0:
            print(dataset_name, ani.shape[0])
            print(ani[['space-0', 'space-1', 'space-2']], '\n')
            if args.move:
                (dataset_dir / 'images-meta.csv').rename(dataset_dir / 'images-meta-old.csv')
                meta.drop(index=ani.index, inplace=True)
                pd.concat([meta, pd.DataFrame(index=ani.index)]).to_csv(dataset_dir / 'images-meta.csv')
        r.append((meta['space-0'] / meta[['space-1', 'space-2']].min(axis='columns')).to_numpy())
        w.append(meta['weight'].to_numpy())
    r = np.concatenate(r).astype(np.int32)
    print(np.unique(r))
    print(np.bincount(r, np.concatenate(w)).astype(np.int32))

if __name__ == '__main__':
    main()
