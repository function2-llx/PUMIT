from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('--move', action='store_true')
    args = parser.parse_args()
    for dataset_dir in Path('datasets-pumit').iterdir():
        dataset_name = dataset_dir.name
        meta = pd.read_csv(dataset_dir / 'images-meta.csv', index_col='key')
        incomplete = meta[pd.isna(meta['modality'])]
        if not incomplete.empty:
            print('incomplete:', dataset_name)
            continue
        large = meta[meta[['shape-1', 'shape-2']].min(axis='columns') > 512]
        if large.shape[0] > 0:
            print(dataset_name, large.shape[0])
            print(large[['shape-1', 'shape-2']], '\n')
            if args.move:
                (dataset_dir / 'images-meta.csv').rename(dataset_dir / 'images-meta-old.csv')
                meta.drop(index=large.index, inplace=True)
                pd.concat([meta, pd.DataFrame(index=large.index)]).to_csv(dataset_dir / 'images-meta.csv')

if __name__ == '__main__':
    main()
