from pathlib import Path

import pandas as pd

dataset_root = Path('processed-data')

def main():
    for dataset_dir in dataset_root.iterdir():
        dataset_name = dataset_dir.name
        print(dataset_name)
        for key in pd.read_csv(dataset_dir / 'images-meta.csv', dtype={'key': 'string'})['key']:
            if not (path := dataset_dir / 'data' / f'{key}.npy').exists():
                print(path)

if __name__ == '__main__':
    main()
