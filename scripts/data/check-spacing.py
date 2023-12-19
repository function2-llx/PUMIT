import numpy as np
import pandas as pd

from pumit.datamodule import DATA_ROOT

def main():
    r = []
    w = []
    for dataset_dir in DATA_ROOT.iterdir():
        dataset_name = dataset_dir.name
        meta: pd.DataFrame = pd.read_pickle(dataset_dir / 'meta.pkl')
        # select 3D items
        meta = meta[meta['shape'].map(lambda shape: shape[0] > 1)]
        if meta.shape[0] == 0:
            continue
        meta_aniso_xy = meta[meta['spacing'].map(lambda spacing: np.abs(spacing[1] - spacing[2]) >= 1e-3)]
        if meta_aniso_xy.shape[0] > 0:
            print(dataset_name, 'aniso', meta_aniso_xy.shape[0])
        dataset_r = meta['spacing'].map(lambda spacing: spacing[0] / spacing[1:].min()).to_numpy()
        r.append(dataset_r)
        if (max_r := dataset_r.max()) >= 16:
            print(dataset_name, 'max_r', max_r)
        w.append(meta['weight'].to_numpy())
    r = np.concatenate(r).astype(np.int32)
    print(np.unique(r))
    print(np.bincount(r, np.concatenate(w)).astype(np.int32))

if __name__ == '__main__':
    main()
