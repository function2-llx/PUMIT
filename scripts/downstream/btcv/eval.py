import json
from pathlib import Path

from jsonargparse import ArgumentParser
from tqdm import tqdm

from monai import transforms as mt
from monai.data import load_decathlon_datalist
from monai.metrics import DiceMetric, SurfaceDistanceMetric, HausdorffDistanceMetric
from monai.networks import one_hot
from monai.utils import MetricReduction

num_fg_classes = 13

def main():
    parser = ArgumentParser()
    parser.add_argument('pred_dir', type=Path)
    parser.add_argument('data_dir', type=Path, default='downstream/data/BTCV')
    parser.add_argument('datalist_path', type=Path, default='downstream/data/BTCV/smit.json')
    args = parser.parse_args()
    metrics = [
        ('dice', DiceMetric(False, MetricReduction.NONE)),
        ('msd', SurfaceDistanceMetric(False, True, 'euclidean', reduction=MetricReduction.NONE)),
        ('hd', HausdorffDistanceMetric(False, directed=False, reduction=MetricReduction.NONE)),
    ]
    data = load_decathlon_datalist(args.datalist_path, data_list_key='validation', base_dir=args.data_dir)
    loader = mt.Compose([
        mt.LoadImage(image_only=True, ensure_channel_first=True),
        mt.ToDevice('cuda'),
    ])

    for case in tqdm(data, ncols=80):
        case_id = Path(case['image']).name.split('.', 1)[0][-4:]
        label = one_hot(loader(case['label'])[None], num_fg_classes + 1)
        pred = one_hot(loader(args.pred_dir / f'img{case_id}.nii.gz')[None], num_fg_classes + 1)
        for _, metric in metrics:
            metric(pred, label)
    ret = {}
    for name, metric in metrics:
        m = metric.aggregate(MetricReduction.MEAN_BATCH)
        if name == 'dice':
            m *= 100
        print(name, m)
        ret[name] = {
            'all': m.tolist(),
            'mean': m.mean().item(),
        }
    Path(args.pred_dir / 'eval.json').write_text(json.dumps(ret, indent=4, ensure_ascii=False))

if __name__ == '__main__':
    main()
