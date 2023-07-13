from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
import torch
import pytorch_lightning as pl

from luolib.conf.utils import instantiate
from luolib.models import load_ckpt
from luolib.utils import DataKey
from monai import transforms as mt

from pumt.tokenizer.model import VQGAN

src = Path('test-images/amos_0065.nii.gz')

def main():
    parser = ArgumentParser()
    parser.add_argument('--dir', type=Path, default=None)
    parser.add_argument('--conf_path', type=Path, default=None)
    parser.add_argument('--ckpt_path', type=Path, default=None)
    parser.add_argument('--out_path', type=Path, default=None)
    args = parser.parse_args()
    if args.conf_path is None:
        args.conf_path = args.dir / 'conf.yaml'
    if args.ckpt_path is None:
        args.ckpt_path = args.dir / 'model.ckpt'
    if args.out_path is None:
        args.out_path = args.dir / f'{src.stem}-rec.nii.gz'
    print(args)

    pl.seed_everything(42)
    torch.set_float32_matmul_precision('high')
    loader = mt.Compose([
        mt.LoadImage(ensure_channel_first=True, image_only=True),
        mt.Orientation('SAR'),
        mt.ScaleIntensityRangePercentiles(0.5, 99.5, 0, 1, clip=True),
        mt.Resize((-1, 256, 256)),
        mt.CenterSpatialCrop((80, 256, 256)),
    ])
    img = loader(src).cuda()
    affine = img.affine
    spacing = img.pixdim.cuda()[None]
    x = img.as_tensor().cuda()[None] * 2 - 1
    print(x.shape, spacing)
    print(affine)
    model: VQGAN = instantiate(args.conf_path).cuda().eval()
    load_ckpt(model, args.ckpt_path)
    with torch.no_grad():
        x_rec, quant_out = model.forward(x, spacing)
        _, _, log_dict = model.loss.forward(x, x_rec, spacing, quant_out.loss)
    for k, v in log_dict.items():
        print(k, v)
    nib.save(nib.Nifti1Image((x_rec[0, 0] / 2 + 0.5).detach().float().cpu().numpy(), affine), args.out_path)

if __name__ == '__main__':
    main()
