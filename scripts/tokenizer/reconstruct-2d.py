from argparse import ArgumentParser
from pathlib import Path

import einops
import torch
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from torchvision import transforms as tvt
from torchvision.utils import save_image

from luolib.conf.utils import instantiate_from_conf
from luolib.models import load_ckpt

from pumt.tokenizer.vq_model import VQModel

src = Path('src/view1_frontal.jpg')

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
        args.out_path = args.dir / f'{src.stem}-rec.png'
    print(args)

    pl.seed_everything(42)
    img = Image.open(src).convert('L')
    transform = tvt.Compose([
        tvt.Resize(512),
        tvt.ToTensor(),
    ])
    x = transform(img).cuda() * 2 - 1
    x = einops.rearrange(x, 'c h w -> 1 c 1 h w')
    from torch.nn import functional as nnf
    # x = nnf.interpolate(x, scale_factor=(1, 0.5, 0.5), mode='trilinear')
    print(x.shape)
    conf = OmegaConf.load(args.conf_path)
    model: VQModel = instantiate_from_conf(conf).cuda().eval()
    load_ckpt(model, args.ckpt_path)
    spacing = torch.tensor([[1e6, 1, 1]], device='cuda')
    with torch.no_grad():
        dec, quant_out = model.forward(x, spacing)
    print('output shape:', dec.shape)
    dec = nnf.interpolate(dec, size=(1, 512, 512), mode='trilinear')
    print(dec.shape, quant_out.loss, quant_out.index.shape)
    save_image((dec[0, :, 0] + 1) / 2, args.out_path)

if __name__ == '__main__':
    main()
