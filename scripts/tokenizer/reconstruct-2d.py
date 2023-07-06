from argparse import ArgumentParser
from pathlib import Path

import einops
import torch
import pytorch_lightning as pl
from PIL import Image

from torchvision import transforms as tvt
from torchvision.utils import save_image

from luolib.conf.utils import instantiate_from_conf
from luolib.models import load_ckpt
from luolib.utils import DataKey

from pumt.tokenizer.vqgan import VQGAN

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
    model: VQGAN = instantiate_from_conf(args.conf_path).cuda().eval()
    load_ckpt(model, args.ckpt_path)
    spacing = torch.tensor([[1e6, 1, 1]], device='cuda')
    with torch.no_grad():
        x_rec, quant_out = model.forward(x, spacing)
        loss, disc_loss, log_dict = model.loss.forward(x, x_rec, spacing, quant_out.loss)
    print('output shape:', x_rec.shape)
    x_rec = nnf.interpolate(x_rec, size=(1, 512, 512), mode='trilinear')
    print('index shape:', quant_out.index.shape)
    print('rec shape:', x_rec.shape)
    for k, v in log_dict.items():
        print(k, v)
    save_image((x_rec[0, :, 0] + 1) / 2, args.out_path)

if __name__ == '__main__':
    main()
