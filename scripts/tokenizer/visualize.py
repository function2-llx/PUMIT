from argparse import ArgumentParser
from pathlib import Path

import einops
from omegaconf import OmegaConf
import torch
from torch.nn import functional as nnf
from torchvision.utils import save_image

from luolib.conf.utils import instantiate
from luolib.models import load_ckpt

from pumt.tokenizer.model import VQGAN

def main():
    parser = ArgumentParser()
    parser.add_argument('--dir', type=Path, default=None)
    parser.add_argument('--conf_path', type=Path, default=None)
    parser.add_argument('--ckpt_path', type=Path, default=None)
    parser.add_argument('--out_path', type=Path, default=None)
    parser.add_argument('--nr', type=int, default=32)
    parser.add_argument('--nc', type=int, default=32)
    args = parser.parse_args()
    if args.conf_path is None:
        args.conf_path = args.dir / 'conf.yaml'
    if args.ckpt_path is None:
        args.ckpt_path = args.dir / 'model.ckpt'
    if args.out_path is None:
        args.out_path = args.dir / 'visualize.png'
    print(args)

    conf = OmegaConf.load(args.conf_path)
    conf['kwargs']['in_channels'] = 3
    model: VQGAN = instantiate(conf).cuda().eval()
    load_ckpt(model, args.ckpt_path)
    z = einops.rearrange(model.quantize.embedding.weight[:args.nr * args.nc], 'n c -> n c 1 1 1')
    with torch.no_grad():
        out = model.decode(z, [torch.tensor([0, 1, 1])] * len(model.decoder.up))
        out = nnf.interpolate(out, (1, 16, 16), mode='trilinear')
    out = nnf.pad(out, (1, 1, 1, 1))
    print(out.shape)
    out = einops.rearrange(out, '(nr nc) c 1 h w -> c (nr h) (nc w)', nr=args.nr)
    print(out.shape)
    save_image(out, args.out_path)

if __name__ == '__main__':
    main()
