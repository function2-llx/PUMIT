from pathlib import Path

import einops
from jsonargparse import ArgumentParser
import torch
from torch.nn import functional as nnf
from torchvision.utils import save_image

from luolib.models import load_ckpt
from pumt.sac import SpatialTensor
from pumt.tokenizer import VQTokenizer

def main():
    torch.set_float32_matmul_precision('high')
    torch.set_default_device('cuda')
    parser = ArgumentParser()
    parser.add_subclass_arguments(VQTokenizer, 'model')
    parser.add_argument('--ckpt_path', type=Path)
    parser.add_argument('--out_path', type=Path)
    parser.add_argument('--nr', type=int, default=32)
    parser.add_argument('--nc', type=int, default=32)
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    print(args)
    model: VQTokenizer = args.model.eval()
    load_ckpt(model, args.ckpt_path, state_dict_key='model')
    z = einops.rearrange(model.quantize.embedding.weight[:args.nr * args.nc], 'n c -> n c 1 1 1')
    z = SpatialTensor(z, aniso_d=5)
    z.num_downsamples = 4
    with torch.no_grad():
        out = model.decoder(z)
    out = nnf.pad(out, (1, 1, 1, 1))
    print(out.shape)
    out = einops.rearrange(out, '(nr nc) c 1 h w -> c (nr h) (nc w)', nr=args.nr)
    print(out.shape)
    save_image(out, args.out_path)

if __name__ == '__main__':
    main()
