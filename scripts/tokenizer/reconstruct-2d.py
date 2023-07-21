from pathlib import Path

from PIL import Image
import einops
from jsonargparse import ArgumentParser
import pytorch_lightning as pl
import torch
from torchvision import transforms as tvt
from torchvision.utils import save_image

from luolib.models import load_ckpt
from pumt.conv import SpatialTensor
from pumt.tokenizer.vqvae import VQVAEModel

src = Path('test-images/T0005.jpg')

def main():
    parser = ArgumentParser()
    parser.add_class_arguments(VQVAEModel, 'model')
    parser.add_argument('--ckpt_path', type=Path)
    parser.add_argument('--out_path', type=Path)
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    print(args)
    img = Image.open(src).convert('RGB')
    transform = tvt.Compose([
        tvt.Resize(512),
        tvt.ToTensor(),
    ])
    x = transform(img).cuda() * 2 - 1
    x = einops.rearrange(x, 'c h w -> 1 c 1 h w')
    x = SpatialTensor(x, 5)
    model: VQVAEModel = args.model.cuda()

    load_ckpt(model, args.ckpt_path, state_dict_key='vqvae')
    with torch.no_grad():
        x_rec, quant_out = model(x)
    print('output shape:', x_rec.shape)
    print('index shape:', quant_out.index.shape)
    save_image((x_rec[0, :, 0] + 1) / 2, args.out_path)

if __name__ == '__main__':
    main()
