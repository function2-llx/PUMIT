from pathlib import Path

import einops
from jsonargparse import ArgumentParser
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from luolib.models import load_ckpt
from pumt import PUMTDataModule
from pumt.sac import SpatialTensor
from pumt.tokenizer import VQTokenizer
from pumt.transforms import rgb_to_gray

output_dir = Path('tokenizer-eval')

def main():
    torch.set_float32_matmul_precision('high')
    parser = ArgumentParser()
    parser.add_subclass_arguments(VQTokenizer, 'model')
    parser.add_class_arguments(PUMTDataModule, 'data')
    parser.add_argument('--ckpt_path', type=Path)
    parser.add_argument('--state_dict_key', type=str, default='model')
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    model: VQTokenizer = args.model.cuda().eval()
    load_ckpt(model, args.ckpt_path)
    dm: PUMTDataModule = args.data
    torch.set_grad_enabled(False)
    cnt = torch.zeros(1024, device='cuda')
    entropy = []
    for i, (x, aniso_d, modality, path) in enumerate(tqdm(dm.val_dataloader())):
        modality: str = modality[0]
        path = Path(path[0])
        x = SpatialTensor(x.cuda(), aniso_d)
        x_rec, vq_out = model.forward(2 * x - 1)
        x_rec = (x_rec + 1) / 2
        x_rec.clamp(0, 1)
        entropy.append(vq_out.entropy.item())
        cnt += einops.reduce(vq_out.index, '... ne -> ne', 'sum')
        if not modality.startswith('RGB'):
            x = rgb_to_gray(x, batched=True)
            x_rec = rgb_to_gray(x_rec, batched=True)
        (rec_save_dir := output_dir / 'rec' / path.parts[-3] / path.stem).mkdir(exist_ok=True, parents=True)
        for j in range(x.shape[2]):
            save_image(x[0, :, j], rec_save_dir / f'{j}.png')
            save_image(x_rec[0, :, j], rec_save_dir / f'{j}-rec.png')

    sns.histplot(
        {
            'index': np.arange(128),
            'utilization': einops.reduce(cnt, '(x y) -> x', 'mean', x=128).cpu().numpy(),
        },
        x='index',
        weights='utilization',
        stat='percent',
        bins=128,
    )
    plt.savefig(output_dir / 'utilization.pdf', bbox_inches='tight', pad_inches=0)
    plt.savefig(output_dir/ 'utilization.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    Path(output_dir / 'entropy.txt').write_text(str(np.mean(entropy)))

if __name__ == '__main__':
    main()
