from pathlib import Path

from jsonargparse import ArgumentParser
import nibabel as nib
import torch
from torchvision.utils import save_image

from luolib.models import spadop
from luolib.models.utils import load_ckpt
from monai import transforms as mt
from monai.data import MetaTensor
from pumit.tokenizer import LOGIT_EPS, VQVisualTokenizer, VQVAEModel
from pumit.transforms import ensure_rgb, rgb_to_gray

src = Path('test-images/amos_0065.nii.gz')

def main():
    torch.set_float32_matmul_precision('medium')
    parser = ArgumentParser()
    parser.add_subclass_arguments(VQVisualTokenizer, 'model')
    parser.add_argument('--ckpt_path', type=Path)
    # parser.add_argument('--out_path', type=Path)
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    print(args)
    loader = mt.Compose(
        [
            mt.LoadImage(ensure_channel_first=True, image_only=True),
            mt.ToTensor(device='cuda'),
            mt.ScaleIntensityRangePercentiles(0.5, 99.5, 0, 1, clip=True),
            mt.Orientation('SAR'),
            mt.Resize((-1, 256, 256)),
            mt.CenterSpatialCrop((80, 256, 256)),
        ],
        lazy=True,
    )
    meta_tensor: MetaTensor = loader(src)
    affine = meta_tensor.affine
    aniso_d = int(max(1, meta_tensor.pixdim[0] / min(meta_tensor.pixdim[1:]))).bit_length() - 1
    x = spadop.SpatialTensor(meta_tensor.as_tensor(), aniso_d)
    nib.save(nib.Nifti1Image(x[0].detach().float().cpu().numpy(), affine.numpy()), 'origin.nii.gz')
    print(x.shape, x.num_pending_hw_downsamples)
    model: VQVisualTokenizer = args.model.cuda().train()
    # load_ckpt(model, args.ckpt_path, state_dict_key='model')
    load_ckpt(model, args.ckpt_path, state_dict_key='state_dict')
    with torch.no_grad():
        x, _ = ensure_rgb(x)
        x = x[None]
        x_rec, x_rec_logit, vq_out = model.autoencode(x.logit(LOGIT_EPS))
        x_rec = rgb_to_gray(x_rec[0])
    for i in range(x_rec.shape[1]):
        save_image(x_rec[:, i], f'r3d/{i}-rec.png')
        save_image(x[:, i], f'r3d/{i}-origin.png')
    # nib.save(nib.Nifti1Image(x_rec.float().cpu().numpy(), affine.numpy()), args.out_path)

if __name__ == '__main__':
    main()
