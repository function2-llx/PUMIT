from collections.abc import Sequence

import numpy as np
import torch
from torch.types import Device

from monai import transforms as mt
from monai.data import get_random_patch
from monai.utils import GridSampleMode, GridSamplePadMode

def get_rotation_matrix(axis: Sequence[float], θ: float) -> np.ndarray:
    cos = np.cos(θ)
    sin = np.sin(θ)
    x, y, z = axis
    return np.array((
        (cos + x * x * (1 - cos), x * y * (1 - cos) - z * sin, x * z * (1 - cos) + y * sin),
        (y * x * (1 - cos) + z * sin, cos + y * y * (1 - cos), y * z * (1 - cos) - x * sin),
        (z * x * (1 - cos) - y * sin, z * y * (1 - cos) + x * sin, cos + z * z * (1 - cos)),
    ))

def smooth_for_resampling(img: torch.Tensor, downsample_scale: Sequence[int]):
    assert len(downsample_scale) == img.ndim - 1
    factors = torch.as_tensor(downsample_scale)
    # use the default sigma in skimage.transform.resize
    anti_aliasing_sigma = ((factors - 1) / 2).clamp(0).tolist()
    anti_aliasing_filter = mt.GaussianSmooth(anti_aliasing_sigma)
    smoothed_img = anti_aliasing_filter(img)
    return smoothed_img

class PUMITLoader(mt.Randomizable, mt.Transform):
    """
    This loader calculates affine transform before loading, then read content necessary and as little as possible from the disk
    """

    def __init__(self, rotate_p: float, device: Device = 'cpu'):
        self.rotate_p = rotate_p
        self.device = device

    def get_rotation(self, spacing: np.ndarray):
        if self.R.uniform() >= self.rotate_p:
            return None, None
        spacing_xy = spacing[1:].min()
        if spacing[0] < 3 * spacing_xy:
            # 3D rotation with restricted axis closing to z-axis
            axis_φ = self.R.uniform(7 * np.pi / 15, np.pi / 2)
        else:
            # dummy 2D rotation along z-axis
            axis_φ = np.pi / 2
        axis_z = np.sin(axis_φ)
        r_xy = np.cos(axis_φ)
        axis_θ = self.R.uniform(0, 2 * np.pi)
        axis = np.array((axis_z, r_xy * np.sin(axis_θ), r_xy * np.cos(axis_θ)))
        θ = self.R.uniform(0, 2 * np.pi)
        return axis, θ

    def __call__(self, data: dict):
        trans_info = data['_trans']
        spacing = data['spacing']
        spacing_xy = spacing[1:].min()
        axis, θ = self.get_rotation(spacing)
        if axis is None:
            rotate_affine = np.eye(3)
        else:
            # just remember that affine is the matrix for inverse transform
            rotate_affine = get_rotation_matrix(axis, -θ)
        scale = trans_info['scale']
        scale_affine = np.diag(scale)
        # note that when scale is not isotropic, the rotate_affine @ scale_affine is NOT commutative
        # determine the multiplication order as follows
        if spacing[0] < 1.5 * spacing_xy and self.R.uniform() < 0.5:
            # when the spacing is not far from being isotropic, may change the order
            affine = rotate_affine @ scale_affine
        else:
            # scale always goes first for anisotropic data
            affine = scale_affine @ rotate_affine
        patch_size = trans_info['size']
        load_size = np.ceil(np.abs(affine) @ patch_size).astype(np.int32)
        load_slice = get_random_patch(
            data['shape'], np.minimum(data['shape'], load_size), self.R,
        )
        img = np.load(data['img'], 'r')
        img = torch.tensor(img[:, *load_slice], device=self.device)
        # create affine with translation
        affine_t = np.eye(4)
        affine_t[:3, :3] = affine
        patch_trans = mt.Compose(
            [
                mt.SpatialPad(patch_size),
                mt.CenterSpatialCrop(patch_size),
                mt.Affine(affine=affine_t, spatial_size=patch_size, image_only=True),
            ],
            lazy=True,
            overrides={
                'mode': GridSampleMode.BILINEAR,
                'padding_mode': GridSamplePadMode.ZEROS,
                'dtype': torch.float32,
            }
        )
        patch_trans.set_random_state(state=self.R)
        patch = patch_trans(img)
        return patch

if __name__ == '__main__':
    R = get_rotation_matrix(np.array((1, 0, 0)), np.pi / 3)
    print(R)
