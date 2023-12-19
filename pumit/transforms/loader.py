from collections.abc import Sequence

import cytoolz
import numpy as np
import torch
from torch.types import Device

from monai import transforms as mt
from monai.data import get_random_patch, to_affine_nd
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
        axis_θ = self.R.uniform(0, 2 * np.pi)
        if spacing[0] < 1.5 * spacing_xy:
            # fully 3D rotation
            axis_φ = self.R.uniform(0, np.pi / 2)
        elif spacing[0] < 3 * spacing_xy:
            # limited 3D rotation, similar to nnU-Net
            axis_φ = self.R.uniform(np.pi / 3, np.pi / 2)
        else:
            # dummy 2D rotation along z-axis
            axis_φ = np.pi / 2
        axis_z = np.sin(axis_φ)
        r_xy = np.cos(axis_φ)
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
            print(axis, θ)
        scale_affine = np.diag(trans_info['scale'])
        # note that when scale is not isotropic, the rotate_affine @ scale_affine is NOT commutative
        # determine the multiplication order as follows
        if spacing[0] < 1.5 * spacing_xy and self.R.uniform() < 0.5:
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
        patch_trans = mt.Compose(
            [
                mt.SpatialPad(patch_size),
                mt.CenterSpatialCrop(patch_size),
                mt.Affine(affine=to_affine_nd(3, affine), image_only=True),
            ],
            lazy=True,
            overrides={
                'mode': GridSampleMode.BILINEAR,
                'padding_mode': GridSamplePadMode.ZEROS,
                'dtype': torch.float32,
            }
        )
        patch = patch_trans(img)
        return patch

if __name__ == '__main__':
    R = get_rotation_matrix(np.array((1, 0, 0)), np.pi / 3)
    print(R)
