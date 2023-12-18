from collections.abc import Sequence

import numpy as np

from monai import transforms as mt

def get_rotation_matrix(axis: Sequence[float], θ: float) -> np.ndarray:
    cos = np.cos(θ)
    sin = np.sin(θ)
    x, y, z = axis
    return np.array((
        (cos + x * x * (1 - cos), x * y * (1 - cos) - z * sin, x * z * (1 - cos) + y * sin),
        (y * x * (1 - cos) + z * sin, cos + y * y * (1 - cos), y * z * (1 - cos) - x * sin),
        (z * x * (1 - cos) - y * sin, z * y * (1 - cos) + x * sin, cos + z * z * (1 - cos)),
    ))

class PUMITLoader(mt.Randomizable):
    """
    This loader calculates cropping & affine transform before loading, then read as little as possible from the disk
    """

    def __init__(self, rotate_p: float):
        self.rotate_p = rotate_p

    def get_rotation(self, spacing: np.ndarray):
        if self.R.uniform() >= self.rotate_p:
            return
        spacing_xy = spacing[1:].min()
        axis_θ = self.R.uniform(0, 2 * np.pi)
        if spacing[0] < 1.5 * spacing_xy:
            # fully 3D rotation
            axis_φ = self.R.uniform(0, np.pi / 2)
        elif spacing[0] < 3 * spacing_xy:
            # limited 3D rotation, following nnU-Net
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
        axis, θ = self.get_rotation(data['spacing'])
        rot_inv = get_rotation_matrix(axis, -θ)
        crop_size = trans_info['size'] * trans_info['scale']


        # else:
        # else:


if __name__ == '__main__':
    R = get_rotation_matrix(np.array((1, 0, 0)), np.pi / 3)
    print(R)
