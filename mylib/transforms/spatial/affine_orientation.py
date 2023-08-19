import nibabel as nib
import torch

from monai import transforms as mt
from monai.data import MetaTensor

class AffineOrientation(mt.Transform):
    def __init__(self, original_affine: torch.Tensor):
        self.orientation = mt.Orientation(nib.orientations.aff2axcodes(original_affine))

    def __call__(self, x: MetaTensor):
        return self.orientation(x)
