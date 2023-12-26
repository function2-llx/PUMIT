import torch

__all__ = [
    'smooth_image',
    'smooth_image_inv',
]

eps = 1e-2

def smooth_image(x: torch.Tensor) -> torch.Tensor:
    return (1 - 2 * eps) * x + eps

def smooth_image_inv(x: torch.Tensor) -> torch.Tensor:
    return ((x - eps) / (1 - 2 * eps)).clamp_(min=0, max=1)
