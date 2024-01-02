import einops
import torch

from luolib.utils import RGB_TO_GRAY_WEIGHT

__all__ = [
    'ensure_rgb',
    'rgb_to_gray',
]

def ensure_rgb(x: torch.Tensor, batched: bool = False) -> tuple[torch.Tensor, bool]:
    if x.shape[batched] == 3:
        not_rgb = False
    else:
        assert x.shape[batched] == 1
        maybe_batch = 'n' if batched else ''
        x = einops.repeat(x, f'{maybe_batch} 1 ... -> c ...', c=3)
        not_rgb = True
    return x, not_rgb

def rgb_to_gray(x: torch.Tensor, batched: bool = False) -> torch.Tensor:
    """x need not be scaled of [0, 1] since sum(RGB_TO_GRAY_WEIGHT) ≈ 1"""
    # RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
    maybe_batch = 'n' if batched else ''
    return einops.rearrange(
        einops.einsum(x, x.new_tensor(RGB_TO_GRAY_WEIGHT), f'{maybe_batch} c ..., c ... -> {maybe_batch} ...'),
        f'{maybe_batch} ... -> {maybe_batch} 1 ...'
    )
