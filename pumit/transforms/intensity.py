import einops
import torch

__all__ = [
    'ensure_rgb',
    'RGB_TO_GRAY_WEIGHT',
    'rgb_to_gray',
]

def ensure_rgb(x: torch.Tensor, batched: bool = False) -> tuple[torch.Tensor, bool]:
    if x.shape[batched] == 3:
        is_rgb = True
    else:
        assert x.shape[batched] == 1
        maybe_batch = 'n' if batched else ''
        x = einops.repeat(x, f'{maybe_batch} 1 ... -> c ...', c=3)
        is_rgb = False
    return x, is_rgb

RGB_TO_GRAY_WEIGHT = (0.299, 0.587, 0.114)

def rgb_to_gray(x: torch.Tensor, batched: bool = False) -> torch.Tensor:
    # RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
    maybe_batch = 'n' if batched else ''
    return einops.rearrange(
        einops.einsum(x, x.new_tensor(RGB_TO_GRAY_WEIGHT), f'{maybe_batch} c ..., c ... -> {maybe_batch} ...'),
        f'{maybe_batch} ... -> {maybe_batch} 1 ...'
    )
