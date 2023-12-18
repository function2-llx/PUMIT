import einops
import torch

__all__ = [
    'ensure_rgb',
    'RGB_TO_GRAY_WEIGHT',
    'rgb_to_gray',
]

def ensure_rgb(x: torch.Tensor, enable: bool = True, batched: bool = False) -> torch.Tensor:
    if enable and x.shape[batched] != 3:
        assert x.shape[batched] == 1
        maybe_batch = 'n' if batched else ''
        x = einops.repeat(x, f'{maybe_batch} 1 ... -> c ...', c=3)
    return x

RGB_TO_GRAY_WEIGHT = (0.299, 0.587, 0.114)

def rgb_to_gray(x: torch.Tensor, batched: bool = False) -> torch.Tensor:
    # RGB to grayscale ref: https://www.itu.int/rec/R-REC-BT.601
    maybe_batch = 'n' if batched else ''
    return einops.rearrange(
        einops.einsum(x, x.new_tensor(RGB_TO_GRAY_WEIGHT), f'{maybe_batch} c ..., c ... -> {maybe_batch} ...'),
        f'{maybe_batch} ... -> {maybe_batch} 1 ...'
    )
