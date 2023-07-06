import einops
import torch

def ensure_rgb(x: torch.Tensor, enable: bool = True) -> torch.Tensor:
    if enable and x.shape[1] != 3:
        assert x.shape[1] == 1
        x = einops.repeat(x, 'n 1 ... -> n c ...', c=3)
    return x

def rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    return einops.einsum(
        x, x.new_tensor([0.299, 0.587, 0.114]),
        'n c ..., c -> n ...',
    )[:, None]
