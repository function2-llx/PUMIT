from functools import cache

import einops
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from luolib.types import tuple3_t

def broadcast_cat(a: torch.Tensor, b: torch.Tensor, dim: int = -1):
    tensors = torch.broadcast_tensors(a, b)
    return torch.cat(tensors, dim)

def rotate_half(x: torch.Tensor):
    x1, x2 = einops.rearrange(x, '... (d r) -> r ... d', r=2)
    return einops.rearrange([-x2, x1], 'r ... d -> ... (d r)')

class SpatialRotaryEmbedding(nn.Module):
    cos_θ: torch.Tensor | None
    sin_θ: torch.Tensor | None

    @property
    def ω(self) -> list[torch.Tensor]:
        # let's wish we can have nn.BufferList soon: https://github.com/pytorch/pytorch/issues/37386 https://github.com/pytorch/pytorch/issues/35735
        return [self.get_buffer(f'ω{i}') for i in range(3)]

    def __init__(self,
        dim: int,
        rescale_shape: tuple3_t[int] = (8, 16, 16),
        base: tuple3_t[float] = (23333, 10000, 10000),
        merge_hw: bool = True,
    ):
        super().__init__()
        assert dim & 1 == 0
        self.rescale_shape = rescale_shape
        self.merge_hw = merge_hw
        if merge_hw:
            # compatible with EVA-02
            dim = [dim, dim >> 1, dim >> 1]
        else:
            dim = [dim, dim, dim]
        for i in range(3):
            self.register_buffer(f'ω{i}', torch.pow(base[i], -torch.arange(0, dim[i], 2) / dim[i]), False)
        self.reset()

    @cache
    def get_θ(self, shape: tuple3_t[int]):
        θ = [
            torch.outer(
                torch.arange(shape[i], device=self.ω[i].device) * self.rescale_shape[i] / shape[i],
                self.ω[i],
            )
            for i in range(3)
        ]
        if self.merge_hw:
            θ_hw = broadcast_cat(θ[1][:, None], θ[2][None, :])
        else:
            θ_hw = θ[1][:, None] + θ[2][None, :]
        return einops.rearrange(θ[0][:, None, None] + θ_hw[None, :], '... d -> (...) d')

    def prepare_batch(self, batch: list[torch.Tensor]):
        shapes = [tuple(x.shape[1:]) for x in batch]
        θ = pad_sequence([self.get_θ(shape) for shape in shapes], True)
        self.cos_θ = einops.repeat(θ.cos(), 'n l d -> n l 1 (d r)', r=2)
        self.sin_θ = einops.repeat(θ.sin(), 'n l d -> n l 1 (d r)', r=2)

    def forward(self, x: torch.Tensor):
        return x * self.cos_θ + rotate_half(x) * self.sin_θ

    def reset(self):
        self.cos_θ = None
        self.sin_θ = None
