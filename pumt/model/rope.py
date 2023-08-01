from functools import lru_cache

import einops
import torch
from torch import nn

from luolib.types import tuple3_t
from monai.utils import fall_back_tuple

def broadcast_cat(a: torch.Tensor, b: torch.Tensor, dim: int = -1):
    tensors = torch.broadcast_tensors(a, b)
    return torch.cat(tensors, dim)

def rotate_half(x: torch.Tensor):
    x1, x2 = einops.rearrange(x, '... (d r) -> r ... d', r=2)
    return einops.rearrange([-x2, x1], 'r ... d -> ... (d r)')

class SpatialRotaryEmbedding(nn.Module):
    @property
    def ω(self) -> list[torch.Tensor]:
        # let's wish we can have nn.BufferList soon: https://github.com/pytorch/pytorch/issues/37386 https://github.com/pytorch/pytorch/issues/35735
        return [self.get_buffer(f'ω{i}') for i in range(3)]

    def __init__(self,dim: int, rescale_shape: tuple3_t[int], base: tuple3_t[float], merge_hw: bool = True):
        super().__init__()
        self.dim = dim
        self.rescale_shape = rescale_shape
        self.merge_hw = merge_hw
        if merge_hw:
            # compatible with EVA-02
            assert dim & 3 == 0
            dim >>= 1
        else:
            assert dim & 1 == 0
        for i in range(3):
            if merge_hw and i > 0:
                exp = -torch.arange(0, dim, 2) / dim
            else:
                exp = -torch.arange(1, dim, 2) / dim
                if i == 0:
                    exp = einops.repeat(exp, '... -> (r ...)', r=2)
            self.register_buffer(f'ω{i}', torch.pow(base[i], exp), False)
        self.reset()

    @lru_cache
    def get_rotation(self, spatial_shape: tuple3_t[int]) -> tuple[torch.Tensor, torch.Tensor]:
        rescale_shape = fall_back_tuple(self.rescale_shape, spatial_shape)
        θ = [
            torch.outer(
                torch.arange(spatial_shape[i], device=self.ω[i].device) * rescale_shape[i] / spatial_shape[i],
                self.ω[i],
            )
            for i in range(3)
        ]
        if self.merge_hw:
            θ_hw = broadcast_cat(θ[1][:, None], θ[2][None, :])
        else:
            θ_hw = θ[1][:, None] + θ[2][None, :]
        θ = θ[0][:, None, None] + θ_hw[None, :]
        θ = einops.rearrange(θ, '... d -> (...) d')
        return θ.cos(), θ.sin()

    def prepare(self, spatial_shape: tuple3_t[int], visible_idx: torch.Tensor | None = None):
        cos_θ, sin_θ = self.get_rotation(spatial_shape)
        self.cos_θ = einops.repeat(cos_θ, 'l d -> l 1 (d r)', r=2)
        self.sin_θ = einops.repeat(sin_θ, 'l d -> l 1 (d r)', r=2)
        if visible_idx is not None:
            visible_idx = einops.repeat(visible_idx, 'n l -> n l d', d=self.dim >> 1).contiguous()
            batch_size = visible_idx.shape[0]
            def gather_visible(x: torch.Tensor):
                x = einops.repeat(x, 'l d -> n l d', n=batch_size)
                x = x.gather(dim=1, index=visible_idx)
                return einops.repeat(x, 'n l d -> n l 1 (d r)', r=2)
            self.cos_θ_visible = gather_visible(cos_θ)
            self.sin_θ_visible = gather_visible(sin_θ)

    def forward(self, x: torch.Tensor):
        if x.shape[1] == self.cos_θ.shape[0]:
            cos_θ = self.cos_θ
            sin_θ = self.sin_θ
        else:
            cos_θ = self.cos_θ_visible
            sin_θ = self.sin_θ_visible
        return x * cos_θ + rotate_half(x) * sin_θ

    def reset(self):
        self.cos_θ = None
        self.sin_θ = None
        self.cos_θ_visible = None
        self.sin_θ_visible = None
