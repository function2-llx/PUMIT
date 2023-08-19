from __future__ import annotations

from functools import lru_cache
import itertools as it
from typing import Mapping, Sequence

from einops import rearrange
import numpy as np
from timm.models.layers import trunc_normal_
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.utils import checkpoint

from monai.networks.blocks import MLPBlock
from monai.networks.layers import DropPath, get_norm_layer
from monai.utils import ensure_tuple_rep

from mylib.models.adaptive_resampling import AdaptiveDownsampling
from mylib.models.layers import Act, Norm
from mylib.utils import channel_first, channel_last

# used in blocks, x is channel last
def window_partition(x: torch.Tensor, window_size: Sequence[int]):
    num_windows = {
        f'n{i}': x.shape[i + 1] // ws
        for i, ws in enumerate(window_size)
    }
    return rearrange(x, 'n (n0 w0) (n1 w1) (n2 w2) c -> (n n0 n1 n2) (w0 w1 w2) c', **num_windows), num_windows

def window_reverse(x: torch.Tensor, window_size: Sequence[int], num_windows: Mapping[str, int]):
    return rearrange(x, '(n n0 n1 n2) (w0 w1 w2) c -> n (n0 w0) (n1 w1) (n2 w2) c', **{
        **num_windows,
        **{
            f'w{i}': ws
            for i, ws in enumerate(window_size)
        }
    })

class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            max_window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = np.power(head_dim, -0.5)

        self.relative_position_bias_table = nn.Parameter(
            torch.empty(
                num_heads,
                np.product(2 * np.array(max_window_size) - 1),
            )
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.BoolTensor | None, relative_position_index: torch.LongTensor):
        qkv = rearrange(self.qkv(x), 'n l (qkv nh ch) -> qkv n nh l ch', qkv=3, nh=self.num_heads).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = rearrange(
            self.relative_position_bias_table[:, relative_position_index],
            'nh (l1 l2) -> nh l1 l2',
            l1=x.shape[1],
        )
        attn = attn + relative_position_bias

        if mask is not None:
            rearrange(attn, '(n nw) nh l1 l2 -> n nh nw l1 l2', nw=mask.shape[0]).masked_fill_(~mask, -torch.inf)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn).to(v.dtype)
        x = rearrange(attn @ v, 'n nh l d -> n l (nh d)')
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    downsamplings: nn.ModuleList | Sequence[AdaptiveDownsampling]

    def __init__(
        self,
        dim: int,
        num_heads: int,
        max_window_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: tuple | str | None = Act.GELU,
        norm_layer: tuple | str | None = Norm.LAYER,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            num_heads: number of attention heads.
            max_window_size: local window size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
        """

        super().__init__()
        self.num_heads = num_heads
        self.norm1 = get_norm_layer(norm_layer, 3, dim)
        self.attn = WindowAttention(
            dim,
            num_heads,
            max_window_size,
            qkv_bias,
            attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = get_norm_layer(norm_layer, 3, dim)
        self.mlp = MLPBlock(
            hidden_size=dim,
            mlp_dim=int(dim * mlp_ratio),
            act=act_layer,
            dropout_rate=drop,
            dropout_mode="swin",
        )

    def forward(
        self,
        x: torch.Tensor,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        attn_mask: torch.BoolTensor,
        relative_position_index: torch.LongTensor,
    ):
        shortcut = x
        x = self.norm1(x)

        shift_size = np.array(shift_size)
        if np.any(shift_size):  # a little faster
            x = x.roll(tuple(shift_size), dims=(1, 2, 3))   # shift towards pad
        parted_x, num_windows = window_partition(x, window_size)
        attn_x = self.attn.forward(parted_x, attn_mask, relative_position_index)
        x = window_reverse(attn_x, window_size, num_windows)
        if np.any(shift_size):
            x = x.roll(tuple(-shift_size), dims=(1, 2, 3))
        x = shortcut + self.drop_path(x)
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

# TODO: there are at most 8 kinds of attention mask for 3D
def compute_attn_mask(img_mask: torch.Tensor, window_size: Sequence[int]) -> torch.BoolTensor:
    mask_windows, _ = window_partition(img_mask[None, ..., None], window_size)
    mask_windows = mask_windows.squeeze(-1)  # squeeze the dummy channel
    attn_mask = mask_windows[:, None, :] == mask_windows[:, :, None]
    return attn_mask    # type: ignore

@lru_cache(maxsize=128)
def compute_relative_position_index(window_size: tuple[int, ...]) -> torch.LongTensor:
    window_size = torch.tensor(window_size)
    # create coordinates of w0 * w1 * w2, 3
    coords_flatten = torch.cartesian_prod(*map(torch.arange, window_size))
    # compute axis-wise relative distance & shift to start from 0
    relative_coords = coords_flatten[:, None] - coords_flatten[None, :] + window_size - 1
    # flatten 3D coordinates to 1D (which might be faster & more convenient for indexing)
    # PyTorch does not support negative stride yet: https://github.com/pytorch/pytorch/issues/59786
    relative_coords[..., :-1] *= (2 * window_size[1:] - 1).flip(dims=[0]).cumprod(dim=0).flip(dims=[0])
    relative_position_index = relative_coords.sum(dim=-1)
    return relative_position_index.view(-1)

class SwinLayer(nn.Module):
    """
    better padding & more concise implementation
    """

    blocks: Sequence[SwinTransformerBlock] | nn.ModuleList

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        max_window_size: int | Sequence[int],
        drop_path_rates: list[float] | float = 0.,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: tuple | str | None = Norm.LAYER,
        use_checkpoint: bool = False,
    ) -> None:
        """
        Args:
            dim (int): Number of input channels.
            depth: number of blocks
            num_heads: number of attention heads.
            max_window_size: local window size.
            drop_path_rates: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.max_window_size = np.array(ensure_tuple_rep(max_window_size, 3))
        self.shift_size = self.max_window_size // 2
        self.use_checkpoint = use_checkpoint
        if isinstance(drop_path_rates, float):
            drop_path_rates = [drop_path_rates] * depth
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim,
                    num_heads,
                    self.max_window_size,
                    mlp_ratio,
                    qkv_bias,
                    drop,
                    attn_drop,
                    drop_path_rate,
                    norm_layer=norm_layer,
                )
                for drop_path_rate in drop_path_rates
            ]
        )

    def forward(self, x: torch.Tensor):
        spatial_shape = np.array(x.shape[2:])
        window_size = np.minimum(self.max_window_size, spatial_shape)
        relative_position_index = compute_relative_position_index(tuple(window_size))
        pad_size = (window_size - spatial_shape % window_size) % window_size
        x = nnf.pad(x, tuple(np.ravel([np.zeros_like(pad_size), np.flip(pad_size)], 'F')))

        img_mask = torch.zeros_like(x[0, 0], dtype=torch.int8)
        if np.any(pad_size):
            img_mask[tuple(map(slice, spatial_shape))] = 1
            attn_mask = compute_attn_mask(img_mask, window_size)
        else:
            attn_mask = None

        shift_size = np.where(window_size < spatial_shape, self.shift_size, 0)    # type: ignore
        if np.any(shift_size):
            for i, slices in enumerate(
                it.product(*[
                    [slice(s, None), slice(s)] if s else [slice(None)]
                    for s in shift_size
                ])
            ):
                img_mask[slices] = i + 1
            shift_attn_mask = compute_attn_mask(img_mask, window_size)
        else:
            shift_attn_mask = attn_mask

        x = channel_last(x).contiguous()
        for block, block_shift_size, block_attn_mask in zip(
            self.blocks,
            it.cycle([np.zeros_like(shift_size), shift_size]),
            it.cycle([attn_mask, shift_attn_mask]),
        ):
            if self.training and self.use_checkpoint:
                x = checkpoint.checkpoint(block, x, window_size, block_shift_size, block_attn_mask, relative_position_index)
            else:
                x = block.forward(x, window_size, block_shift_size, block_attn_mask, relative_position_index)
        return channel_first(x)

    def extra_repr(self) -> str:
        return f'max_window_size={self.max_window_size.tolist()}'
