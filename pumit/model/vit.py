from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import einops
from timm.layers import DropPath
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.utils import checkpoint
from xformers import ops as xops

from mylib.models import load_ckpt
from mylib.types import NoWeightDecayParameter, param3_t, tuple2_t, tuple3_t
from monai.utils import ensure_tuple_rep

from pumit import sac
from pumit.sac import resample
from .rope import SpatialRotaryEmbedding

class PatchEmbed(nn.Module):
    def __init__(self, patch_size: param3_t[int] = 16, in_chans: int = 3, embed_dim: int = 768, adaptive: bool = True, flatten: bool = True):
        super().__init__()
        self.patch_size: tuple3_t[int] = ensure_tuple_rep(patch_size, 3)
        self.adaptive = adaptive
        self.proj = sac.InflatableInputConv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, adaptive=adaptive)
        self.flatten = flatten

    def forward(self, x: torch.Tensor, flatten: bool | None = None) -> torch.Tensor:
        flatten = self.flatten if flatten is None else flatten
        x = self.proj(x)
        if flatten:
            x = einops.rearrange(x, 'n c ... -> n (...) c')
        return x

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float = None,
        proj_drop: float = 0.,
        sub_ln: bool = False,
        rope: SpatialRotaryEmbedding | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim, _rem = divmod(dim, num_heads)
        assert _rem == 0
        self.scale = qk_scale or self.head_dim ** -0.5

        self.sub_ln = sub_ln
        if self.sub_ln:
            self.q_proj = nn.Linear(dim, dim, bias=False)
            self.k_proj = nn.Linear(dim, dim, bias=False)
            self.v_proj = nn.Linear(dim, dim, bias=False)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=False)

        if qkv_bias:
            self.q_bias = NoWeightDecayParameter(torch.zeros(dim))
            self.v_bias = NoWeightDecayParameter(torch.zeros(dim))
        else:
            self.register_parameter('q_bias', None)
            self.register_parameter('v_bias', None)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)

        self.rope = rope

    def expand_head(self, x: torch.Tensor):
        return einops.rearrange(x, 'n l (nh d) -> n l nh d', nh=self.num_heads)

    def apply_rope(self, x: torch.Tensor):
        if self.rope is not None:
            x = torch.cat([x[:, :1], self.rope(x[:, 1:])], dim=1)
        return x

    def forward(self, x: torch.Tensor):
        if self.sub_ln:
            q = self.expand_head(nnf.linear(x, self.q_proj.weight, self.q_bias))
            k = self.expand_head(nnf.linear(x, self.k_proj.weight))
            v = self.expand_head(nnf.linear(x, self.v_proj.weight, self.v_bias))
        else:
            qkv_bias = None
            if self.q_bias is not None:
                qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
            qkv = einops.rearrange(
                nnf.linear(input=x, weight=self.qkv.weight, bias=qkv_bias),
                'n l (qkv nh d) -> qkv n nh l d', qkv=3,
            )
            q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.apply_rope(q)
        k = self.apply_rope(k)
        x = einops.rearrange(
            # if using amp, v.dtype here will the autocast dtype
            xops.memory_efficient_attention(q.type_as(v), k.type_as(v), v),
            'n l nh d -> n l (nh d)',
        )
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        sub_ln: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.w1 = nn.Linear(in_features, hidden_features)
        self.w2 = nn.Linear(in_features, hidden_features)
        self.act = nn.SiLU()
        self.ffn_ln = nn.LayerNorm(hidden_features) if sub_ln else nn.Identity()
        self.w3 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = self.act(x1) * x2
        x = self.ffn_ln(hidden)
        x = self.w3(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float | int = 4.,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        proj_drop: float = 0.,
        drop_path: float = 0.,
        sub_ln: bool = False,
        rope: SpatialRotaryEmbedding = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, proj_drop, sub_ln, rope=rope)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden_dim, sub_ln=sub_ln)

    def forward(self, x: torch.Tensor):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

@dataclass
class Checkpoint:
    path: Path | None = None
    state_dict_key: str = 'state_dict'

class ViT(nn.Module):
    def __init__(
        self, *,
        in_channels: int = 3,
        patch_size: param3_t[int] = 16,
        adaptive_patch_embed: bool = True,
        embed_dim: int = 768,
        pos_embed_shape: tuple3_t[int],
        pretrained_pos_embed_shape: tuple2_t[int] | tuple3_t[int] | None = None,
        rope_rescale_shape: tuple3_t[int] = (-1, -1, -1),
        rope_base: tuple3_t[float] = (2333., 10000., 10000.),
        rope_merge_hw: bool = True,
        depth: int = 12,
        num_heads: int = 12,
        sub_ln: bool = True,
        mlp_ratio: float = 8 / 3,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        grad_ckpt: bool = False,
        patch_embed_grad_scale: float = 1.,
        pretrained_ckpt: Checkpoint,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim, adaptive_patch_embed, False)
        self.cls_token = NoWeightDecayParameter(torch.empty(1, 1, embed_dim))
        self.pos_embed = NoWeightDecayParameter(torch.empty(1, embed_dim, *pos_embed_shape))
        nn.init.trunc_normal_(self.cls_token, std=.02)
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.pretrained_pos_embed_shape = pretrained_pos_embed_shape
        self.pos_drop = nn.Dropout(drop_rate, inplace=True)
        self.rope = SpatialRotaryEmbedding(embed_dim // num_heads, rope_rescale_shape, rope_base, rope_merge_hw)
        self.num_heads = num_heads
        self.sub_ln = sub_ln
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks: Sequence[Block] | nn.ModuleList = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                proj_drop=drop_rate,
                drop_path=dpr[i],
                sub_ln=sub_ln,
                rope=self.rope,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.grad_ckpt = grad_ckpt
        self.patch_embed_grad_scale = patch_embed_grad_scale
        load_ckpt(self, pretrained_ckpt.path, pretrained_ckpt.state_dict_key)

    def prepare_seq_input(self, x: torch.Tensor):
        x = self.patch_embed(x)
        shape = x.shape[2:]
        x += resample(self.pos_embed, shape)
        x = self.pos_drop(x)
        x = torch.cat(
            [
                self.cls_token.expand(x.shape[0], -1, -1),
                einops.rearrange(x, 'n c ... -> n (...) c'),
            ],
            dim=1,
        )
        return x, shape

    def forward_features(self, x: torch.Tensor):
        x, shape = self.prepare_seq_input(x)
        self.rope.prepare(shape)
        states = []
        for block in self.blocks:
            if self.training and self.grad_ckpt:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
            states.append(x)
        self.rope.reset()
        return states

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)[-1]
        return self.norm(x)

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        for k in list(state_dict):
            if k.endswith('rope.freqs_cos') or k.endswith('rope.freqs_sin'):
                state_dict.pop(k)
        if (weight := state_dict.get('patch_embed.proj.weight')) is not None:
            d = self.patch_embed.patch_size[0]
            if weight.ndim == 4:
                # conv2d weight from EVA-02
                weight = nnf.interpolate(weight.float(), self.patch_embed.patch_size[1:], mode='bicubic')
                weight = einops.repeat(weight / d, 'co ci ... -> co ci d ...', d=d)
            else:
                # weight from PUMIT
                weight = einops.reduce(
                    weight,
                    'co ci (dr dc) ... -> co ci dr ...',
                    'sum',
                    dr=d,
                )
            state_dict['patch_embed.proj.weight'] = weight

        if (pos_embed := state_dict.get('pos_embed')) is not None:
            if pos_embed.ndim == 3:
                cls_pos_embed, pos_embed = pos_embed[:, 1], pos_embed[:, 1:]
                state_dict['cls_token'] += cls_pos_embed
                h, w = self.pretrained_pos_embed_shape
                pos_embed = einops.repeat(
                    pos_embed, '1 (h w) c -> 1 c d h w',
                    d=self.pos_embed.shape[2], h=h, w=w,
                )
            state_dict['pos_embed'] = resample(pos_embed, self.pos_embed.shape[2:])

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
