import einops
import numpy as np
from timm.layers import DropPath
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.nn.utils.rnn import pad_sequence
from torch.utils import checkpoint
from xformers import ops as xops
from xformers.ops.fmha import BlockDiagonalMask

from luolib.types import NoWeightDecayParameter, param3_t, tuple2_t, tuple3_t
from monai.utils import ensure_tuple_rep

from pumt.conv import InflatableConv3d, SpatialTensor
from .rope import SpatialRotaryEmbedding

class PatchEmbed(nn.Module):
    def __init__(self, patch_size: param3_t[int] = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = ensure_tuple_rep(patch_size, 3)
        self.proj = InflatableConv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: SpatialTensor, flatten: bool = False) -> torch.Tensor:
        x = self.proj(x).as_tensor()
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

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor):
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
            xops.memory_efficient_attention(q.type_as(v), k.type_as(v), v, attn_bias),
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

    def forward(self, x: torch.Tensor, attn_bias: BlockDiagonalMask):
        x = x + self.drop_path(self.attn(self.norm1(x), attn_bias))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ViT(nn.Module):
    def __init__(
        self,
        in_chans: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        abs_pos_embed_shape: tuple3_t[int] = (8, 16, 16),
        pretrained_abs_pos_embed_shape: tuple2_t[int] | tuple3_t[int] | None = None,
        rope_rescale_shape: tuple3_t[int] = (8, 16, 16),
        rope_base: tuple3_t[float] = (23333., 10000., 10000.),
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
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.cls_token = NoWeightDecayParameter(torch.empty(1, embed_dim))
        # self.mask_token = NoWeightDecayParameter(torch.empty(1, 1, embed_dim))
        self.pos_embed = NoWeightDecayParameter(torch.empty(1, embed_dim, *abs_pos_embed_shape))
        self.pretrained_abs_pos_embed_shape = pretrained_abs_pos_embed_shape
        self.pos_drop = nn.Dropout(drop_rate, inplace=True)
        self.rope = SpatialRotaryEmbedding(embed_dim // num_heads, rope_rescale_shape, rope_base, rope_merge_hw)
        self.num_heads = num_heads
        self.sub_ln = sub_ln
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
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

    def patch_embed_batch(self, batch: list[torch.Tensor]) -> list[torch.Tensor]:
        ret = []
        for x in batch:
            x = self.patch_embed(x[None])
            x += nnf.interpolate(
                nnf.interpolate(
                    self.pos_embed,
                    tuple(np.minimum(self.pos_embed.shape[2:], x.shape[2:])),
                    mode='area'
                ),
                x.shape[2:],
                mode='trilinear',
            )
            ret.append(self.pos_drop(x)[0])
        return ret

    def flatten_batch(self, batch: list[torch.Tensor]):
        batch = [
            torch.cat(
                [
                    self.cls_token,
                    einops.rearrange(x, 'c ... -> (...) c')
                ],
                dim=0,
            )
            for x in batch
        ]
        x = pad_sequence(batch, True)
        batch_size, max_len = x.shape[:2]
        dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None
        # cutlassF requires 8 | stride[-2]
        attn_bias = x.new_full(
            (x.shape[0], x.shape[1], (x.shape[1] - 1 >> 3) + 1 << 3),
            -torch.inf,
            dtype=dtype,
        )
        for i in range(batch_size):
            seq_len = batch[i].shape[0]
            attn_bias[i, :seq_len, :seq_len] = 0
        attn_bias = einops.repeat(attn_bias, 'n ... -> n nh ...', nh=self.num_heads)[..., :x.shape[1]]
        return x, attn_bias

    def prepare_batch(self, batch: list[torch.Tensor]):
        batch = self.patch_embed_batch(batch)
        self.rope.prepare_batch(batch)
        x, attn_bias = self.flatten_batch(batch)
        x = x.detach() * (1 - self.patch_embed_grad_scale) + x * self.patch_embed_grad_scale
        return x, attn_bias

    def forward_features(self, batch: list[torch.Tensor]):
        x, attn_bias = self.prepare_batch(batch)
        for block in self.blocks:
            if self.grad_ckpt:
                x = checkpoint.checkpoint(block, x, attn_bias)
            else:
                x = block(x, attn_bias)
        return self.norm(x)

    def forward(self, batch: list[torch.Tensor]):
        return self.forward_features(batch)

class ViTForMIM(ViT):
    pass
    # def __init__(
    #     self,
    #     in_chans: int = 3,
    #     patch_size: int = 16,
    #     embed_dim: int = 768,
    #     abs_pos_embed_shape: tuple3_t[int] = (8, 16, 16),
    #     pretrained_abs_pos_embed_shape: tuple2_t[int] | tuple3_t[int] | None = None,
    #     depth: int = 12,
    #     num_heads: int = 12,
    #     mlp_ratio: float = 4.,
    #     qkv_bias: bool = True,
    #     qk_scale: float | None = None,
    #     drop_rate: float = 0.,
    #     drop_path_rate: float = 0.,
    #     predict_dim: int = 768,
    #     grad_ckpt: bool = False,
    #     patch_embed_grad_scale: float = 1.,
    #     sub_ln: bool = False,
    # ):
    #     super().__init__()
    #     self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
    #     self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
    #     self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    #     self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
    #     self.pos_embed = nn.Parameter(torch.zeros(1, np.prod(abs_pos_embed_shape) + 1, embed_dim))
    #     self.pretrained_abs_pos_embed_shape = pretrained_abs_pos_embed_shape
    #     self.pos_drop = nn.Dropout(drop_rate)
    #     # half_head_dim = embed_dim // num_heads // 2
    #     # hw_seq_len = img_size // patch_size
    #     self.rope = SpatialRotaryEmbedding(
    #         embed_dim // num_heads,
    #         abs_pos_embed_shape,
    #         # merge_hw=True,
    #         # dim=half_head_dim,
    #         # pt_seq_len=hw_seq_len,
    #     )
    #     self.sub_ln = sub_ln
    #
    #     dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
    #     self.blocks = nn.ModuleList(
    #         [
    #             Block(
    #                 dim=embed_dim,
    #                 num_heads=num_heads,
    #                 mlp_ratio=mlp_ratio,
    #                 qkv_bias=qkv_bias,
    #                 qk_scale=qk_scale,
    #                 proj_drop=drop_rate,
    #                 drop_path=dpr[i],
    #                 sub_ln=sub_ln,
    #                 rope=self.rope,
    #             )
    #             for i in range(depth)
    #         ]
    #     )
    #
    #     self.norm = nn.LayerNorm(embed_dim)
    #     self.lm_head = nn.Linear(embed_dim, predict_dim)
    #     self.grad_ckpt = grad_ckpt
    #     self.patch_embed_grad_scale = patch_embed_grad_scale
    #
    # def no_weight_decay(self):
    #     return {'pos_embed', 'cls_token'}
    #
    # def get_num_layers(self):
    #     return len(self.blocks)
    #
    # def forward_features(self, x: torch.Tensor, mask: torch.Tensor):
    #     x = self.patch_embed(x, bool_masked_pos=mask)
    #     x = x.detach() * (1 - self.patch_embed_grad_scale) + x * self.patch_embed_grad_scale
    #
    #     batch_size, seq_len = x.shape[:2]
    #
    #     cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    #     mask_token = self.mask_token.expand(batch_size, seq_len, -1)
    #
    #     w = mask.unsqueeze(-1).type_as(mask_token)
    #     x = x * (1 - w) + mask_token * w
    #
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     if self.pos_embed is not None:
    #         x = x + self.pos_embed
    #     x = self.pos_drop(x)
    #
    #     rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
    #
    #     if self.grad_ckpt:
    #         for i in range(len(self.blocks)):
    #             x = torch.utils.checkpoint.checkpoint(self.blocks[i], x, rel_pos_bias)
    #     else:
    #         for blk in self.blocks:
    #             x = blk(x, rel_pos_bias=rel_pos_bias)
    #
    #     return self.norm(x)
    #
    # def forward(self, image_input, bool_masked_pos):
    #     image_features = self.forward_features(image_input, bool_masked_pos)
    #     image_features = image_features[:, 1:]
    #     image_features = self.lm_head(image_features[bool_masked_pos])
    #
    #     return image_features
