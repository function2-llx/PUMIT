import einops
import torch
from torch import nn

from luolib.types import param3_t
from monai.utils import ensure_tuple_rep
from pumt.conv import InflatableConv3d

class PatchEmbed(nn.Module):
    def __init__(self, patch_size: param3_t[int] = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = ensure_tuple_rep(patch_size, 3)
        self.proj = InflatableConv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor):
        # x = self.proj(x).flatten(2).transpose(1, 2)
        return einops.rearrange(self.proj(x), 'n c ... -> n (...) c')

class Block(nn.Module):

    def __init__(
        self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
        drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        window_size=None, attn_head_dim=None, use_decoupled_rel_pos_bias=False,
        depth=None,
        postnorm=False,
        deepnorm=False,
        subln=False,
        xattn=False,
        swiglu=False,
        naiveswiglu=False,
        rope=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size,
            use_decoupled_rel_pos_bias=use_decoupled_rel_pos_bias, attn_head_dim=attn_head_dim,
            deepnorm=deepnorm,
            subln=subln,
            norm_layer=norm_layer,
            xattn=xattn,
            rope=rope,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        if swiglu:
            self.mlp = xops.SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim
            )  # hidden_features: 2/3
        elif naiveswiglu:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer,
            )
        else:
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                subln=subln,
                norm_layer=norm_layer
            )

        if init_values is not None and init_values > 0:
            self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

        self.deepnorm = deepnorm
        if self.deepnorm:
            self.alpha = math.pow(2.0 * depth, 0.25)

        self.postnorm = postnorm

    def forward(self, x, rel_pos_bias=None, attn_mask=None):
        if self.gamma_1 is None:
            if self.postnorm:
                x = x + self.drop_path(
                    self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                )
                x = x + self.drop_path(self.norm2(self.mlp(x)))
            elif self.deepnorm:
                residual = x
                x = self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm1(x)

                residual = x
                x = self.mlp(x)
                x = self.drop_path(x)
                x = residual * self.alpha + x
                x = self.norm2(x)
            else:
                x = x + self.drop_path(
                    self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                )
                x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            if self.postnorm:
                x = x + self.drop_path(
                    self.gamma_1 * self.norm1(self.attn(x, rel_pos_bias=rel_pos_bias, attn_mask=attn_mask))
                )
                x = x + self.drop_path(self.gamma_2 * self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(
                    self.gamma_1 * self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias, attn_mask=attn_mask)
                )
                x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x

class VisionTransformerForMaskedImageModeling(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        init_std=0.02,
        drop_path_rate=0.,
        norm_layer=None,
        init_values=None,
        attn_head_dim=None,
        predict_feature_dim=768,
        grad_ckpt=False,
        stop_grad_conv1=False,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_shared_decoupled_rel_pos_bias=False,
        rope=False,
        postnorm=False,
        deepnorm=False,
        subln=False,
        xattn=False,
        swiglu=False,
        naiveswiglu=False,
        xavier_normal_init=False,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        if use_shared_decoupled_rel_pos_bias:
            assert self.rel_pos_bias is None
            self.rel_pos_bias = DecoupledRelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads
            )

        if rope:
            half_head_dim = embed_dim // num_heads // 2
            hw_seq_len = img_size // patch_size
            self.rope = VisionRotaryEmbeddingFast(
                dim=half_head_dim,
                pt_seq_len=hw_seq_len,
            )
        else:
            self.rope = None

        self.subln = subln
        self.swiglu = swiglu
        self.naiveswiglu = naiveswiglu

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                    window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                    attn_head_dim=attn_head_dim,
                    depth=depth,
                    postnorm=postnorm,
                    deepnorm=deepnorm,
                    subln=subln,
                    xattn=xattn,
                    swiglu=swiglu,
                    naiveswiglu=naiveswiglu,
                    rope=self.rope,
                )
                for i in range(depth)]
        )

        self.norm = norm_layer(embed_dim) if not deepnorm else nn.Identity()

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, predict_feature_dim)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)

        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)

        if xavier_normal_init:
            self.apply(self._xavier_normal_init)
            w = self.patch_embed.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        else:  # ori BEiT init
            self.apply(self._init_weights)
            self.fix_init_weight()

        if postnorm:
            self._reinit_respostnorm_ln()

        if deepnorm:
            init_scale = math.pow(8.0 * depth, 0.25)
            for name, p in self.named_parameters():
                if (
                    'mlp.fc' in name
                    or 'mlp.w' in name
                    or 'attn.proj' in name
                    or 'attn.v_proj' in name
                ):
                    print('deepnorm rescale:', name, '/', init_scale)
                    p.data.div_(init_scale)

        if subln:
            init_scale = math.sqrt(math.log(depth * 2))
            for name, p in self.named_parameters():
                if (
                    'mlp.fc' in name
                    or 'mlp.w' in name
                    or 'attn.proj' in name
                    or 'attn.v_proj' in name
                ):
                    print('subln rescale:', name, 'x', init_scale)
                    p.data.mul_(init_scale)

        self.grad_ckpt = grad_ckpt
        self.stop_grad_conv1 = stop_grad_conv1

    def _reinit_respostnorm_ln(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            if self.swiglu or self.naiveswiglu:
                rescale(layer.mlp.w3.weight.data, layer_id + 1)
            else:
                rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _xavier_normal_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_final_patch_size(self):
        return self.patch_embed.patch_size

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, bool_masked_pos):
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)

        if self.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1

        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        if self.grad_ckpt:
            for i in range(len(self.blocks)):
                x = torch.utils.checkpoint.checkpoint(self.blocks[i], x, rel_pos_bias)
        else:
            for blk in self.blocks:
                x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward(self, image_input, bool_masked_pos):
        image_features = self.forward_features(image_input, bool_masked_pos)
        image_features = image_features[:, 1:]
        image_features = self.lm_head(image_features[bool_masked_pos])

        return image_features
