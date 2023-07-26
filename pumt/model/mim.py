from collections.abc import Sequence
from typing import Any

import einops
from lightning import LightningModule
import torch
from torch import nn
from torch.utils import checkpoint

from luolib.types import NoWeightDecayParameter
from .vit import ViT
from ..conv import SpatialTensor
from ..tokenizer import VQVAEModel

class ViTForMIM(ViT, LightningModule):
    # tokenizer typed as dict https://github.com/omni-us/jsonargparse/issues/330
    def __init__(self, *args, tokenizer: dict, mask_ratio: float = 0.85, mask_layer_ids: Sequence[int], **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = VQVAEModel.from_pretrained(**tokenizer).eval()
        assert self.tokenizer.stride == self.patch_embed.patch_size
        self.tokenizer.requires_grad_(False)
        self.mask_token = NoWeightDecayParameter(torch.empty(1, 1, self.embed_dim))
        self.mask_ratio = mask_ratio
        self.mask_layer_ids = set(mask_layer_ids)
        self.mim_head = nn.Linear(self.embed_dim, self.tokenizer.codebook_size)

    def state_dict(self, *args, **kwargs):
        return {
            k: v for k, v in super().state_dict(*args, **kwargs)
            if k.startswith('tokenizer.')
        }

    def training_step(self, batch: tuple[torch.Tensor, int], *args, **kwargs):
        x = SpatialTensor(*batch)
        token_ids = self.tokenizer.tokenize(x)
        x, shape = self.apply_patch_embed(x)
        batch_size, seq_len = x.shape[:2]
        seq_len -= 1  # exclude cls token
        num_visible_patches = int(seq_len * (1 - self.mask_ratio))
        visible_idx, _ = x.new_ones(batch_size, seq_len).multinomial(num_visible_patches).sort()
        self.rope.prepare(shape, visible_idx, self.num_heads)
        visible_idx = einops.repeat(
            visible_idx + 1,
            'n l -> n l d', d=x.shape[-1],
        ).contiguous()
        for i, block in enumerate(self.blocks):
            if self.grad_ckpt:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        self.rope.reset()
        return self.norm(x)
