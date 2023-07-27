from collections.abc import Sequence

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
    def __init__(self, *args, tokenizer: VQVAEModel, mask_ratio: float = 0.85, mask_layer_ids: Sequence[int], **kwargs):
        """mask_layer_ids: layers that include mask tokens as input"""
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        assert self.tokenizer.stride == self.patch_embed.patch_size
        self.tokenizer.requires_grad_(False)
        self.mask_token = NoWeightDecayParameter(torch.empty(1, 1, self.embed_dim))
        self.mask_ratio = mask_ratio
        self.mask_layer_ids = set(mask_layer_ids)
        self.mim_head = nn.Linear(self.embed_dim, self.tokenizer.codebook_size)
        self.mim_loss = nn.CrossEntropyLoss()

    def state_dict(self, *args, **kwargs):
        return {
            k: v for k, v in super().state_dict(*args, **kwargs)
            if k.startswith('tokenizer.')
        }

    def forward(self, x: SpatialTensor, visible_idx: torch.Tensor | None = None):
        if visible_idx is None:
            return super().forward(x)
        x, spatial_shape = self.prepare_seq_input(x)
        self.rope.prepare(spatial_shape, visible_idx)
        batch_size, seq_len, dim = x.shape
        seq_len -= 1  # exclude cls token
        visible_idx = einops.repeat(visible_idx, 'n l -> n l d', d=dim).contiguous()
        x = torch.cat([
                self.cls_token.expand(batch_size, 1, -1),
                self.mask_token.expand(batch_size, seq_len, -1).scatter(dim=1, index=visible_idx, src=x),
            ],
            dim=1,
        )
        # cls token should always be visible
        visible_idx = torch.cat([visible_idx.new_zeros(batch_size, 1, dim), visible_idx + 1], dim=1)
        for i, block in enumerate(self.blocks):
            x_layer = x if i in self.mask_layer_ids else x.gather(dim=1, index=visible_idx)
            if self.grad_ckpt:
                x_layer = checkpoint.checkpoint(block, x_layer)
            else:
                x_layer = block(x_layer)
            if i in self.mask_layer_ids:
                x = x_layer
            else:
                x.scatter_(dim=1, index=visible_idx, src=x_layer)
        self.rope.reset()
        return self.norm(x)

    def training_step(self, batch: tuple[torch.Tensor, int], *args, **kwargs):
        x = SpatialTensor(*batch)
        token_ids = self.tokenizer.tokenize(x)
        batch_size, seq_len = token_ids.shape[:2]
        num_visible_patches = int(seq_len * (1 - self.mask_ratio))
        visible_idx, _ = token_ids.new_ones(batch_size, seq_len).multinomial(num_visible_patches).sort()
        hidden_states = self(x, visible_idx)[:, 1:]
        masked_mask = hidden_states.new_ones(batch_size, seq_len, dtype=torch.bool).scatter_(dim=1, index=visible_idx, value=False)
        token_probs = self.mim_head(hidden_states[masked_mask])
        return self.mim_loss(token_probs, token_ids[masked_mask])
