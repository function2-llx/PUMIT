import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.models.blocks import InflatableConv3d
from luolib.utils import channel_last, channel_first
from monai.config import PathLike

@dataclass
class VectorQuantizerOutput:
    z_q: torch.Tensor
    loss: torch.Tensor
    index: torch.Tensor
    probs: torch.Tensor | None = None

# modified from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: Literal['nearst', 'gumbel', 'soft'],
        beta: float = 0.25,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mode = mode
        if mode == 'nearest':
            self.beta = beta
            self.register_module('proj', None)
        else:
            # calculate categorical distribution over embeddings
            # use conv here to match the implementation in VQGAN's repository
            self.proj = InflatableConv3d(embedding_dim, num_embeddings, 1)

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        # load pre-trained Gumbel VQGAN
        if (weight := state_dict.pop(f'{prefix}embed.weight', None)) is not None:
            state_dict[f'{prefix}embedding.weight'] = weight
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, z: torch.Tensor):
        if self.mode == 'nearest':
            z = channel_last(z).contiguous()
            # distances from z to embeddings, exclude |z|^2
            d = torch.sum(self.embedding.weight ** 2, dim=1) \
                - 2 * einops.einsum(z, self.embedding.weight, '... d, ne d -> ... ne')
            index = d.argmin(dim=-1)
            z_q = self.embedding(index)
            loss = ((z_q.detach() - z) ** 2).mean() + self.beta * ((z_q - z.detach()) ** 2).mean()
            # preserve gradients
            z_q = z + (z_q - z).detach()
            # reshape back to match original input shape
            z_q = channel_first(z_q).contiguous()
            return VectorQuantizerOutput(z_q, loss, index)
        else:
            logits = self.proj(z)
            probs = logits.softmax(dim=1)
            kld = -((logits - logits.logsumexp(dim=1, keepdim=True)) * probs).sum(dim=1)
            loss = kld.mean()
            if self.mode == 'gumbel':
                if self.training:
                    one_hot_prob = nnf.gumbel_softmax(logits, hard=True, dim=1)
                    index = one_hot_prob.argmax(dim=1)
                    z_q = einops.einsum(one_hot_prob, self.embedding.weight, 'n ne ..., ne d -> n d ...')
                else:
                    index = probs.argmax(dim=1)
                    z_q = channel_first(self.embedding(index)).contiguous()
            else:
                index = probs
                z_q = einops.einsum(probs, self.embedding.weight, 'n ne ..., ne d -> n d ...')
            return VectorQuantizerOutput(z_q, loss, index, probs)

    def get_codebook_entry(self, index: torch.Tensor):
        if self.mode == 'soft':
            # assume soft index to be channel first
            z_q = einops.einsum(index, self.embedding.weight, 'n ne ..., ne d -> n d ...')
        else:
            z_q = self.embedding(index)
            z_q = channel_first(z_q).contiguous()
        return z_q
