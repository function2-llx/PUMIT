from dataclasses import dataclass
from typing import Literal

import einops
from lightning import Fabric
import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from mylib.utils import channel_first, channel_last

@dataclass
class VectorQuantizerOutput:
    z_q: torch.Tensor
    index: torch.Tensor
    loss: torch.Tensor
    diversity: float
    logits: torch.Tensor | None = None
    entropy: torch.Tensor | None = None

# modified from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: Literal['nearest', 'gumbel', 'soft'],
        in_channels: int | None = None,
        beta: float = 0.25,
        hard_gumbel: bool = True,
        t_min: float = 1e-6,
        t_max: float = 0.9,
    ):
        super().__init__()
        self.mode = mode
        in_channels = in_channels or embedding_dim
        if mode == 'nearest':
            self.beta = beta
            self.register_module('proj', None)
        else:
            # calculate categorical distribution over embeddings
            self.proj = nn.Linear(in_channels, num_embeddings)
            if mode == 'gumbel':
                self.hard_gumbel = hard_gumbel
                self.t_min = t_min
                self.t_max = t_max
                self.temperature = 1.

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    @property
    def num_embeddings(self):
        return self.embedding.num_embeddings

    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim

    def adjust_temperature(self, global_step: int, max_steps: int):
        if self.mode == 'gumbel':
            # cosine decay
            self.temperature = self.t_min + 0.5 * (self.t_max - self.t_min) * (1 + np.cos(min(global_step / max_steps, 1.) * np.pi))

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.pop(f'{prefix}embed.weight', None)) is not None:
            # load pre-trained Gumbel VQGAN
            state_dict[f'{prefix}embedding.weight'] = weight
        if (weight := state_dict.get(proj_weight_key := f'{prefix}proj.weight')) is not None:
            # convert 1x1 conv2d weight (from VQGAN with Gumbel softmax) to linear
            if weight.ndim == 4 and weight.shape[2:] == (1, 1):
                state_dict[proj_weight_key] = weight.squeeze()
        elif self.mode != 'nearest' and (weight := state_dict.get(f'{prefix}embedding.weight')) is not None:
            # dot product as logit
            state_dict[proj_weight_key] = weight
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, z: torch.Tensor, fabric: Fabric | None = None):
        z = channel_last(z).contiguous()
        if self.mode == 'nearest':
            # distances from z to embeddings, exclude |z|^2
            d = torch.sum(self.embedding.weight ** 2, dim=1) \
                - 2 * einops.einsum(z, self.embedding.weight, '... d, ne d -> ... ne')
            index = d.argmin(dim=-1)
            z_q = self.embedding(index)
            loss = ((z_q.detach() - z) ** 2).mean() + self.beta * ((z_q - z.detach()) ** 2).mean()
            # preserve gradients
            z_q = z + (z_q - z).detach()
            # reshape back to match original input shape
            diversity = sum([i.unique().numel() for i in index]) / np.prod(z_q.shape[:-1])
            z_q = channel_first(z_q).contiguous()
            return VectorQuantizerOutput(z_q, index, loss, diversity)
        else:
            logits: torch.Tensor = self.proj(z)
            probs = logits.softmax(dim=-1)
            mean_probs = einops.reduce(probs, '... d -> d', reduction='mean')
            if self.training:
                # don't use all_reduce: https://github.com/Lightning-AI/lightning/issues/18228
                detached = mean_probs.detach()
                mean_probs = fabric.all_gather(detached).mean(dim=0) - detached + mean_probs
            loss = (mean_probs * mean_probs.log()).sum()
            entropy = -einops.einsum(probs, logits.log_softmax(dim=-1), '... ne, ... ne -> ...').mean()
            if self.mode == 'gumbel' and self.training:
                index_probs = nnf.gumbel_softmax(logits, self.temperature, self.hard_gumbel, dim=-1)
            else:
                index_probs = probs
            z_q = einops.einsum(index_probs, self.embedding.weight, '... ne, ne d -> ... d')
            # https://github.com/pytorch/pytorch/issues/103142
            diversity = sum([i.unique().numel() for i in probs.argmax(dim=-1)]) / np.prod(z_q.shape[:-1])
            z_q = channel_first(z_q).contiguous()
            return VectorQuantizerOutput(z_q, index_probs, loss, diversity, logits, entropy)

    def get_codebook_entry(self, index: torch.Tensor):
        if self.mode == 'soft':
            # assume soft index to be channel first
            z_q = einops.einsum(index, self.embedding.weight, 'n ne ..., ne d -> n d ...')
        else:
            z_q = self.embedding(index)
            z_q = channel_first(z_q).contiguous()
        return z_q
