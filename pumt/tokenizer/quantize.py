import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import einops
import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

from pumt.conv import InflatableConv3d
from luolib.utils import channel_last, channel_first
from monai.config import PathLike

@dataclass
class VectorQuantizerOutput:
    z_q: torch.Tensor
    index: torch.Tensor
    loss: torch.Tensor
    probs: torch.Tensor | None = None

# modified from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
class VectorQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mode: Literal['nearest', 'gumbel', 'soft'],
        hard_gumbel: bool = True,
        beta: float = 0.25,
    ):
        super().__init__()
        self.mode = mode
        if mode == 'nearest':
            self.beta = beta
            self.register_module('proj', None)
        else:
            # calculate categorical distribution over embeddings
            self.proj = nn.Linear(embedding_dim, num_embeddings)
            if mode == 'gumbel':
                self.hard_gumbel = hard_gumbel
                self.temperature = 1.

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)

    def adjust_temperature(self, global_step: int, max_steps: int):
        if self.mode == 'gumbel' and self.hard_gumbel:
            t_min, t_max = 1e-6, 1.
            self.temperature = t_min + 0.5 * (t_max - t_min) * (1 + np.cos(min(global_step / max_steps, 1.) * np.pi))

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.pop(f'{prefix}embed.weight', None)) is not None:
            # load pre-trained Gumbel VQGAN
            state_dict[f'{prefix}embedding.weight'] = weight
        if (weight := state_dict.get(proj_weight_key := f'{prefix}proj.weight')) is not None:
            # convert 1x1 conv2d weight (from VQGAN with Gumbel softmax) to linear
            if weight.ndim == 4 and weight.shape[2:] == (1, 1):
                state_dict[proj_weight_key] = weight.squeeze()
        elif self.mode != 'nearest':
            # dot product as logit
            state_dict[proj_weight_key] = einops.rearrange(state_dict[f'{prefix}embedding.weight'], 'ne d -> ne d 1 1 1')
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, z: torch.Tensor):
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
            z_q = channel_first(z_q).contiguous()
            return VectorQuantizerOutput(z_q, index, loss)
        else:
            logits: torch.Tensor = self.proj(z)
            probs = logits.softmax(dim=-1)
            entropy = einops.einsum(logits.log_softmax(dim=-1), probs, '... ne, ... ne -> ...')
            loss = entropy.mean()
            if self.mode == 'gumbel':
                if self.training:
                    one_hot_prob = nnf.gumbel_softmax(logits, self.temperature, self.hard_gumbel, dim=-1)
                    index = one_hot_prob.argmax(dim=-1, keepdim=True)
                    z_q = einops.einsum(one_hot_prob, self.embedding.weight, '... ne, ne d -> n ... d')
                else:
                    index = probs.argmax(dim=1)
                    z_q = self.embedding(index)
            else:
                index = probs
                z_q = einops.einsum(probs, self.embedding.weight, '... ne, ne d -> ... d')
            z_q = channel_first(z_q).contiguous()
            return VectorQuantizerOutput(z_q, index, loss, probs)

    def get_codebook_entry(self, index: torch.Tensor):
        if self.mode == 'soft':
            # assume soft index to be channel first
            z_q = einops.einsum(index, self.embedding.weight, 'n ne ..., ne d -> n d ...')
        else:
            z_q = self.embedding(index)
            z_q = channel_first(z_q).contiguous()
        return z_q
