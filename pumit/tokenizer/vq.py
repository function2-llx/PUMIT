from dataclasses import dataclass

import einops
from lightning import Fabric
import numpy as np
import torch
from torch import nn
from torch.nn import functional as nnf

@dataclass
class VectorQuantizerOutput:
    z_q: torch.Tensor
    """vector quantization results, input for decoder"""
    index: torch.Tensor
    """codebook index, can be a discrete one or soft one (categorical distribution over codebook)"""
    loss: torch.Tensor
    """quantization loss (e.g., |z_q - e|, or prior distribution regularization)"""
    util_var: float
    """evaluate if the utilization of the codebook is "uniform" enough"""
    logits: torch.Tensor | None = None
    """original logits over the codebook for probabilistic VQ"""
    entropy: torch.Tensor | None = None
    """entropy of original probability distribution"""

class VectorQuantizer(nn.Module):
    # modified from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/vqvae/quantize.py
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    @property
    def num_embeddings(self):
        return self.embedding.num_embeddings

    @property
    def embedding_dim(self):
        return self.embedding.embedding_dim

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if (weight := state_dict.pop(f'{prefix}embed.weight', None)) is not None:
            # load pre-trained Gumbel VQGAN
            state_dict[f'{prefix}embedding.weight'] = weight
        if (weight := state_dict.get(proj_weight_key := f'{prefix}proj.weight')) is not None:
            # convert 1x1 conv2d weight (from VQGAN with Gumbel softmax) to linear
            if weight.ndim == 4 and weight.shape[2:] == (1, 1):
                state_dict[proj_weight_key] = weight.squeeze()
        elif isinstance(self, NNVQ) and (weight := state_dict.get(f'{prefix}embedding.weight')) is not None:
            # dot product as logit
            state_dict[proj_weight_key] = weight
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    # def cal_diversity(self, index: torch.Tensor):
    #     bias_correction = self.num_embeddings
    #     if index.ndim == 4:
    #         # discrete
    #         raise NotImplementedError
    #     else:
    #         # probabilistic
    #         p = einops.reduce(index, '... n_e -> n_e', 'mean')
    #         var = p.var(unbiased=False)
    #     return var * bias_correction

    def forward(self, z: torch.Tensor, fabric: Fabric | None = None) -> VectorQuantizerOutput:
        """
        Args:
            z: channel-last feature map
            fabric: the Fabric instance used during DDP training
        """
        raise NotImplementedError

class NNVQ(VectorQuantizer):
    """
    Nearest Neighbor Vector Quantizer, embedding_dim is supposed to = channels of input feature map
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25):
        """
        Args:
            beta: "commitment loss" weight
        """
        super().__init__(num_embeddings, embedding_dim)
        # following VQGAN
        nn.init.uniform_(self.embedding.weight, -1.0 / num_embeddings, 1.0 / num_embeddings)
        self.beta = beta

    def forward(self, z: torch.Tensor, fabric: Fabric | None = None):
        # distances from z to embeddings, exclude |z|^2
        d = torch.sum(self.embedding.weight ** 2, dim=1) \
            - 2 * einops.einsum(z, self.embedding.weight, '... d, ne d -> ... ne')
        index = d.argmin(dim=-1)
        z_q = self.embedding(index)
        loss = ((z_q.detach() - z) ** 2).mean() + self.beta * ((z_q - z.detach()) ** 2).mean()
        # preserve gradients
        z_q = z + (z_q - z).detach()
        # reshape back to match original input shape
        # https://github.com/pytorch/pytorch/issues/103142
        diversity = sum([i.unique().numel() for i in index]) / np.prod(z_q.shape[:-1])
        return VectorQuantizerOutput(z_q, index, loss, diversity)

def cal_entropy(probs: torch.Tensor, logits: torch.Tensor):
    log_probs = logits.log_softmax(dim=-1)
    return -einops.einsum(probs, log_probs, '... ne, ... ne -> ...').mean()

class ProbabilisticVQ(VectorQuantizer):
    def __init__(self, num_embeddings: int, embedding_dim: int, in_channels: int | None = None, pdr_eps: float = 1e-6):
        """
        Args:
            in_channels: input feature map channels
        """
        super().__init__(num_embeddings, embedding_dim)
        in_channels = in_channels or embedding_dim
        # calculate categorical distribution over embeddings
        self.proj = nn.Linear(in_channels, num_embeddings)
        self.pdr_eps = pdr_eps

    def get_pdr_loss(self, probs: torch.Tensor, fabric: Fabric | None):
        """calculate prior distribution regularization and util_var"""
        mean_probs = einops.reduce(probs, '... d -> d', reduction='mean')
        if fabric is not None and fabric.world_size > 1:
            detached = mean_probs.detach()
            # note: don't change the calculation order as fabric.all_reduce modifies in-place
            mean_probs = mean_probs - detached + fabric.all_reduce(detached)
        return (mean_probs * (mean_probs + self.pdr_eps).log()).sum(), mean_probs.var(unbiased=False) * self.num_embeddings

    def embed_index(self, index_probs: torch.Tensor):
        z_q = einops.einsum(index_probs, self.embedding.weight, '... ne, ne d -> ... d')
        return z_q

    def project_over_codebook(self, z: torch.Tensor):
        logits = self.proj(z)
        probs = logits.softmax(dim=-1)
        entropy = -einops.einsum(probs, logits.log_softmax(dim=-1), '... ne, ... ne -> ...').mean()
        return logits, probs, entropy

class GumbelVQ(ProbabilisticVQ):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        in_channels: int | None = None,
        hard_gumbel: bool = True,
        t_min: float = 1e-6,
        t_max: float = 0.9,
    ):
        super().__init__(num_embeddings, embedding_dim, in_channels)
        self.hard_gumbel = hard_gumbel
        self.t_min = t_min
        self.t_max = t_max
        self.temperature = 1.

    def adjust_temperature(self, global_step: int, max_steps: int):
        # just cosine decay
        self.temperature = self.t_min + 0.5 * (self.t_max - self.t_min) * (1 + np.cos(min(global_step / max_steps, 1.) * np.pi))

    def forward(self, z: torch.Tensor, fabric: Fabric | None = None):
        logits, probs, entropy = self.project_over_codebook(z)
        loss, util_var = self.get_pdr_loss(probs, fabric)
        if self.training:
            index_probs = nnf.gumbel_softmax(logits, self.temperature, self.hard_gumbel, dim=-1)
        else:
            index_probs = probs
        z_q = self.embed_index(index_probs)
        return VectorQuantizerOutput(z_q, index_probs, loss, util_var, logits, entropy)

class SoftVQ(ProbabilisticVQ):
    def __init__(self, num_embeddings: int, embedding_dim: int, in_channels: int | None = None, prune: int | None = 3):
        super().__init__(num_embeddings, embedding_dim, in_channels)
        assert prune is None or prune > 0
        self.prune = prune

    def forward(self, z: torch.Tensor, fabric: Fabric | None = None):
        logits, probs, entropy = self.project_over_codebook(z)
        if self.prune is None:
            index_probs = probs
        else:
            with torch.no_grad():
                top_logits, top_indices = torch.topk(logits, self.prune, dim=-1)
                # write additional lines for autocast to work
                top_probs = top_logits.softmax(dim=-1)
                index_probs = torch.zeros_like(logits, dtype=top_probs.dtype)
                index_probs.scatter_(-1, top_indices, top_probs)
            index_probs = index_probs + probs - probs.detach()
        loss, util_var = self.get_pdr_loss(index_probs, fabric)
        z_q = self.embed_index(index_probs)
        return VectorQuantizerOutput(z_q, index_probs, loss, util_var, logits, entropy)
