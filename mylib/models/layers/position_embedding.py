import math

import einops
import torch
from torch import nn

from mylib.utils import flatten

# modified from transformers.models.mask2former.modeling_mask2former.Mask2FormerSinePositionEmbedding
class PositionEmbedding(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        spatial_dims: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float | None = 2 * math.pi,
        flatten: bool = False,
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.feature_dim = feature_dim
        self.num_pos_feats = (feature_dim - 1) // spatial_dims + 1
        self.spatial_dims = spatial_dims
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

        self.flatten = flatten

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = x.new_zeros((x.shape[0], *x.shape[2:]), dtype=torch.bool)
        not_mask = ~mask
        embeds = torch.stack([not_mask.cumsum(dim=i, dtype=torch.float32) for i in range(1, self.spatial_dims + 1)], dim=-1)
        if self.normalize:
            eps = 1e-6
            for i in range(self.spatial_dims):
                spatial_slice = [slice(None)] * self.spatial_dims
                spatial_slice[i] = slice(-1, None)
                embeds[..., i] /= (embeds[:, *spatial_slice, i] + eps)
            embeds *= self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos = embeds[..., None] / dim_t
        pos = einops.rearrange([pos[..., 0::2].sin(), pos[..., 1::2].cos()], 'li2 n ... sp d -> n (sp d li2) ...')
        if self.flatten:
            pos = flatten(pos)[..., :self.feature_dim]
        else:
            pos = pos[:, :self.feature_dim]
        return pos
