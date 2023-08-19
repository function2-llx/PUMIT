import random
from collections.abc import Sequence

import einops
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.utils import checkpoint

from monai.networks.layers import Conv

from mylib.utils import flatten
from ..init import init_common
from ..layers import PositionEmbedding

class MaskedAttentionDecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_attention_heads: int,
        dim_feedforward: int,
        dropout: float = 0.,
        pre_norm: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.pre_norm = pre_norm
        self.dropout = dropout

        self.cross_attn = nn.MultiheadAttention(embed_dim, num_attention_heads, dropout, batch_first=True)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)

        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            add_bias_kv=True,   # follow the reference implementation,
            batch_first=True,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, dim_feedforward)
        self.activation_fn = nn.ReLU()
        self.activation_dropout = dropout
        self.fc2 = nn.Linear(dim_feedforward, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    @property
    def post_norm(self):
        return not self.pre_norm

    def forward(
        self,
        hidden_states: torch.Tensor,
        query_position_embeddings: torch.Tensor,
        key_hidden_states: torch.Tensor,
        key_position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.cross_attn_layer_norm(hidden_states)
        hidden_states, cross_attn_weights = self.cross_attn.forward(
            query=hidden_states + query_position_embeddings,
            key=key_hidden_states + key_position_embeddings,
            value=key_hidden_states,
            attn_mask=attention_mask,
            key_padding_mask=None,
        )
        hidden_states = nnf.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if self.post_norm:
            hidden_states = self.cross_attn_layer_norm(hidden_states)

        # Self Attention Block
        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn.forward(
            query=(query := hidden_states + query_position_embeddings),
            key=query,
            value=hidden_states,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if self.post_norm:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        if self.pre_norm:
            hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        if self.post_norm:
            hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states

# modified from transformers.models.mask2former.modeling_mask2former.Mask2FormerMaskedAttentionDecoder
class MaskedAttentionDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        feature_channels: int,
        num_attention_heads: int,
        dim_feedforward: int | None = None,
        num_decoder_layers: int = 9,
        num_feature_levels: int = 3,
        num_queries: int = 100,
        pixel_embedding_dim: int = None,
        dropout: float = 0.,
        pre_norm: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.interpolate_mode = 'bilinear' if spatial_dims == 2 else 'trilinear'
        self.num_attention_heads = num_attention_heads
        self.hidden_dim = hidden_dim = feature_channels
        self.dropout = dropout
        self.query_embedding = nn.Embedding(num_queries, hidden_dim)
        self.query_position_embedding = nn.Embedding(num_queries, hidden_dim)
        self.num_feature_levels = num_feature_levels
        self.level_embedding = nn.Embedding(num_feature_levels, hidden_dim)
        self.key_position_embedding = PositionEmbedding(feature_channels, spatial_dims, flatten=True)
        if dim_feedforward is None:
            dim_feedforward = feature_channels * 8
        self.pre_norm = pre_norm
        self.layers: Sequence[MaskedAttentionDecoderLayer] | nn.ModuleList = nn.ModuleList([
            MaskedAttentionDecoderLayer(feature_channels, num_attention_heads, dim_feedforward, dropout, pre_norm)
            for _ in range(num_decoder_layers)
        ])
        # this is redundant for post-norm, but let's follow the reference implementation
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.layer_drop = dropout

        if pixel_embedding_dim is None:
            pixel_embedding_dim = feature_channels
        self.mask_query_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pixel_embedding_dim),
        )
        self.pixel_embedding_projection = Conv[Conv.CONV, spatial_dims](feature_channels, pixel_embedding_dim, 1)

        self.gradient_checkpointing = gradient_checkpointing

        self.apply(init_common)

    def predict_mask(
        self,
        hidden_states: torch.Tensor,
        pixel_embedding: torch.Tensor,
        manual_mask: torch.Tensor | None = None,
        attention_mask_shape: Sequence[int] | None = None,
    ):
        mask_embeddings = self.layer_norm(hidden_states)
        projected_mask_embeddings = self.mask_query_projection(mask_embeddings)
        mask_logits = einops.einsum(projected_mask_embeddings, pixel_embedding, 'n nq c, n c ... -> n nq ...')
        if attention_mask_shape is None:
            return mask_logits
        with torch.no_grad():
            if manual_mask is None:
                attention_mask = nnf.interpolate(mask_logits, attention_mask_shape, mode=self.interpolate_mode).sigmoid()
            else:
                attention_mask = nnf.interpolate(manual_mask.float(), attention_mask_shape, mode=self.interpolate_mode)
            # `attn_mask` parameter of https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html#torch.nn.MultiheadAttention.forward
            # "For a binary mask, a True value indicates that the corresponding position is not allowed to attend."
            attention_mask = einops.rearrange((attention_mask < 0.5).bool(), 'n nq ... -> n nq (...)')
            # no restriction for empty mask
            attention_mask.masked_fill_(attention_mask.all(dim=-1, keepdim=True), False)
            attention_mask = einops.repeat(attention_mask, 'n nq nk -> (n M) nq nk', M=self.num_attention_heads)
        return mask_embeddings, mask_logits, attention_mask

    def forward(
        self,
        feature_maps: list[torch.Tensor],
        feature_map_for_pixel_embedding: torch.Tensor,
        manual_mask: torch.Tensor | None = None,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        batch_size = feature_map_for_pixel_embedding.shape[0]
        hidden_states = einops.repeat(self.query_embedding.weight, '... -> n ...', n=batch_size)
        query_position_embeddings = einops.repeat(self.query_position_embedding.weight, '... -> n ...', n=batch_size)
        key_hidden_states = [
            flatten(feature_map) + self.level_embedding.weight[i]
            for i, feature_map in enumerate(feature_maps)
        ]
        key_position_embeddings = [self.key_position_embedding(x) for x in feature_maps]
        pixel_embedding = self.pixel_embedding_projection(feature_map_for_pixel_embedding)
        mask_embeddings, mask_logits, attention_mask = self.predict_mask(
            hidden_states,
            pixel_embedding,
            manual_mask,
            feature_maps[0].shape[2:]
        )
        layers_mask_embeddings = [mask_embeddings]
        layers_mask_logits = [mask_logits]

        for idx, layer in enumerate(self.layers):
            if self.training and (random.uniform(0, 1) < self.layer_drop):
                continue
            level_index = idx % self.num_feature_levels
            if self.training and self.gradient_checkpointing:
                hidden_states = checkpoint.checkpoint(
                    layer, hidden_states, query_position_embeddings, key_hidden_states[level_index],
                    key_position_embeddings[level_index], attention_mask,
                )
            else:
                hidden_states = layer.forward(
                    hidden_states, query_position_embeddings, key_hidden_states[level_index],
                    key_position_embeddings[level_index], attention_mask,
                )

            mask_embeddings, mask_logits, attention_mask = self.predict_mask(
                hidden_states,
                pixel_embedding,
                manual_mask,
                feature_maps[(idx + 1) % self.num_feature_levels].shape[2:],
            )
            layers_mask_embeddings.append(mask_embeddings)
            layers_mask_logits.append(mask_logits)

        return layers_mask_embeddings, layers_mask_logits

def main():
    bs = 2
    sp = 3
    feature_channels = 192
    t_decoder = MaskedAttentionDecoder(sp, feature_channels).cuda()
    spatial_shapes = [
        (s, ) * sp
        for s in [48, 24, 12, 6]
    ]
    feature_maps = [
        torch.randn(bs, feature_channels, *shape, device='cuda')
        for shape in spatial_shapes
    ]
    t_decoder.forward(feature_maps[1:], feature_maps[0])

if __name__ == '__main__':
    main()
