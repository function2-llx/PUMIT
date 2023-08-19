import warnings
from collections.abc import Sequence

import einops
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.utils import checkpoint

from mylib.models.blocks import get_conv_layer
from mylib.models.init import init_common
from mylib.models.layers import Norm, Act, PositionEmbedding

__all__ = []

def get_spatial_pattern(spatial_shape: Sequence[int]):
    spatial_dims = len(spatial_shape)
    spatial_pattern = ' '.join(map(lambda i: f's{i}', range(spatial_dims)))
    spatial_dict = {
        f's{i}': s
        for i, s in enumerate(spatial_shape)
    }
    return spatial_pattern, spatial_dict

class MultiscaleDeformableSelfAttention(nn.Module):
    """
    Multiscale deformable attention originally proposed in Deformable DETR.
    """

    def __init__(self, embed_dim: int, num_heads: int, n_levels: int, n_points: int, spatial_dims: int):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim (d_model) must be divisible by num_heads, but got {embed_dim} and {num_heads}"
            )
        dim_per_head = embed_dim // num_heads
        # check if dim_per_head is power of 2
        if not ((dim_per_head & (dim_per_head - 1) == 0) and dim_per_head != 0):
            warnings.warn(
                "You'd better set embed_dim (d_model) in DeformableDetrMultiscaleDeformableAttention to make the"
                " dimension of each attention head a power of 2 which is more efficient in the authors' CUDA"
                " implementation."
            )

        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points
        self.spatial_dims = spatial_dims

        self.sampling_offsets = nn.Linear(embed_dim, num_heads * n_levels * n_points * spatial_dims)
        self.attention_weights = nn.Linear(embed_dim, num_heads * n_levels * n_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        reference_points: torch.Tensor,  # (n, n_seq, L, sp), normalized to [-1, 1], no align corners
        spatial_shapes: torch.Tensor
    ) -> torch.Tensor:
        # add position embeddings to the hidden states before projecting to queries and keys
        hidden_states = hidden_states + position_embeddings
        values = self.value_proj(hidden_states)
        values = einops.rearrange(values, '... (M d) -> ... M d', M=self.n_heads)
        sampling_offsets = self.sampling_offsets(hidden_states)
        sampling_offsets = einops.rearrange(
            sampling_offsets, '... (M L K sp) -> ... M L K sp',
            M=self.n_heads, L=self.n_levels, K=self.n_points, sp=self.spatial_dims,
        )
        offset_normalizer = spatial_shapes
        sampling_points = einops.rearrange(reference_points, '... L sp -> ... 1 L 1 sp') \
            + sampling_offsets / einops.rearrange(offset_normalizer, 'L sp -> 1 L 1 sp')
        sampling_points = sampling_points.flip(dims=(-1, ))
        value_list = values.split(spatial_shapes.prod(dim=-1).tolist(), dim=1)
        sampling_value_list = []
        dummy_dim = ' 1 ' if self.spatial_dims == 3 else ' '
        for level_id, spatial_shape in enumerate(spatial_shapes):
            spatial_pattern, spatial_dict = get_spatial_pattern(spatial_shape)
            level_value = einops.rearrange(
                value_list[level_id],
                f'n ({spatial_pattern}) M d -> (n M) d {spatial_pattern}',
                **spatial_dict,
            )
            sampling_value = nnf.grid_sample(
                level_value,
                einops.rearrange(
                    sampling_points[:, :, :, level_id],
                    f'n nq M K sp -> (n M){dummy_dim}nq K sp',
                ),
                mode='bilinear', padding_mode="zeros", align_corners=False,
            )
            sampling_value_list.append(sampling_value)
        sampling_values = einops.rearrange(sampling_value_list, f'L (n M) d{dummy_dim}nq K -> n nq M (L K) d', M=self.n_heads)
        # nk = L * K
        attention_weights = einops.rearrange(
            self.attention_weights(hidden_states), '... (M nk) -> ... M 1 nk', M=self.n_heads,
        ).softmax(dim=-1)
        output = einops.rearrange(attention_weights @ sampling_values, '... M 1 d -> ... (M d)')
        output = self.output_proj(output)

        return output

class MultiscaleDeformablePixelDecoderLayer(nn.Module):
    def __init__(self, spatial_dims: int, embed_dim: int, num_heads: int, n_levels: int, n_points: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.ms_deform_sa = MultiscaleDeformableSelfAttention(embed_dim, num_heads, n_levels, n_points, spatial_dims)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = dropout
        self.activation_fn = nn.functional.relu
        self.activation_dropout = dropout
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ms_deform_sa(hidden_states, position_embeddings, reference_points, spatial_shapes)
        hidden_states = nnf.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        # following the post-norm used by deformable detr
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nnf.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nnf.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        return hidden_states

# from transformers.models.mask2former.modeling_mask2former import Mask2FormerPixelDecoder
class MultiscaleDeformablePixelDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        backbone_feature_channels: Sequence[int],
        feature_dim: int,
        num_heads: int,
        num_feature_levels: int = 3,
        n_points: int = 4,
        num_layers: int = 6,
        mlp_dim: int | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.interpolate_mode = 'bilinear' if spatial_dims == 2 else 'trilinear'
        self.num_fpn_levels = len(backbone_feature_channels) - num_feature_levels
        self.input_projections = nn.ModuleList([
            get_conv_layer(
                spatial_dims, backbone_feature_channel, feature_dim, kernel_size=1,
                norm=(Norm.GROUP, {'num_groups': 32, 'num_channels': feature_dim}),
                act=None,
                bias=i >= self.num_fpn_levels   # follow the reference implementation
            )
            for i, backbone_feature_channel in enumerate(backbone_feature_channels)
        ])
        self.position_embedding = PositionEmbedding(feature_dim, spatial_dims)
        self.level_embedding = nn.Embedding(num_feature_levels, feature_dim)
        if mlp_dim is None:
            mlp_dim = 4 * feature_dim
        self.layers: Sequence[MultiscaleDeformablePixelDecoderLayer] | nn.ModuleList = nn.ModuleList([
            MultiscaleDeformablePixelDecoderLayer(spatial_dims, feature_dim, num_heads, num_feature_levels, n_points, mlp_dim)
            for _ in range(num_layers)
        ])
        self.output_convs = nn.ModuleList([
            get_conv_layer(
                spatial_dims, feature_dim, feature_dim, kernel_size=3, bias=False,
                norm=(Norm.GROUP, {'num_groups': 32, 'num_channels': feature_dim}),
                act=Act.RELU,
            )
            for _ in range(self.num_fpn_levels)
        ])

        self.gradient_checkpointing = gradient_checkpointing

        self.apply(init_common)

    def no_weight_decay(self):
        return {'level_embedding'}

    @staticmethod
    def get_reference_points(spatial_shapes: torch.Tensor, batch_size: int):
        device = spatial_shapes.device
        reference_points = torch.cat(
            [
                torch.cartesian_prod(*[
                    torch.linspace(start := -1 + 1 / s, -start, s, dtype=torch.float32, device=device)
                    for s in spatial_shape
                ])
                for spatial_shape in spatial_shapes
            ],
            dim=0,
        )
        reference_points = einops.repeat(
            reference_points,
            'nq sp -> n nq L sp', n=batch_size, L=spatial_shapes.shape[0],
        )
        return reference_points

    def forward_deformable_layers(self, feature_maps: list[torch.Tensor]):
        position_embeddings = [
            self.position_embedding(x)
            for x in feature_maps
        ]
        hidden_states = torch.cat(
            [
                einops.rearrange(feature_map, 'n c ... -> n (...) c')
                for feature_map in feature_maps
            ],
            dim=1,
        )
        spatial_shapes = torch.tensor([embed.shape[2:] for embed in position_embeddings], device=hidden_states.device)
        position_embeddings = torch.cat(
            [
                einops.rearrange(pe, 'n c ... -> n (...) c') + self.level_embedding.weight[i]
                for i, pe in enumerate(position_embeddings)
            ],
            dim=1,
        )
        reference_points = self.get_reference_points(spatial_shapes, hidden_states.shape[0])

        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                hidden_states = checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    position_embeddings,
                    reference_points,
                    spatial_shapes,
                )
            else:
                hidden_states = layer.forward(hidden_states, position_embeddings, reference_points, spatial_shapes)
        return [
            einops.rearrange(x, f'n ({spatial_pattern}) d -> n d {spatial_pattern}', **spatial_dict)
            for x, (spatial_pattern, spatial_dict) in zip(
                hidden_states.split(spatial_shapes.prod(dim=-1).tolist(), dim=1),
                [get_spatial_pattern(spatial_shape) for spatial_shape in spatial_shapes]
            )
        ]

    def forward(self, backbone_feature_maps: list[torch.Tensor]):
        feature_maps = [
            projection(feature_map)
            for projection, feature_map in zip(self.input_projections, backbone_feature_maps)
        ]
        outputs = self.forward_deformable_layers(feature_maps[self.num_fpn_levels:])[::-1]

        for lateral, output_conv in zip(feature_maps, self.output_convs):
            output = lateral + nnf.interpolate(outputs[-1], lateral.shape[2:], mode=self.interpolate_mode)
            outputs.append(output_conv(output))
        return outputs[::-1]

def main():
    bs = 2
    sp = 3
    spatial_shapes = [
        [48] * sp,
        [24] * sp,
        [12] * sp,
        [6] * sp,
    ]
    feature_channels = [96, 192, 384, 768]
    feature_maps = [
        torch.randn(bs, c, *spatial_shape, device='cuda')
        for c, spatial_shape in zip(feature_channels, spatial_shapes)
    ]
    decoder = MultiscaleDeformablePixelDecoder(sp, feature_channels).cuda()
    print(decoder)
    decoder.forward(feature_maps, torch.tensor(0))

if __name__ == '__main__':
    main()
