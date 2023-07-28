from collections.abc import Sequence

from torch import nn

from luolib.types import param3_t, tuple3_t
from .base import VQTokenizerBase
from .vqvae import ResnetBlock
from ..conv import SpatialTensor, AdaptiveUpsampleWithPostConv
from ..model.vit import PatchEmbed

class SimpleVQTokenizer(VQTokenizerBase):
    def __init__(
        self,
        stride: param3_t[int],
        in_channels: int,
        hidden_dims: Sequence[int],
        mlp_act_layer: type[nn.Module] = nn.GELU,
        mlp_act_layer_kwargs: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        for s in self.stride:
            assert s & s - 1 == 0
        assert self.stride[1] == self.stride[2]
        self.patch_embed = PatchEmbed(stride, in_channels, hidden_dims[0])
        mlp_act_layer_kwargs = mlp_act_layer_kwargs or {}
        self.encoder = nn.Sequential(
            self.patch_embed,
            nn.LayerNorm(hidden_dims[0]),
            mlp_act_layer(**mlp_act_layer_kwargs),
        )
        for i in range(1, len(hidden_dims)):
            self.encoder.extend([
                nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                nn.LayerNorm(hidden_dims[i]),
                mlp_act_layer(**mlp_act_layer_kwargs),
            ])

        self.decoder = nn.Sequential(

        )


    @property
    def stride(self) -> tuple3_t[int]:
        return self.patch_embed.patch_size

    def encode(self, x: SpatialTensor) -> SpatialTensor:
        pass

    def decode(self, x: SpatialTensor) -> SpatialTensor:
        pass
