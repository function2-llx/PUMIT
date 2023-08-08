from collections.abc import Sequence

from lightning import LightningModule
import torch
from torch import nn

from luolib.models.decoders.full_res import FullResAdapter
from pumt.model import SimpleViTAdapter

class CHAOSModel(LightningModule):
    def __init__(
        self,
        backbone: SimpleViTAdapter,
        decoder: FullResAdapter,
        seg_feature_channels: Sequence[int],
        num_fg_classes: int = 4,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.seg_heads = nn.ModuleList([
            nn.Conv3d(c, num_fg_classes + 1, 1)
            for c in seg_feature_channels
        ])

    def forward(self, x: torch.Tensor):
        feature_maps = self.backbone(x)
        seg_feature_maps = self.decoder(feature_maps, x)[::-1]
        return [
            seg_head(feature_map)
            for seg_head, feature_map in zip(self.seg_heads, seg_feature_maps)
        ]
