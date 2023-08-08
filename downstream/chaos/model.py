from collections.abc import Sequence

import einops
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.models.decoders.full_res import FullResAdapter
from luolib.utils.lightning import LightningModule
from monai.data import MetaTensor
from monai.losses import DiceCELoss

from pumt.model import SimpleViTAdapter

class CHAOSModel(LightningModule):
    loss_weight: torch.Tensor

    def __init__(
        self,
        backbone: SimpleViTAdapter,
        decoder: FullResAdapter,
        seg_feature_channels: Sequence[int],
        num_fg_classes: int = 4,
        loss: DiceCELoss | None = None,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.decoder = decoder
        self.seg_heads = nn.ModuleList([
            nn.Conv3d(c, num_fg_classes + 1, 1)
            for c in seg_feature_channels
        ])
        self.loss = loss
        loss_weight = torch.tensor([0.5 ** i for i in range(len(self.seg_heads))])
        self.register_buffer('loss_weight', loss_weight / loss_weight.sum(), persistent=False)

    def forward(self, x: torch.Tensor):
        feature_maps = self.backbone(x)
        seg_feature_maps = self.decoder(feature_maps, x)[::-1]
        return [
            seg_head(feature_map)
            for seg_head, feature_map in zip(self.seg_heads, seg_feature_maps)
        ]

    def training_step(self, batch: tuple[torch.Tensor, ...], *args, **kwargs):
        img, mean, std, label = batch
        img = (img - mean) / std
        logits = self(img)
        loss = torch.stack([
            self.loss(nnf.interpolate(logit, label.shape[2:], mode='trilinear'), label)
            for logit in logits
        ])
        loss = torch.dot(loss, self.loss_weight)
        self.log('train/loss', loss)
        return loss
