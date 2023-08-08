from collections.abc import Sequence

from lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.models.decoders.full_res import FullResAdapter
from monai.losses import DiceCELoss

from pumt.model import SimpleViTAdapter

class CHAOSModel(LightningModule):
    loss_weight: torch.Tensor

    def __init__(
        self,
        *,
        backbone: SimpleViTAdapter,
        decoder: FullResAdapter,
        seg_feature_channels: Sequence[int],
        num_fg_classes: int = 4,
        loss: DiceCELoss | None = None,
    ):
        super().__init__()
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

    def compute_loss(self, output_logits: list[torch.Tensor] | torch.Tensor, seg_label: torch.Tensor):
        if isinstance(output_logits, list):
            seg_loss = torch.stack([
                self.seg_loss_fn(
                    nnf.interpolate(output_logit, seg_label.shape[2:], mode='trilinear'),
                    seg_label,
                )
                for output_logit in output_logits
            ])
            return seg_loss[0], torch.dot(seg_loss, self.loss_weight)
        else:
            return self.seg_loss_fn(output_logits, seg_label)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], *args, **kwargs):
        img, label = batch
        logits = self(img)
        loss = torch.stack([
            self.loss(nnf.interpolate(logit, label.shape[2:], mode='trilinear'), label)
            for logit in logits
        ])
        loss = torch.dot(loss, self.loss_weight)
        return loss
