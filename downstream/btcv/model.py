from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as nnf
from tqdm import tqdm

from luolib.models.decoders.full_res import FullResAdapter
from luolib.types import tuple3_t
from luolib.utils.lightning import LightningModule
from monai.data import MetaTensor
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import BlendMode, MetaKeys, MetricReduction

from pumt.model import SimpleViTAdapter
from pumt.model.vit import resample

class BTCVModel(LightningModule):
    loss_weight: torch.Tensor

    def __init__(
        self,
        *args,
        backbone: SimpleViTAdapter,
        decoder: FullResAdapter,
        seg_feature_channels: Sequence[int],
        num_fg_classes: int = 13,
        loss: DiceCELoss | None = None,
        sample_size: tuple3_t[int],
        sw_batch_size: int = 4,
        sw_overlap: float = 0.5,
        **kwargs,
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
        self.sample_size = sample_size
        self.sw_batch_size = sw_batch_size
        self.sw_overlap = sw_overlap
        self.dice_metric = DiceMetric(num_classes=num_fg_classes + 1)

    def get_seg_feature_maps(self, x: torch.Tensor):
        feature_maps = self.backbone(x)
        seg_feature_maps = self.decoder(feature_maps, x)[::-1]
        return seg_feature_maps

    def training_step(self, batch: tuple[torch.Tensor, ...], *args, **kwargs):
        img, label = batch
        img = img * 2 - 1
        seg_feature_maps = self.get_seg_feature_maps(img)
        logits = [
            seg_head(feature_map)
            for seg_head, feature_map in zip(self.seg_heads, seg_feature_maps)
        ]
        loss = torch.stack([
            self.loss(nnf.interpolate(logit, label.shape[2:], mode='trilinear'), label)
            for logit in logits
        ])
        loss = torch.dot(loss, self.loss_weight)
        self.log('train/loss', loss)
        return loss

    def forward(self, x: torch.Tensor):
        seg_feature_map = self.get_seg_feature_maps(x)[0]
        return self.seg_heads[0](seg_feature_map).softmax(dim=1)

    def sw_infer(self, x: torch.Tensor):
        return sliding_window_inference(
            x, self.sample_size, self.sw_batch_size, self, self.sw_overlap, BlendMode.GAUSSIAN,
        )

    def tta_infer(self, x: torch.Tensor):
        tta_flips = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        prob = None
        for flip_idx in tqdm(tta_flips, ncols=80, desc='tta inference'):
            cur_prob = self.sw_infer(torch.flip(x, flip_idx) if flip_idx else x)
            if flip_idx:
                cur_prob = torch.flip(cur_prob, flip_idx)
            if prob is None:
                prob = cur_prob
            else:
                prob += cur_prob
        prob /= len(tta_flips)
        return prob

    def on_validation_epoch_start(self) -> None:
        self.dice_metric.reset()

    def validation_step(self, batch: tuple[torch.Tensor, ...], *args, **kwargs):
        img, label = batch
        img = img * 2 - 1
        img = img.as_tensor()
        prob = self.sw_infer(img)
        prob = resample(prob, label.shape[2:])
        pred = prob.argmax(dim=1, keepdim=True)
        self.dice_metric(pred, label)

    def on_validation_epoch_end(self) -> None:
        dice = self.dice_metric.aggregate(MetricReduction.MEAN_BATCH) * 100
        for i in range(dice.shape[0]):
            self.log(f'val/dice/{i}', dice[i])
        self.log(f'val/dice/avg', dice[1:].mean())
