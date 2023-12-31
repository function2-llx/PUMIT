from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import nibabel as nib
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
import torch
from torch import nn
from torch.nn import functional as nnf
from tqdm import tqdm

from luolib.lightning import LightningModule
from luolib.models import MaskFormer
from luolib import transforms as lt
from luolib.types import tuple3_t
from monai.data import MetaTensor, affine_to_spacing
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.utils import BlendMode, ImageMetaKey, InterpolateMode, MetaKeys, MetricReduction

class DeepSupervisionWrapper(nn.Module):
    weight: torch.Tensor

    def __init__(self, loss: nn.Module, num_ds: int, mode: InterpolateMode = InterpolateMode.NEAREST_EXACT):
        super().__init__()
        self.loss = loss
        weight = torch.tensor([1 / (1 << i) for i in range(num_ds)])
        weight /= weight.sum()
        self.register_buffer('weight', weight)
        self.mode = mode

    @staticmethod
    @lru_cache(1)
    def prepare_labels(label: torch.Tensor, spatial_shapes: tuple[torch.Size, ...], mode: InterpolateMode) -> list[torch.Tensor]:
        label_shape = label.shape[2:]
        return [
            nnf.interpolate(label, shape, mode=mode) if label_shape != shape else label
            for shape in spatial_shapes
        ]

    def forward(self, deep_logits: list[torch.Tensor], label: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        spatial_shapes = tuple([logits.shape[2:] for logits in deep_logits])
        deep_labels = self.prepare_labels(label, spatial_shapes, self.mode)
        ds_losses = [
            self.loss(logits, deep_label)
            for logits, deep_label in zip(deep_logits, deep_labels)
        ]
        ds_loss = torch.dot(self.weight, torch.stack(ds_losses))
        return ds_loss, ds_losses

num_fg_classes: int = 17

class MSDModel(MaskFormer, LightningModule):
    loss_weight: torch.Tensor

    def __init__(
        self,
        *args,
        loss: DiceFocalLoss | None = None,
        sample_size: tuple3_t[int],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.loss = loss
        self.ds_wrapper = DeepSupervisionWrapper(loss, len(self.pixel_embedding_levels))
        # loss_weight = torch.tensor([0.5 ** i for i in range(len(self.seg_heads))])
        # self.register_buffer('loss_weight', loss_weight / loss_weight.sum(), persistent=False)
        self.sample_size = sample_size
        # self.sw_batch_size = sw_batch_size
        # self.sw_overlap = sw_overlap
        # self.sw_softmax = sw_softmax

    def input_norm(self, x: torch.Tensor):
        return 2 * x - 1

    def get_seg_feature_maps(self, x: torch.Tensor):
        feature_maps = self.backbone(x)
        seg_feature_maps = self.decoder(feature_maps, x)[::-1]
        return seg_feature_maps

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], *args, **kwargs):
        img, label = batch
        img = self.input_norm(img)
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
        return self.seg_heads[0](seg_feature_map)

    def sw_infer(self, x: torch.Tensor, softmax: bool = True):
        y = sliding_window_inference(
            x, self.sample_size, self.sw_batch_size, self, self.sw_overlap, BlendMode.GAUSSIAN,
        )
        if softmax:
            y = y.softmax(dim=1)
        return y

    def tta_infer(self, x: torch.Tensor, softmax: bool = True):
        tta_flips = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        out = None
        for flip_idx in tqdm(tta_flips, ncols=80, desc='tta inference'):
            cur_out = self.sw_infer(torch.flip(x, flip_idx) if flip_idx else x, softmax)
            if flip_idx:
                cur_out = torch.flip(cur_out, flip_idx)
            if out is None:
                out = cur_out
            else:
                out += cur_out
        out /= len(tta_flips)
        return out

    def on_validation_epoch_start(self) -> None:
        self.dice_metric.reset()

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], *args, **kwargs):
        img, label = batch
        img = self.input_norm(img)
        prob = self.sw_infer(img, self.sw_softmax)
        prob = resample(prob, label.shape[2:])
        pred = prob.argmax(dim=1, keepdim=True)
        self.dice_metric(pred, label)
        self.log('val/loss', self.loss(prob.log(), label))

    def on_validation_epoch_end(self) -> None:
        dice = self.dice_metric.aggregate(MetricReduction.MEAN_BATCH) * 100
        for i in range(dice.shape[0]):
            self.log(f'val/dice/{i}', dice[i])
        self.log(f'val/dice/avg', dice[1:].mean())

    @property
    def test_output_dir(self):
        return self.run_dir / 'pred'

    def on_test_epoch_start(self) -> None:
        self.dice_metric.reset()
        self.test_output_dir.mkdir()

    def test_step(self, batch: tuple[MetaTensor, torch.Tensor], *args, **kwargs):
        img, label = batch
        meta = img[0].meta
        case = Path(meta[ImageMetaKey.FILENAME_OR_OBJ]).name.split('.')[0]
        img = self.input_norm(img)
        prob = self.tta_infer(img, self.sw_softmax)[0]
        affine = meta[MetaKeys.ORIGINAL_AFFINE]
        inverse_orientation = lt.AffineOrientation(affine)
        prob = inverse_orientation(prob)
        prob = resample_data_or_seg_to_shape(
            prob, meta[MetaKeys.SPATIAL_SHAPE].tolist(), prob.pixdim.numpy(), affine_to_spacing(affine).numpy(),
            False, 1, 0, None,
        )
        prob = torch.as_tensor(prob, device=label.device)
        pred = prob.argmax(dim=0, keepdim=True)
        self.dice_metric(pred[None], label)
        nib.save(nib.Nifti1Image(pred[0].byte().cpu().numpy(), affine.numpy()), self.test_output_dir / f'{case}.nii.gz')

    def on_test_epoch_end(self) -> None:
        dice = self.dice_metric.aggregate(MetricReduction.MEAN_BATCH) * 100
        for i in range(dice.shape[0]):
            self.log(f'test/dice/{i}', dice[i])
        self.log(f'test/dice/avg', dice[1:].mean())
