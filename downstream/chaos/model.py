from collections.abc import Sequence
from pathlib import Path
import shutil

from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
import torch
from torch import nn
from torch.nn import functional as nnf
from tqdm import tqdm

from luolib import transforms as lt
from luolib.models.decoders.full_res import FullResAdapter
from luolib.types import tuple3_t
from luolib.utils.lightning import LightningModule
from monai.data import MetaTensor, affine_to_spacing
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss
from monai.utils import BlendMode, ImageMetaKey, MetaKeys

from pumt.model import SimpleViTAdapter
from downstream.chaos.data import extract_template, save_pred

class CHAOSModel(LightningModule):
    loss_weight: torch.Tensor

    def __init__(
        self,
        *args,
        backbone: SimpleViTAdapter,
        decoder: FullResAdapter,
        seg_feature_channels: Sequence[int],
        num_fg_classes: int = 4,
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

    def get_seg_feature_maps(self, x: torch.Tensor):
        feature_maps = self.backbone(x)
        seg_feature_maps = self.decoder(feature_maps, x)[::-1]
        return seg_feature_maps

    def training_step(self, batch: tuple[torch.Tensor, ...], *args, **kwargs):
        img, mean, std, label = batch
        img = (img - mean) / std
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

    @property
    def predict_save_dir(self):
        return self.run_dir / 'submit'

    def on_predict_start(self) -> None:
        extract_template(self.predict_save_dir)

    def tta_infer(self, x: torch.Tensor):
        tta_flips = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
        prob = None
        for flip_idx in tqdm(tta_flips, ncols=80, desc='tta inference', disable=True):
            cur_prob = sliding_window_inference(
                torch.flip(x, flip_idx) if flip_idx else x,
                self.sample_size, self.sw_batch_size, self, self.sw_overlap, BlendMode.GAUSSIAN,
            )
            if flip_idx:
                cur_prob = torch.flip(cur_prob, flip_idx)
            if prob is None:
                prob = cur_prob
            else:
                prob += cur_prob
        prob /= len(tta_flips)
        return prob

    def predict_step(self, batch: tuple[torch.Tensor, ...], batch_idx: int, dataloader_idx: int = 0):
        img, mean, std = batch
        img: MetaTensor = (img - mean) / std
        meta = img[0].meta
        original_path = Path(meta[ImageMetaKey.FILENAME_OR_OBJ])
        split, num, modality = original_path.parts[-4:-1]
        img_rel_path = f'MR/{num}/{modality}'
        prob = self.tta_infer(img)[0]
        affine = meta[MetaKeys.ORIGINAL_AFFINE]
        inverse_orientation = lt.AffineOrientation(affine)
        prob = inverse_orientation(prob)
        prob = resample_data_or_seg_to_shape(
            prob, meta[MetaKeys.SPATIAL_SHAPE].tolist(), prob.pixdim.numpy(), affine_to_spacing(affine).numpy(),
            False, 1, 0, None,
        )
        pred = prob.argmax(dim=0).byte()
        dicom_ref_dir = Path(f'datasets/CHAOS/{split}_Sets') / img_rel_path / 'DICOM_anon'
        if modality == 'T1DUAL':
            dicom_ref_dir /= 'InPhase'
        save_pred(pred.cpu().numpy(), self.predict_save_dir / 'Task5' / img_rel_path / 'Results', dicom_ref_dir)
        pred[pred != 1] = 0
        save_pred(pred.cpu().numpy(), self.predict_save_dir / 'Task3' / img_rel_path / 'Results', dicom_ref_dir)

    def on_predict_end(self) -> None:
        shutil.make_archive(
            str(self.run_dir / f'{self.run_dir.name}-submit'), 'zip', self.predict_save_dir, '.',
            verbose=True,
        )
