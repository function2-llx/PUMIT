from lightning import LightningModule
import torch
from torch import nn
from torch.nn import functional as nnf

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks import one_hot
from monai.networks.layers import Conv
from monai.utils import MetricReduction

from mylib.conf import SegExpConf, ExpConfBase, SegCommonConf
from mylib.utils import DataKey
from .model_base import ExpModelBase
from ..init import init_common

class SegInferer(LightningModule):
    conf: ExpConfBase | SegCommonConf

    def seg_predictor(self, x):
        return self.forward(x)

    def sw_infer(self, img: torch.Tensor, progress: bool = None, softmax: bool = False):
        ret = sliding_window_inference(
            img,
            roi_size=self.conf.sample_shape,
            sw_batch_size=self.conf.sw_batch_size,
            predictor=self.seg_predictor,
            overlap=self.conf.sw_overlap,
            mode=self.conf.sw_blend_mode,
            progress=progress,
        )
        if softmax:
            ret = ret.softmax(dim=1)
        return ret

    def tta_infer(self, img: torch.Tensor, progress: bool = None, softmax: bool = False):
        pred = self.sw_infer(img, progress, softmax)
        for flip_idx in self.tta_flips:
            pred += torch.flip(self.sw_infer(torch.flip(img, flip_idx), progress, softmax), flip_idx)
        pred /= len(self.tta_flips) + 1
        return pred

    def infer(self, img: torch.Tensor, progress: bool = None, tta_softmax: bool = False):
        if progress is None:
            progress = self.trainer.testing if self._trainer is not None else True

        if self.conf.do_tta:
            return self.tta_infer(img, progress, tta_softmax)
        else:
            return self.sw_infer(img, progress)

class SegModel(ExpModelBase, SegInferer):
    conf: SegExpConf

    def create_decoder(self):
        return create_model(self.conf.decoder, decoder_registry)

    def __init__(self, conf: SegExpConf):
        super().__init__(conf)
        self.decoder = self.create_decoder()
        with torch.no_grad():
            self.decoder.eval()
            dummy_input, dummy_encoder_output = self.backbone_dummy()
            dummy_decoder_output = self.decoder.forward(dummy_encoder_output.feature_maps, dummy_input)
            if conf.print_shape:
                print('decoder output shapes:')
                for x in dummy_decoder_output.feature_maps:
                    print(x.shape)
            decoder_feature_sizes = [feature.shape[1] for feature in dummy_decoder_output.feature_maps]
        # decoder feature map: from small to large
        # i-th seg head for the last i-th output from decoder, i.e., the 0-th seg head for the largest output
        self.seg_heads = nn.ModuleList([
            Conv[Conv.CONV, conf.spatial_dims](decoder_feature_sizes[-i - 1], conf.num_seg_classes, kernel_size=1)
            for i in range(conf.num_seg_heads)
        ])
        seg_head_weights = torch.tensor([0.5 ** i for i in range(conf.num_seg_heads)])
        self.seg_head_weights = nn.Parameter(seg_head_weights / seg_head_weights.sum(), requires_grad=False)

        if conf.multi_label:
            self.seg_loss_fn = DiceFocalLoss(
                include_background=conf.loss_include_background,
                to_onehot_y=False,
                sigmoid=True,
                softmax=False,
                squared_pred=conf.dice_squared,
                smooth_nr=conf.dice_nr,
                smooth_dr=conf.dice_dr,
                gamma=conf.focal_gamma,
            )
        else:
            self.seg_loss_fn = DiceCELoss(
                include_background=conf.loss_include_background,
                to_onehot_y=True,
                sigmoid=False,
                softmax=True,
                squared_pred=conf.dice_squared,
                smooth_nr=conf.dice_nr,
                smooth_dr=conf.dice_dr,
            )
        self.val_metrics = {
            'dice': DiceMetric(include_background=True),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone.forward(x)
        feature_maps = self.decoder.forward(output.feature_maps, x).feature_maps
        if self.conf.self_ensemble:
            return torch.stack([
                nnf.interpolate(seg_head(fm), x.shape[2:], mode=self.interpolate_mode)
                for fm, seg_head in zip(reversed(feature_maps), self.seg_heads)
            ]).mean(dim=0)
        else:
            ret = self.seg_heads[0](feature_maps[-1])
            if ret.shape[2:] != x.shape[2:]:
                ret = nnf.interpolate(ret, x.shape[2:], mode=self.interpolate_mode)
            return ret

    def compute_loss(self, output_logits: list[torch.Tensor] | torch.Tensor, seg_label: torch.Tensor):
        if isinstance(output_logits, list):
            seg_loss = torch.stack([
                self.seg_loss_fn(
                    nnf.interpolate(output_logit, seg_label.shape[2:], mode=self.interpolate_mode),
                    seg_label,
                )
                for output_logit in output_logits
            ])
            return seg_loss[0], torch.dot(seg_loss, self.seg_head_weights)
        else:
            return self.seg_loss_fn(output_logits, seg_label)

    def training_step(self, batch: dict, *args, **kwargs):
        img = batch[DataKey.IMG]
        seg_label = batch[DataKey.SEG]
        # from mylib.utils import IndexTracker
        # for i in range(img.shape[0]):
        #     IndexTracker(img[i, 1].cpu().numpy(), seg_label[i, 1].cpu().numpy())
        backbone_output: BackboneOutput = self.backbone(img)
        decoder_output = self.decoder.forward(backbone_output.feature_maps, img)
        seg_outputs = [
            seg_head(feature_map)
            for seg_head, feature_map in zip(self.seg_heads, reversed(decoder_output.feature_maps))
        ]
        single_loss, ds_loss = self.compute_loss(seg_outputs, seg_label)
        self.log('train/single_loss', single_loss)
        self.log('train/ds_loss', ds_loss)
        return ds_loss

    def on_validation_epoch_start(self):
        if self.conf.val_empty_cuda_cache:
            torch.cuda.empty_cache()
        for metric in self.val_metrics.values():
            metric.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], *args, **kwargs):
        conf = self.conf
        img = batch[DataKey.IMG]
        seg = batch[DataKey.SEG]
        pred_logit = self.sw_infer(img)
        pred_logit = nnf.interpolate(pred_logit, seg.shape[2:], mode=self.interpolate_mode)
        loss = self.compute_loss(pred_logit, seg)
        self.log('val/loss', loss, sync_dist=True)

        if conf.multi_label:
            pred = (pred_logit.sigmoid() > 0.5).long()
        else:
            pred = pred_logit.argmax(dim=1, keepdim=True)
            pred = one_hot(pred, conf.num_seg_classes)
            seg = one_hot(seg, conf.num_seg_classes)
        for k, metric in self.val_metrics.items():
            metric(pred, seg)

    def on_validation_epoch_end(self) -> None:
        if self.conf.val_empty_cuda_cache:
            torch.cuda.empty_cache()

        for name, metric in self.val_metrics.items():
            m = metric.aggregate(reduction=MetricReduction.MEAN_BATCH)
            for i in range(m.shape[0]):
                self.log(f'val/{name}/{i}', m[i], sync_dist=True)
            avg = m.nanmean() if self.conf.multi_label else m[1:].nanmean()
            self.log(f'val/{name}/avg', avg, sync_dist=True)
