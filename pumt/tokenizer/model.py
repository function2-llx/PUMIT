from collections import OrderedDict
from collections.abc import Sequence
from copy import copy
from typing import Any

import cytoolz
import numpy as np
from lightning import LightningModule
from lightning.pytorch.cli import instantiate_class
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from luolib.models import adaptive_resampling as ar
from luolib.models.blocks import InflatableConv3d, InflatableInputConv3d, InflatableOutputConv3d
from luolib.models.utils import get_no_weight_decay_keys, split_by_weight_decay
from luolib.utils import DataKey
from luolib.types import LRSchedulerConfig

from .loss import VQGANLoss
from .quantize import VectorQuantizer, VectorQuantizerOutput
from .utils import ensure_rgb, rgb_to_gray

def get_norm_layer(in_channels: int, num_groups: int = 32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def act_layer(x: torch.Tensor):
    # swish
    return x * torch.sigmoid(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.,
    ):
        super().__init__()
        self.in_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = get_norm_layer(in_channels)
        self.conv1 = InflatableConv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = get_norm_layer(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = InflatableConv3d(out_channels, out_channels, kernel_size=3, padding=1)
        if in_channels != out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = InflatableConv3d(in_channels, out_channels, kernel_size=3, padding=1)
            else:
                self.nin_shortcut = InflatableConv3d(in_channels, out_channels, kernel_size=1)

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if len(state_dict) == 0:
            # thank you,<s> USA </s>residual connection
            for name in ['conv1', 'conv2']:
                conv: InflatableConv3d = getattr(self, name)
                state_dict[f'{prefix}{name}.weight'] = torch.zeros_like(conv.weight)
                state_dict[f'{prefix}{name}.bias'] = torch.zeros_like(conv.bias)
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, x: torch.Tensor):
        h = x
        h = self.norm1(h)
        h = act_layer(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = act_layer(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if x.shape[1] != h.shape[1]:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

# modified from vq gan. weird, this is single head attention
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = get_norm_layer(in_channels)
        self.q = InflatableConv3d(in_channels, in_channels, kernel_size=1)
        self.k = InflatableConv3d(in_channels, in_channels, kernel_size=1)
        self.v = InflatableConv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = InflatableConv3d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, *spatial_shape = q.shape
        l = np.prod(spatial_shape)
        q = q.reshape(b, c, l)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, l)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, l)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, *spatial_shape)

        h_ = self.proj_out(h_)

        return x + h_

class EncoderDownLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        use_attn: bool,
        downsample: bool,
        gradient_checkpointing: bool,
    ):
        # use_attn = False
        super().__init__()
        self.block: Sequence[ResnetBlock] | nn.ModuleList = nn.ModuleList([
            ResnetBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_res_blocks)
        ])
        if use_attn:
            self.attn = nn.ModuleList([AttnBlock(out_channels) for _ in range(num_res_blocks)])
        else:
            self.attn = nn.ModuleList([nn.Identity() for _ in range(num_res_blocks)])
        if downsample:
            self.downsample = ar.AdaptiveConvDownsample(out_channels, kernel_size=3)
        else:
            self.register_module('downsample', None)

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: torch.Tensor, spacing: torch.Tensor):
        for block, attn in zip(self.block, self.attn):
            if self.training and self.gradient_checkpointing:
                x = checkpoint(cytoolz.compose(attn, block), x)
            else:
                x = block(x)
                x = attn(x)
        if self.downsample is not None:
            x, spacing, downsample_mask = self.downsample(x, spacing)
        else:
            downsample_mask = x.new_zeros(x.shape[0], 3, dtype=bool)
        return x, spacing, downsample_mask

class Encoder(nn.Module):
    down: Sequence[EncoderDownLayer] | nn.ModuleList

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        layer_channels: Sequence[int],
        num_res_blocks: int,
        # num_interpolations: int = 0,
        attn_layer_ids: Sequence[int] | None = None,
        mid_attn: bool = True,
        dropout: float = 0.,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        num_layers = len(layer_channels)
        attn_layer_ids = attn_layer_ids or []
        self.conv_in = InflatableInputConv3d(in_channels, layer_channels[0], kernel_size=3, padding=1)
        # downsampling
        self.down = nn.ModuleList([
            EncoderDownLayer(
                layer_channels[0] if i == 0 else layer_channels[i - 1],
                layer_channels[i],
                num_res_blocks,
                use_attn=i in attn_layer_ids,
                downsample=i != num_layers - 1,
                gradient_checkpointing=gradient_checkpointing,
            )
            for i in range(num_layers)
        ])
        last_channels = layer_channels[-1]

        # middle
        self.mid = nn.Sequential(OrderedDict(
            block_1=ResnetBlock(last_channels, dropout=dropout),
            **(
                dict(attn_1=AttnBlock(last_channels)) if mid_attn
                else {}
            ),
            block_2=ResnetBlock(last_channels, dropout=dropout),
        ))

        # end
        self.norm_out = get_norm_layer(last_channels)
        self.conv_out = InflatableConv3d(last_channels, z_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, spacing: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        x = self.conv_in(x)
        downsample_masks = []
        for down in self.down:
            x, spacing, downsample_mask = down(x, spacing)
            downsample_masks.append(downsample_mask)
        x = self.mid(x)
        # end
        x = self.norm_out(x)
        x = act_layer(x)
        z = self.conv_out(x)
        return z, spacing, downsample_masks

class DecoderUpLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        use_attn: bool,
        upsample: bool,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        num_res_blocks += 1  # following VQGAN
        self.block = nn.ModuleList([
            ResnetBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_res_blocks)
        ])
        if use_attn:
            self.attn = nn.ModuleList([AttnBlock(out_channels) for _ in range(num_res_blocks)])
        else:
            self.attn = nn.ModuleList([nn.Identity() for _ in range(num_res_blocks)])
        if upsample:
            self.upsample = ar.AdaptiveConvUpsample(out_channels)
        else:
            self.register_module('upsample', None)

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: torch.Tensor, upsample_mask: torch.Tensor | None = None):
        for block, attn in zip(self.block, self.attn):
            if self.training and self.gradient_checkpointing:
                x = checkpoint(cytoolz.compose(attn, block), x)
            else:
                x = block(x)
                x = attn(x)
        if self.upsample is not None:
            x = self.upsample(x, upsample_mask)
        return x

class Decoder(nn.Module):
    up: Sequence[DecoderUpLayer] | nn.ModuleList

    def __init__(
        self,
        in_channels: int,  # number of channels of output reconstruction
        z_channels: int,
        layer_channels: Sequence[int],
        num_res_blocks: int,
        attn_layer_ids: Sequence[int] | None = None,
        mid_attn: bool = True,
        dropout: float = 0.0,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        num_layers = len(layer_channels)
        attn_layer_ids = attn_layer_ids or []
        last_channels = layer_channels[-1]

        # z to last feature map channels
        self.conv_in = InflatableConv3d(z_channels, last_channels, kernel_size=3, padding=1)

        # middle
        self.mid = nn.Sequential(OrderedDict(
            block_1=ResnetBlock(last_channels, dropout=dropout),
            **(
                dict(attn_1=AttnBlock(last_channels)) if mid_attn
                else {}
            ),
            block_2=ResnetBlock(last_channels, dropout=dropout),
        ))

        # upsampling
        self.up = nn.ModuleList([
            DecoderUpLayer(
                layer_channels[i + 1] if i + 1 < num_layers else layer_channels[-1],
                layer_channels[i],
                num_res_blocks,
                use_attn=i in attn_layer_ids,
                upsample=i != 0,
                gradient_checkpointing=gradient_checkpointing,
            )
            for i in range(num_layers)
        ])

        # end
        first_channels = layer_channels[0]
        self.norm_out = get_norm_layer(first_channels)
        self.conv_out = InflatableOutputConv3d(first_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, upsample_masks: list[torch.Tensor]):
        # convert z to last feature map channels
        x = self.conv_in(z)

        # middle
        x = self.mid(x)

        # upsampling
        for up, upsample_mask in zip(reversed(self.up), reversed([None] + upsample_masks[:-1])):
            x = up(x, upsample_mask)

        # end
        x = self.norm_out(x)
        x = act_layer(x)
        x = self.conv_out(x)
        return x

class VQGAN(LightningModule):
    def __init__(
        self,
        z_channels: int,
        embedding_dim: int,
        ed_kwargs: dict,
        vq_kwargs: dict,
        loss_kwargs: dict,
        force_rgb: bool = True,
        num_pre_downsamples: int = 0,
        optimizer: dict | None = None,
        lr_scheduler_config: LRSchedulerConfig | None = None,
        disc_optimizer: dict | None = None,
        disc_lr_scheduler_config: LRSchedulerConfig | None = None,
    ):
        super().__init__()
        self.encoder = Encoder(**ed_kwargs)
        self.decoder = Decoder(**ed_kwargs)
        self.quantize = VectorQuantizer(**vq_kwargs)
        self.quant_conv = InflatableConv3d(z_channels, embedding_dim, 1)
        self.post_quant_conv = InflatableConv3d(embedding_dim, z_channels, 1)
        self.loss = VQGANLoss(**loss_kwargs)
        self.force_rgb = force_rgb
        self.num_pre_downsamples = num_pre_downsamples
        if num_pre_downsamples > 0:
            self.pre_downsample = ar.AdaptiveInterpolateDownsample()
            self.post_upsample = ar.AdaptiveUpsample()
        else:
            self.register_module('pre_downsample', None)
            self.register_module('post_upsample', None)

        self.optimizer = optimizer
        self.lr_scheduler_config = lr_scheduler_config
        self.disc_optimizer = disc_optimizer
        self.disc_lr_scheduler_config = disc_lr_scheduler_config
        self.automatic_optimization = False

    def encode(self, x: torch.Tensor, spacing: torch.Tensor | None = None):
        if spacing is None:
            spacing = x.new_ones(x.shape[0], 3)
        pre_downsample_masks = []
        for _ in range(self.num_pre_downsamples):
            x, spacing, mask = self.pre_downsample(x, spacing)
            pre_downsample_masks.append(mask)
        x = ensure_rgb(x, self.force_rgb)
        z, spacing, downsample_masks = self.encoder(x, spacing)
        z = self.quant_conv(z)
        quant_out = self.quantize(z)
        return quant_out, downsample_masks, pre_downsample_masks

    def decode(
        self,
        z_q: torch.Tensor,
        upsample_masks: list[torch.Tensor],
        post_upsample_masks: list[torch.Tensor],
        to_gray: bool = False,
    ):
        z_q = self.post_quant_conv(z_q)
        x_rec = self.decoder(z_q, upsample_masks)
        if to_gray:
            x_rec = rgb_to_gray(x_rec)
        for post_upsample_mask in reversed(post_upsample_masks):
            x_rec = self.post_upsample(x_rec, post_upsample_mask)
        return x_rec

    def forward(self, x: torch.Tensor, spacing: torch.Tensor | None = None) -> tuple[torch.Tensor, VectorQuantizerOutput]:
        quant_out, downsample_masks, pre_downsample_masks = self.encode(x, spacing)
        x_rec = self.decode(
            quant_out.z_q,
            downsample_masks,
            pre_downsample_masks,
            self.force_rgb and x.shape[1] == 1,
        )
        return x_rec, quant_out

    def log_dict_split(self, data: dict, split: str, batch_size: int | None = None):
        self.log_dict({f'{split}/{k}': v for k, v in data.items()}, batch_size=batch_size, sync_dist=True)

    def forward_batch(self, batch: list[dict]) -> dict:
        batch_size = len(batch)
        batch_log_dict = None
        for sample in batch:
            x, spacing = cytoolz.get([DataKey.IMG, DataKey.SPACING], sample)
            x = x[None]
            spacing = spacing[None]
            x_rec, quant_out = self(x, spacing)
            loss, disc_loss, log_dict = self.loss(
                x, x_rec, spacing, quant_out.loss, self.global_step,
                adaptive_weight_ref=self.decoder.conv_out.weight,
            )
            if self.training:
                self.manual_backward(loss / batch_size)
                self.manual_backward(disc_loss / batch_size)
            if batch_log_dict is None:
                batch_log_dict = log_dict
            else:
                for k, v in log_dict.items():
                    batch_log_dict[k] += v
        for k in batch_log_dict:
            batch_log_dict[k] /= batch_size
        return batch_log_dict

    def on_train_batch_start(self, *args, **kwargs):
        lr_scheduler, disc_lr_scheduler = self.lr_schedulers()
        lr_scheduler.step_update(self.global_step)
        disc_lr_scheduler.step_update(self.global_step)

    def lr_scheduler_step(self, *args, **kwargs) -> None:
        # make lightning happy with the incompatible API
        pass

    def training_step(self, batch: list[dict], *args, **kwargs):
        optimizer, disc_optimizer = self.optimizers()
        optimizer.zero_grad()
        disc_optimizer.zero_grad()
        if self.quantize.mode == 'gumbel' and not self.quantize.hard_gumbel:
            self.quantize.adjust_temperature(self.global_step)
        batch_log_dict = self.forward_batch(batch)
        optimizer.step()
        disc_optimizer.step()
        self.log_dict_split(batch_log_dict, 'train', len(batch))

    def validation_step(self, batch: list[dict], *args, **kwargs):
        batch_log_dict = self.forward_batch(batch)
        self.log_dict_split(batch_log_dict, 'val', len(batch))

    def configure_optimizers(self):
        no_weight_decay_keys = get_no_weight_decay_keys(self)
        optimizer = instantiate_class(
            split_by_weight_decay(
                [
                    (name, param)
                    for child_name, child in self.named_children() if child_name != 'loss'
                    for name, param in child.named_parameters(prefix=child_name) if param.requires_grad
                ],
                no_weight_decay_keys
            ),
            self.optimizer,
        )
        disc_optimizer = instantiate_class(
            split_by_weight_decay(
                [
                    (name, param)
                    for name, param in self.loss.named_parameters(prefix='loss') if param.requires_grad
                ],
                no_weight_decay_keys,
            ),
            self.disc_optimizer,
        )
        from timm.scheduler.scheduler import Scheduler as TIMMScheduler
        lr_scheduler_config = copy(self.lr_scheduler_config)
        lr_scheduler_config.scheduler = instantiate_class(optimizer, lr_scheduler_config.scheduler)
        assert isinstance(lr_scheduler_config.scheduler, TIMMScheduler)
        disc_lr_scheduler_config = copy(self.disc_lr_scheduler_config)
        disc_lr_scheduler_config.scheduler = instantiate_class(disc_optimizer, disc_lr_scheduler_config.scheduler)
        assert isinstance(disc_lr_scheduler_config.scheduler, TIMMScheduler)
        return [optimizer, disc_optimizer], [vars(lr_scheduler_config), vars(disc_lr_scheduler_config)]

#     def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
#         log = dict()
#         x = self.get_input(batch, self.image_key)
#         x = x.to(self.device)
#         if only_inputs:
#             log["inputs"] = x
#             return log
#         xrec, _ = self(x)
#         if x.shape[1] > 3:
#             # colorize with random projection
#             assert xrec.shape[1] > 3
#             x = self.to_rgb(x)
#             xrec = self.to_rgb(xrec)
#         log["inputs"] = x
#         log["reconstructions"] = xrec
#         if plot_ema:
#             with self.ema_scope():
#                 xrec_ema, _ = self(x)
#                 if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
#                 log["reconstructions_ema"] = xrec_ema
#         return log
#
#     def to_rgb(self, x):
#         assert self.image_key == "segmentation"
#         if not hasattr(self, "colorize"):
#             self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
#         x = F.Conv3d(x, weight=self.colorize)
#         x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
#         return x
