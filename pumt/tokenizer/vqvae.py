from collections import OrderedDict
from collections.abc import Sequence
from copy import copy
from pathlib import Path

import cytoolz
import numpy as np
from lightning import LightningModule
from lightning.pytorch.cli import instantiate_class
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from luolib.models.utils import split_weight_decay_keys, load_ckpt, create_param_groups
from luolib.utils import DataKey
from luolib.types import LRSchedulerConfig

from .loss import VQGANLoss
from .quantize import VectorQuantizer, VectorQuantizerOutput
from .utils import ensure_rgb, rgb_to_gray
from ..conv import (
    AdaptiveConvDownsample, AdaptiveInterpolationDownsample, AdaptiveUpsample, InflatableConv3d, InflatableInputConv3d,
    InflatableOutputConv3d,
    SpatialTensor,
)

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
            self.downsample = InflatableConv3d(out_channels, out_channels, kernel_size=(2, 3, 3), stride=2, padding=(0, 1, 1))
        else:
            self.register_module('downsample', None)

        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, x: SpatialTensor):
        for block, attn in zip(self.block, self.attn):
            if self.training and self.gradient_checkpointing:
                x = checkpoint(cytoolz.compose(attn, block), x)
            else:
                x = block(x)
                x = attn(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class Encoder(nn.Module):
    down: Sequence[EncoderDownLayer] | nn.ModuleList

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        layer_channels: Sequence[int],
        num_res_blocks: int,
        attn_layer_ids: Sequence[int] | None = None,
        mid_attn: bool = False,
        additional_interpolation: bool = False,
        dropout: float = 0.,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        num_layers = len(layer_channels)
        attn_layer_ids = attn_layer_ids or []
        self.conv_in = InflatableInputConv3d(in_channels, layer_channels[0], kernel_size=3, padding=1)
        self.additional_interpolation = AdaptiveInterpolationDownsample() if additional_interpolation else nn.Identity()
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

    def forward(self, x: SpatialTensor) -> SpatialTensor:
        x = self.conv_in(x)
        x = self.additional_interpolation(x)
        for down in self.down:
            x = down(x)
        x = self.mid(x)
        # end
        x = self.norm_out(x)
        x = act_layer(x)
        z = self.conv_out(x)
        return z

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
            pass
            # self.upsample = pumt.blocks.AdaptiveConvUpsample(out_channels)
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
        mid_attn: bool = False,
        additional_interpolation: bool = False,
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
        self.additional_interpolation = AdaptiveUpsample() if additional_interpolation else nn.Identity()

        # end
        first_channels = layer_channels[0]
        self.norm_out = get_norm_layer(first_channels)
        self.conv_out = InflatableOutputConv3d(first_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor):
        x = self.conv_in(z)
        x = self.mid(x)
        for up in reversed(self.up):
            x = up(x)
        x = self.additional_interpolation(x)
        # end
        x = self.norm_out(x)
        x = act_layer(x)
        x = self.conv_out(x)
        return x

class VQVAEModel(nn.Module):
    def __init__(
        self,
        z_channels: int,
        embedding_dim: int,
        ed_kwargs: dict,
        vq_kwargs: dict,
        force_rgb: bool = True,
    ):
        super().__init__()
        self.encoder = Encoder(**ed_kwargs)
        self.decoder = Decoder(**ed_kwargs)
        self.quant_conv = InflatableConv3d(z_channels, embedding_dim, 1)
        self.quantize = VectorQuantizer(**vq_kwargs)
        self.post_quant_conv = InflatableConv3d(embedding_dim, z_channels, 1)
        self.force_rgb = force_rgb

    def encode(self, x: SpatialTensor) -> SpatialTensor:
        x = x.copy_meta_to(ensure_rgb(x, self.force_rgb))
        return self.encoder(x)

    def do_quantize(self, z: SpatialTensor) -> tuple[SpatialTensor, VectorQuantizerOutput]:
        z = self.quant_conv(z)
        quant_out: VectorQuantizerOutput = self.quantize(z)
        z = self.post_quant_conv(z.copy_meta_to(quant_out.z_q))
        return z, quant_out

    def decode(self, z_q: SpatialTensor, to_gray: bool = False) -> SpatialTensor:
        x_rec = self.decoder(z_q)
        if to_gray:
            x_rec = x_rec.copy_meta_to(rgb_to_gray(x_rec))
        return x_rec

    def forward(self, x: SpatialTensor) -> tuple[torch.Tensor, VectorQuantizerOutput]:
        z = self.encode(x)
        z, quant_out = self.do_quantize(z)
        x_rec = self.decode(z, self.force_rgb and x.shape[1] == 1)
        return x_rec, quant_out

class VQGAN(VQVAEModel, LightningModule):
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
        ckpt_path: Path | None = None,
    ):
        VQVAEModel.__init__(self, z_channels, embedding_dim, ed_kwargs, vq_kwargs, force_rgb, num_pre_downsamples)
        LightningModule.__init__(self)
        self.loss = VQGANLoss(**loss_kwargs)
        self.optimizer = optimizer
        self.lr_scheduler_config = lr_scheduler_config
        self.disc_optimizer = disc_optimizer
        self.disc_lr_scheduler_config = disc_lr_scheduler_config
        load_ckpt(self, ckpt_path)
        self.automatic_optimization = False

    def log_dict_split(self, data: dict, split: str, batch_size: int | None = None):
        self.log_dict({f'{split}/{k}': v for k, v in data.items()}, batch_size=batch_size, sync_dist=True)

    def lr_scheduler_step(self, *args, **kwargs) -> None:
        # make lightning happy with the incompatible API: https://github.com/Lightning-AI/lightning/issues/18074
        pass

    def configure_optimizers(self):
        decay_keys, no_decay_keys = split_weight_decay_keys(self)
        optimizer = instantiate_class(
            create_param_groups(
                [
                    (name, param)
                    for child_name, child in self.named_children() if child_name != 'loss'
                    for name, param in child.named_parameters(prefix=child_name) if param.requires_grad
                ],
                decay_keys,
                no_decay_keys,
            ),
            self.optimizer,
        )
        disc_optimizer = instantiate_class(
            create_param_groups(
                [
                    (name, param)
                    for name, param in self.loss.named_parameters(prefix='loss') if param.requires_grad
                ],
                decay_keys,
                no_decay_keys,
            ),
            self.disc_optimizer,
        )
        from timm.scheduler.scheduler import Scheduler as TIMMScheduler
        def instantiate_and_check(config: LRSchedulerConfig, optimizer):
            config = copy(config)
            config.scheduler = instantiate_class(optimizer, config.scheduler)
            assert isinstance(config.scheduler, TIMMScheduler)
            assert config.interval == 'step'
            return vars(config)

        return [optimizer, disc_optimizer], [
            instantiate_and_check(self.lr_scheduler_config, optimizer),
            instantiate_and_check(self.disc_lr_scheduler_config, disc_optimizer)
        ]

    def on_fit_start(self) -> None:
        from lightning.pytorch.loops import _TrainingEpochLoop
        class MyTrainingEpochLoop(_TrainingEpochLoop):
            @property
            def global_step(self):
                # https://github.com/Lightning-AI/lightning/issues/17958
                return super().global_step // 2
        self.trainer.fit_loop.epoch_loop.__class__ = MyTrainingEpochLoop

    def on_train_batch_start(self, *args, **kwargs):
        for config in self.trainer.lr_scheduler_configs:
            if self.global_step % config.frequency == 0:
                config.scheduler.step_update(self.global_step)

    def configure_gradient_clipping(self, optimizer, *args, **kwargs) -> None:
        self.clip_gradients(optimizer, 1, 'norm')

    def training_step(self, batch: list[dict], *args, **kwargs):
        optimizer, disc_optimizer = self.optimizers()
        optimizer.zero_grad()
        disc_optimizer.zero_grad()
        if self.quantize.mode == 'gumbel' and not self.quantize.hard_gumbel:
            self.quantize.adjust_temperature(self.global_step, self.trainer.max_steps)
        batch_size = len(batch)
        batch_log_dict = None
        batch_loss, batch_disc_loss = None, None
        for sample in batch:
            x, spacing = cytoolz.get([DataKey.IMG, DataKey.SPACING], sample)
            x = x[None]
            spacing = spacing[None]
            self.toggle_optimizer(optimizer)
            x_rec, quant_out = self(x, spacing)
            self.loss.adjust_gan_weight(self.global_step)
            loss, log_dict = self.loss.forward_gen(x, x_rec, spacing, quant_out.loss)
            if batch_loss is None:
                batch_loss = loss
            else:
                batch_loss += loss
            self.untoggle_optimizer(optimizer)
            self.toggle_optimizer(disc_optimizer)
            disc_loss = self.loss.forward_disc(x, x_rec, spacing, log_dict)
            if batch_disc_loss is None:
                batch_disc_loss = disc_loss
            else:
                batch_disc_loss += disc_loss
            self.untoggle_optimizer(disc_optimizer)
            if batch_log_dict is None:
                batch_log_dict = log_dict
            else:
                for k, v in log_dict.items():
                    batch_log_dict[k] += v
        self.manual_backward(batch_loss)
        self.manual_backward(batch_disc_loss)
        optimizer.step()
        disc_optimizer.step()
        for k in batch_log_dict:
            batch_log_dict[k] /= batch_size
        self.log_dict_split(batch_log_dict, 'train', len(batch))

    def validation_step(self, batch: list[dict], *args, **kwargs):
        batch_size = len(batch)
        batch_log_dict = None
        for sample in batch:
            x, spacing = cytoolz.get([DataKey.IMG, DataKey.SPACING], sample)
            x = x[None]
            spacing = spacing[None]
            x_rec, quant_out = self(x, spacing)
            loss, log_dict = self.loss.forward_gen(x, x_rec, spacing, quant_out.loss)
            self.loss.forward_disc(x, x_rec, spacing, log_dict)
            if batch_log_dict is None:
                batch_log_dict = log_dict
            else:
                for k, v in log_dict.items():
                    batch_log_dict[k] += v
        for k in batch_log_dict:
            batch_log_dict[k] /= batch_size
        self.log_dict_split(batch_log_dict, 'val', len(batch))
