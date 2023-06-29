from collections import OrderedDict
from collections.abc import Sequence
import itertools as it

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from luolib.models.adaptive_resampling import AdaptiveDownsample, AdaptiveUpsample
from luolib.models.blocks import InflatableConv3d, InflatableInputConv3d, InflatableOutputConv3d
from luolib.utils import PathLike

from .quantize import VectorQuantizer

def get_norm_layer(in_channels: int, num_groups: int = 32):
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)

def nonlinearity(x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
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
    ):
        super().__init__()
        self.block = nn.ModuleList([
            ResnetBlock(in_channels if i == 0 else out_channels, out_channels)
            for i in range(num_res_blocks)
        ])
        if use_attn:
            self.attn = nn.ModuleList([AttnBlock(out_channels) for _ in range(num_res_blocks)])
        else:
            self.attn = nn.ModuleList([nn.Identity() for _ in range(num_res_blocks)])
        if downsample:
            self.downsample = AdaptiveDownsample(out_channels, kernel_size=3)
        else:
            self.register_module('downsample', None)

    def forward(self, x: torch.Tensor, spacing: torch.Tensor):
        for block, attn in zip(self.block, self.attn):
            x = block(x)
            x = attn(x)
        if self.downsample is not None:
            x, spacing, downsample_mask = self.downsample.forward(x, spacing)
        else:
            downsample_mask = x.new_zeros(x.shape[0], 3, dtype=bool)
        return x, spacing, downsample_mask

class Encoder(nn.Module):
    down: Sequence[EncoderDownLayer] | nn.ModuleList

    def __init__(
        self,
        in_channels: int,
        layer_channels: Sequence[int],
        num_res_blocks: int,
        z_channels: int,
        dropout: float = 0.,
    ):
        super().__init__()
        num_layers = len(layer_channels)

        # downsampling
        self.conv_in = InflatableInputConv3d(in_channels, layer_channels[0], kernel_size=3, padding=1)
        self.down = nn.ModuleList([
            EncoderDownLayer(
                layer_channels[0] if i == 0 else layer_channels[i - 1],
                layer_channels[i],
                num_res_blocks,
                # apply attention when stride >= 16
                # in the original implementation, they apply attention when the resolution reaches 16 (from 256)
                use_attn=i >= 4,
                downsample=i != num_layers - 1,
            )
            for i in range(num_layers)
        ])
        last_channels = layer_channels[-1]

        # middle
        self.mid = nn.Sequential(OrderedDict(
            block_1=ResnetBlock(last_channels, dropout=dropout),
            attn_1=AttnBlock(last_channels),
            block_2=ResnetBlock(last_channels, dropout=dropout),
        ))

        # end
        self.norm_out = get_norm_layer(last_channels)
        self.conv_out = InflatableConv3d(last_channels, z_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, spacing: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        x = self.conv_in(x)
        downsample_masks = []
        for down in self.down:
            x, spacing, downsample_mask = down.forward(x, spacing)
            downsample_masks.append(downsample_mask)
        x = self.mid(x)
        # end
        x = self.norm_out(x)
        x = nonlinearity(x)
        x = self.conv_out(x)
        return x, spacing, downsample_masks

class DecoderUpLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        use_attn: bool,
        upsample: bool,
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
            self.upsample = AdaptiveUpsample(out_channels)
        else:
            self.register_module('upsample', None)

    def forward(self, x: torch.Tensor, upsample_mask: torch.Tensor | None = None):
        for block, attn in zip(self.block, self.attn):
            x = block(x)
            x = attn(x)
        if self.upsample is not None:
            x = self.upsample.forward(x, upsample_mask)
        return x

class Decoder(nn.Module):
    up: Sequence[DecoderUpLayer] | nn.ModuleList

    def __init__(
        self,
        out_channels: int,
        layer_channels: Sequence[int],
        num_res_blocks: int,
        z_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        num_layers = len(layer_channels)
        last_channels = layer_channels[-1]

        # z to last feature map channels
        self.conv_in = InflatableConv3d(z_channels, last_channels, kernel_size=3, padding=1)

        # middle
        self.mid = nn.Sequential(OrderedDict(
            block_1=ResnetBlock(last_channels, dropout=dropout),
            attn_1=AttnBlock(last_channels),
            block_2=ResnetBlock(last_channels, dropout=dropout),
        ))

        # upsampling
        self.up = nn.ModuleList([
            DecoderUpLayer(
                layer_channels[i + 1] if i + 1 < num_layers else layer_channels[-1],
                layer_channels[i],
                num_res_blocks,
                use_attn=i >= 4,
                upsample=i != 0,
            )
            for i in range(num_layers)
        ])

        # end
        first_channels = layer_channels[0]
        self.norm_out = get_norm_layer(first_channels)
        self.conv_out = InflatableOutputConv3d(first_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor, upsample_masks: list[torch.Tensor]):
        # convert z to last feature map channels
        x = self.conv_in(z)

        # middle
        x = self.mid(x)

        # upsampling
        for up, upsample_mask in zip(reversed(self.up), reversed(upsample_masks[:-1])):
            x = up.forward(x, upsample_mask)
        x = self.up[0](x)

        # end
        x = self.norm_out(x)
        x = nonlinearity(x)
        x = self.conv_out(x)
        return x

class VQModel(pl.LightningModule):
    def __init__(
        self,
        in_channels: int,
        layer_channels: Sequence[int],
        num_res_blocks: int,
        z_channels: int,
        # lossconfig,
        num_embeddings: int,
        embed_dim: int,
        # ckpt_path=None,
        dropout: float = 0.,
        remap: PathLike | None = None,
        sane_index_shape: bool = False,  # tell vector quantizer to return indices as bhw
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = num_embeddings
        self.encoder = Encoder(in_channels, layer_channels, num_res_blocks, z_channels, dropout)
        self.decoder = Decoder(in_channels, layer_channels, num_res_blocks, z_channels, dropout)
        self.quantize = VectorQuantizer(num_embeddings, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = InflatableConv3d(z_channels, embed_dim, 1)
        self.post_quant_conv = InflatableConv3d(embed_dim, z_channels, 1)
        # self.loss = instantiate_from_config(lossconfig)

        # self.scheduler_config = scheduler_config
        # self.lr_g_factor = lr_g_factor
        self.gradient_checkpointing = gradient_checkpointing

    #     @contextmanager
    #     def ema_scope(self, context=None):
    #         if self.use_ema:
    #             self.model_ema.store(self.parameters())
    #             self.model_ema.copy_to(self)
    #             if context is not None:
    #                 print(f"{context}: Switched to EMA weights")
    #         try:
    #             yield None
    #         finally:
    #             if self.use_ema:
    #                 self.model_ema.restore(self.parameters())
    #                 if context is not None:
    #                     print(f"{context}: Restored training weights")
    #
    #     def init_from_ckpt(self, path, ignore_keys=list()):
    #         sd = torch.load(path, map_location="cpu")["state_dict"]
    #         keys = list(sd.keys())
    #         for k in keys:
    #             for ik in ignore_keys:
    #                 if k.startswith(ik):
    #                     print("Deleting key {} from state_dict.".format(k))
    #                     del sd[k]
    #         missing, unexpected = self.load_state_dict(sd, strict=False)
    #         print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
    #         if len(missing) > 0:
    #             print(f"Missing Keys: {missing}")
    #             print(f"Unexpected Keys: {unexpected}")
    #
    #     def on_train_batch_end(self, *args, **kwargs):
    #         if self.use_ema:
    #             self.model_ema(self)
    #
    def encode(self, x: torch.Tensor, spacing: torch.Tensor | None = None):
        if spacing is None:
            spacing = x.new_ones(x.shape[0], 3)
        if self.training and self.gradient_checkpointing:
            h, spacing, downsample_masks = checkpoint(self.encoder, x, spacing)
        else:
            h, spacing, downsample_masks = self.encoder.forward(x, spacing)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize.forward(h)
        return quant, downsample_masks, emb_loss, info

    #
    #     def encode_to_prequant(self, x):
    #         h = self.encoder(x)
    #         h = self.quant_conv(h)
    #         return h

    def decode(self, quant: torch.Tensor, upsample_masks: list[torch.Tensor]):
        quant = self.post_quant_conv(quant)
        if self.training and self.gradient_checkpointing:
            dec = checkpoint(self.decoder, quant, upsample_masks)
        else:
            dec = self.decoder.forward(quant, upsample_masks)
        return dec

    #     def decode_code(self, code_b):
    #         quant_b = self.quantize.embed_code(code_b)
    #         dec = self.decode(quant_b)
    #         return dec

    def forward(self, x: torch.Tensor, spacing: torch.Tensor | None = None, return_pred_indices: bool = False):
        quant, downsample_masks, diff, (_, _, ind) = self.encode(x, spacing)
        dec = self.decode(quant, downsample_masks)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

#     def get_input(self, batch, k):
#         x = batch[k]
#         if len(x.shape) == 3:
#             x = x[..., None]
#         x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
#         if self.batch_resize_range is not None:
#             lower_size = self.batch_resize_range[0]
#             upper_size = self.batch_resize_range[1]
#             if self.global_step <= 4:
#                 # do the first few batches with max size to avoid later oom
#                 new_resize = upper_size
#             else:
#                 new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))
#             if new_resize != x.shape[2]:
#                 x = F.interpolate(x, size=new_resize, mode="bicubic")
#             x = x.detach()
#         return x
#
#     def training_step(self, batch, batch_idx, optimizer_idx):
#         # https://github.com/pytorch/pytorch/issues/37142
#         # try not to fool the heuristics
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss, ind = self(x, return_pred_indices=True)
#
#         if optimizer_idx == 0:
#             # autoencode
#             aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
#                                             last_layer=self.get_last_layer(), split="train",
#                                             predicted_indices=ind)
#
#             self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#             return aeloss
#
#         if optimizer_idx == 1:
#             # discriminator
#             discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
#                                                 last_layer=self.get_last_layer(), split="train")
#             self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
#             return discloss
#
#     def validation_step(self, batch, batch_idx):
#         log_dict = self._validation_step(batch, batch_idx)
#         with self.ema_scope():
#             log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
#         return log_dict
#
#     def _validation_step(self, batch, batch_idx, suffix=""):
#         x = self.get_input(batch, self.image_key)
#         xrec, qloss, ind = self(x, return_pred_indices=True)
#         aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
#                                         self.global_step,
#                                         last_layer=self.get_last_layer(),
#                                         split="val" + suffix,
#                                         predicted_indices=ind
#                                         )
#
#         discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
#                                             self.global_step,
#                                             last_layer=self.get_last_layer(),
#                                             split="val" + suffix,
#                                             predicted_indices=ind
#                                             )
#         rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
#         self.log(f"val{suffix}/rec_loss", rec_loss,
#                  prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
#         self.log(f"val{suffix}/aeloss", aeloss,
#                  prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
#         if version.parse(pl.__version__) >= version.parse('1.4.0'):
#             del log_dict_ae[f"val{suffix}/rec_loss"]
#         self.log_dict(log_dict_ae)
#         self.log_dict(log_dict_disc)
#         return self.log_dict
#
#     def configure_optimizers(self):
#         lr_d = self.learning_rate
#         lr_g = self.lr_g_factor * self.learning_rate
#         print("lr_d", lr_d)
#         print("lr_g", lr_g)
#         opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
#                                   list(self.decoder.parameters()) +
#                                   list(self.quantize.parameters()) +
#                                   list(self.quant_conv.parameters()) +
#                                   list(self.post_quant_conv.parameters()),
#                                   lr=lr_g, betas=(0.5, 0.9))
#         opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
#                                     lr=lr_d, betas=(0.5, 0.9))
#
#         if self.scheduler_config is not None:
#             scheduler = instantiate_from_config(self.scheduler_config)
#
#             print("Setting up LambdaLR scheduler...")
#             scheduler = [
#                 {
#                     'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
#                     'interval': 'step',
#                     'frequency': 1
#                 },
#                 {
#                     'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
#                     'interval': 'step',
#                     'frequency': 1
#                 },
#             ]
#             return [opt_ae, opt_disc], scheduler
#         return [opt_ae, opt_disc], []
#
#     def get_last_layer(self):
#         return self.decoder.conv_out.weight
#
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
