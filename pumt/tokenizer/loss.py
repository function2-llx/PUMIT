from typing import Literal

import einops
import torch
from torch import nn

from luolib.models.adaptive_resampling import AdaptiveConvDownsample
from luolib.models.blocks import InflatableConv3d, InflatableInputConv3d

from .lpips import LPIPS
from .utils import ensure_rgb

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int, num_downsample_layers: int = 2, base_channels: int = 64):
        super().__init__()

        layer_channels = [
            base_channels << min(i, 3)
            for i in range(num_downsample_layers + 1)
        ]
        self.main = nn.ModuleList()
        self.main.extend([
            AdaptiveConvDownsample(in_channels, layer_channels[0], 4, InflatableInputConv3d),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        for i in range(1, num_downsample_layers + 1):  # gradually increase the number of filters
            self.main.extend([
                AdaptiveConvDownsample(layer_channels[i - 1], layer_channels[i], kernel_size=4, bias=False) if i < num_downsample_layers
                else InflatableConv3d(layer_channels[i - 1], layer_channels[i], 4, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(layer_channels[i], affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.main.append(InflatableConv3d(layer_channels[-1], 1, kernel_size=4, stride=1, padding=1))

    def forward(self, x: torch.Tensor, spacing: torch.Tensor):
        for module in self.main:
            if isinstance(module, AdaptiveConvDownsample):
                x, spacing, _ = module(x, spacing)
            else:
                x = module(x)
        return x

@torch.no_grad()
def calculate_adaptive_weight(
    loss1: torch.Tensor,
    loss2: torch.Tensor,
    ref_param: nn.Parameter,
    eps: float = 1e-4,
    minv: float = 0.,
    maxv: float = 1e4,
):
    grad1, = torch.autograd.grad(loss1, ref_param, retain_graph=True)
    grad2, = torch.autograd.grad(loss2, ref_param, retain_graph=True)
    weight = grad1.norm() / (grad2.norm() + eps)
    weight.clamp_(minv, maxv)
    return weight

def hinge_loss(score_real: torch.Tensor, score_fake: torch.Tensor):
    return (1 - score_real).relu().mean() + (1 + score_fake).relu().mean()

class VQGANLoss(nn.Module):
    def __init__(
        self,
        in_channels: int,
        quant_weight: float = 1.0,
        rec_loss: Literal['l1', 'l2'] = 'l1',
        rec_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        max_perceptual_slices: int = 16,
        gan_weight: float = 1.0,
        gan_start_step: int = 0,
        disc_num_downsample_layers: int = 3,
        disc_base_channels: int = 64,
        disc_force_rgb: bool = True,
    ):
        super().__init__()
        self.quant_weight = quant_weight
        self.rec_weight = rec_weight
        match rec_loss:
            case 'l1':
                self.rec_loss = nn.L1Loss()
            case 'l2':
                self.rec_loss = nn.MSELoss()
            case _:
                raise ValueError
        self.perceptual_loss = LPIPS().eval()
        print(f'{self.__class__.__name__}: running with LPIPS')
        self.perceptual_weight = perceptual_weight
        self.max_perceptual_slices = max_perceptual_slices
        self.gan_start_step = gan_start_step
        self.gan_weight = gan_weight

        self.discriminator = PatchDiscriminator(in_channels, disc_num_downsample_layers, disc_base_channels)
        self.disc_force_rgb = disc_force_rgb
        print(f'{self.__class__.__name__}: running with hinge W-GAN loss')

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if not any(key.startswith(f'{prefix}discriminator.main') for key in state_dict):
            # checkpoint of VQGAN trained with Gumbel softmax is so amazing
            disc_prefix = f'{prefix}discriminator.'
            for key in list(state_dict.keys()):
                if key.startswith(f'{disc_prefix}discriminators.'):
                    weight = state_dict.pop(key)
                    disc_id, suffix = key[len(disc_prefix) + len('discriminators.'):].split('.', 1)
                    # pick the discriminator with patch size = 8
                    if int(disc_id) == 1 and not any(
                        key.endswith(bn_var) for bn_var in ['running_mean', 'running_var', 'num_batches_tracked']
                    ):
                        state_dict[f'{disc_prefix}{suffix}'] = weight
        perceptual_loss_prefix = f'{prefix}perceptual_loss.'
        for key in list(state_dict.keys()):
            if key.startswith(perceptual_loss_prefix):
                state_dict.pop(key)
        # make `model.load_state_dict` happy about no missing keys
        state_dict.update(self.perceptual_loss.state_dict(prefix=perceptual_loss_prefix))

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        x: torch.Tensor,
        x_rec: torch.Tensor,
        spacing: torch.Tensor,
        quant_loss: torch.Tensor,
        global_step: int = 0,
        adaptive_weight_ref: nn.Parameter | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        x = x.contiguous()
        x_rec = x_rec.contiguous()

        rec_loss = self.rec_loss(x, x_rec)
        if self.perceptual_weight > 0:
            if self.training and x.shape[2] > self.max_perceptual_slices:
                slice_indexes = einops.repeat(
                    x.new_ones(x.shape[0], x.shape[2]).multinomial(self.max_perceptual_slices),
                    'n d -> n c d h w', c=x.shape[1], h=x.shape[3], w=x.shape[4],
                )
                sampled_x = x.gather(dim=2, index=slice_indexes)
                sampled_x_rec = x_rec.gather(dim=2, index=slice_indexes)
            else:
                sampled_x = x
                sampled_x_rec = x_rec
            perceptual_loss = self.perceptual_loss(
                einops.rearrange(sampled_x, 'n c d h w -> (n d) c h w'),
                einops.rearrange(sampled_x_rec, 'n c d h w -> (n d) c h w'),
            )
        else:
            perceptual_loss = torch.zeros_like(rec_loss)
        vq_loss = rec_loss + self.perceptual_weight * perceptual_loss + self.quant_weight * quant_loss

        x = ensure_rgb(x, self.disc_force_rgb)
        x_rec = ensure_rgb(x_rec, self.disc_force_rgb)
        # generator part
        if self.training:
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)
        score_fake = self.discriminator(x_rec, spacing)
        gan_loss = -score_fake.mean()
        if self.training:
            if global_step >= self.gan_start_step:
                gan_weight = self.gan_weight * calculate_adaptive_weight(vq_loss, gan_loss, ref_param=adaptive_weight_ref)
            else:
                gan_weight = 0
        else:
            gan_weight = self.gan_weight
        loss = vq_loss + gan_weight * gan_loss

        # discriminator part
        if self.training:
            self.discriminator.train()
            if torch.is_grad_enabled():
                self.discriminator.requires_grad_(True)
        score_real = self.discriminator(x.detach(), spacing)
        score_fake = self.discriminator(x_rec.detach(), spacing)
        disc_loss = 0.5 * hinge_loss(score_real, score_fake)
        return loss, disc_loss, {
            'loss': loss,
            'rec_loss': rec_loss,
            'perceptual_loss': perceptual_loss,
            'quant_loss': quant_loss,
            'vq_loss': vq_loss,
            'gan_loss': gan_loss,
            'disc_loss': disc_loss,
        }
