from typing import Literal

import torch
from torch import nn

from luolib.models.adaptive_resampling import AdaptiveDownsample
from luolib.models.blocks import InflatableConv3d, InflatableInputConv3d

from .lpips import LPIPS

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int, num_downsample_layers: int = 2, base_channels: int = 64):
        super().__init__()
        norm_layer = nn.InstanceNorm3d

        layer_channels = [
            base_channels << min(i, 3)
            for i in range(num_downsample_layers + 1)
        ]
        self.main = nn.Sequential()
        self.main.extend([
            AdaptiveDownsample(in_channels, layer_channels[0], 4, InflatableInputConv3d),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        for i in range(1, num_downsample_layers + 1):  # gradually increase the number of filters
            self.main.extend([
                AdaptiveDownsample(layer_channels[i - 1], layer_channels[i], kernel_size=4) if i < num_downsample_layers
                else InflatableConv3d(layer_channels[i - 1], layer_channels[i], 4, stride=1, padding=2),
                norm_layer(layer_channels[i]),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.main.append(InflatableConv3d(layer_channels[-1], 1, kernel_size=4, stride=1, padding=2))

    def forward(self, x: torch.Tensor):
        return self.main(x)

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

class VQGANLoss(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        quant_weight: float = 1.0,
        rec_loss: Literal['l1', 'l2'] = 'l1',
        rec_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_num_downsample_layers: int = 3,
        disc_base_channels: int = 64,
        gan_weight: float = 1.0,
        gan_start_step: int = 0,
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
        print(f'{self.__class__.__name__}: Running with LPIPS.')
        self.perceptual_weight = perceptual_weight
        self.gan_start_step = gan_start_step
        self.gan_weight = gan_weight

        self.discriminator = PatchDiscriminator(in_channels, disc_num_downsample_layers, disc_base_channels)
        print(f'{self.__class__.__name__} running with hinge W-GAN loss.')

    def forward(
        self,
        x: torch.Tensor,
        x_rec: torch.Tensor,
        quant_loss: torch.Tensor,
        global_step: int = 0,
        adaptive_weight_ref: nn.Parameter | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        x = x.contiguous()
        x_rec = x_rec.contiguous()
        # generator part
        rec_loss = self.rec_loss(x, x_rec)
        if self.perceptual_weight > 0:
            perceptual_loss = self.perceptual_loss(x, x_rec)
        else:
            perceptual_loss = torch.zeros_like(rec_loss)
        vq_loss = rec_loss + self.perceptual_weight * perceptual_loss + self.quant_weight * quant_loss
        score_fake = self.discriminator(x_rec)
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
        score_real = self.discriminator(x.detach())
        score_fake = self.discriminator(x_rec.detach())
        disc_loss = 0.5 * ((1 - score_real).relu() + (1 + score_fake).relu())
        return loss, disc_loss, {
            'loss': loss,
            'rec_loss': rec_loss,
            'perceptual_loss': perceptual_loss,
            'quant_loss': quant_loss,
            'vq_loss': vq_loss,
            'gan_loss': gan_loss,
            'disc_loss': disc_loss,
        }
