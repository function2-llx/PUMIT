from typing import Literal

import einops
from lightning import Fabric
import torch
from torch import nn

from luolib.models import spadop

from .discriminator import PatchDiscriminatorBase
from .lpips import LPIPS
from .quantize import VectorQuantizerOutput

@torch.no_grad()
def calculate_adaptive_weight(
    loss1: torch.Tensor,
    loss2: torch.Tensor,
    ref_param: nn.Parameter,
    eps: float = 1e-6,
    minv: float = 0.,
    maxv: float = 1e4,
    fabric: Fabric | None = None,
):
    """calculate adaptive weight balancing the gradient norm for loss2"""
    grad1, = torch.autograd.grad(loss1, ref_param, retain_graph=True)
    grad2, = torch.autograd.grad(loss2, ref_param, retain_graph=True)
    if fabric is not None:
        grad1 = fabric.all_reduce(grad1.contiguous())
        grad2 = fabric.all_reduce(grad2.contiguous())
    weight = grad1.norm() / (grad2.norm() + eps)
    weight.clamp_(minv, maxv)
    return weight

def hinge_loss(score_real: torch.Tensor, score_fake: torch.Tensor):
    return (1 - score_real).relu().mean() + (1 + score_fake).relu().mean()

class VQVTLoss(nn.Module):
    def __init__(
        self,
        quant_weight: float,
        entropy_weight: float,
        discriminator: PatchDiscriminatorBase,
        rec_loss: Literal['l1', 'l2'] = 'l1',
        rec_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        max_perceptual_slices: int = 16,
        gan_weight: float = 1.0,
        adaptive_gan_weight: bool = False,
    ):
        super().__init__()
        self.quant_weight = quant_weight
        self.entropy_weight = entropy_weight
        self.rec_weight = rec_weight
        match rec_loss:
            case 'l1':
                self.rec_loss = nn.L1Loss()
            case 'l2':
                self.rec_loss = nn.MSELoss()
            case _:
                raise ValueError
        self.perceptual_loss = LPIPS()
        print(f'{self.__class__.__name__}: running with LPIPS')
        self.perceptual_weight = perceptual_weight
        self.max_perceptual_slices = max_perceptual_slices
        self.gan_weight = gan_weight
        self.adaptive_gan_weight = adaptive_gan_weight
        assert discriminator is not None
        self.discriminator = discriminator
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

    def forward_gen(
        self,
        x: spadop.SpatialTensor,
        x_rec: spadop.SpatialTensor,
        vq_out: VectorQuantizerOutput,
        use_gan_loss: bool,
        ref_param: nn.Parameter | None = None,
        fabric: Fabric | None = None,
    ) -> tuple[torch.Tensor, dict]:
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
        vq_loss = rec_loss + self.perceptual_weight * perceptual_loss + self.quant_weight * vq_out.loss
        if vq_out.entropy is not None:
            vq_loss = vq_loss + self.entropy_weight * vq_out.entropy

        score_fake = self.discriminator(x_rec)
        gan_loss = -score_fake.mean()
        if use_gan_loss:
            gan_weight = self.gan_weight
            if self.adaptive_gan_weight:
                gan_weight *= calculate_adaptive_weight(
                    vq_loss.as_subclass(torch.Tensor),
                    gan_loss.as_subclass(torch.Tensor),
                    ref_param,
                    fabric=fabric,
                )
        else:
            gan_weight = 0
        loss = vq_loss + gan_weight * gan_loss
        log_dict = {
            'loss': loss,
            'rec_loss': rec_loss,
            'perceptual_loss': perceptual_loss,
            'quant_loss': vq_out.loss,
            'vq_loss': vq_loss,
            'gan_loss': gan_loss,
            'gan_weight': gan_weight,
        }
        if vq_out.entropy is not None:
            log_dict['entropy'] = vq_out.entropy
            log_dict['diversity'] = vq_out.diversity
        return loss, log_dict

    def forward_disc(self, x: spadop.SpatialTensor, x_rec: spadop.SpatialTensor, log_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        score_real = self.discriminator(x.detach())
        score_fake = self.discriminator(x_rec.detach())
        disc_loss = 0.5 * hinge_loss(score_real, score_fake)
        log_dict['disc_loss'] = disc_loss
        return disc_loss, log_dict
