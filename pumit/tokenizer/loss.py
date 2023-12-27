from typing import Literal

from lightning import Fabric
import torch
from torch import nn

from luolib.losses import SlicePerceptualLoss
from luolib.models import spadop

from .discriminator import PatchDiscriminatorBase
from .vq import VectorQuantizerOutput

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
        post_family: Literal['Laplace', 'normal'],
        rec_scale: bool,
        rec_weight: float,
        perceptual_loss: SlicePerceptualLoss,
        perceptual_weight: float,
        discriminator: PatchDiscriminatorBase,
        gan_weight: float = 1.0,
        adaptive_gan_weight: bool = False,
    ):
        """
        Args:
            post_family: family of post distribution, laplace → l1, normal → l2
            rec_scale: whether reconstruction has the scale parameter
            adaptive_gan_weight: do you like VQGAN?
        """
        super().__init__()
        self.quant_weight = quant_weight
        self.entropy_weight = entropy_weight
        self.rec_weight = rec_weight
        self.post_family = post_family
        self.rec_scale = rec_scale
        reduction = 'none' if rec_scale else 'mean'
        match post_family:
            case 'Laplace':
                self.rec_loss_fn = nn.L1Loss(reduction=reduction)
            case 'normal':
                self.rec_loss_fn = nn.MSELoss(reduction=reduction)
            case _:
                raise ValueError
        self.perceptual_loss = perceptual_loss
        print(f'{self.__class__}: running with {self.perceptual_loss.__class__}, function={self.perceptual_loss.perceptual_function.__class__}')
        self.perceptual_weight = perceptual_weight
        self.gan_weight = gan_weight
        self.adaptive_gan_weight = adaptive_gan_weight
        assert discriminator is not None
        self.discriminator = discriminator
        print(f'{self.__class__}: running with hinge W-GAN loss')

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

    def rec_loss(self, x_rec_logit: torch.Tensor, x_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_rec_logit_mu = x_rec_logit[:, :x_logit.shape[1]]
        if self.rec_scale:
            x_rec_logit_log_scale = x_rec_logit[:, x_logit.shape[1]:]
            # scale=1/b for Laplace, 1/var for normal
            x_rec_logit_scale = x_rec_logit_log_scale.exp()
            diff = self.rec_loss_fn(x_rec_logit_mu, x_logit)
            loss = -x_rec_logit_log_scale.mean() + (diff * x_rec_logit_scale).mean()
            diff = diff.mean()
        else:
            diff = loss = self.rec_loss_fn(x_rec_logit_mu, x_logit)
        return loss, diff

    def forward_gen(
        self,
        x: spadop.SpatialTensor,
        x_logit: spadop.SpatialTensor,
        x_rec: spadop.SpatialTensor,
        x_rec_logit: spadop.SpatialTensor,
        vq_out: VectorQuantizerOutput,
        use_gan_loss: bool,
        gan_weight_ref_param: nn.Parameter | None = None,
        fabric: Fabric | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x_rec_logit: this main be (mu, scale) of the reconstruction logit
        """
        rec_loss, diff = self.rec_loss(x_rec_logit, x_logit)
        perceptual_loss = self.perceptual_loss(x_rec, x)
        vq_loss = rec_loss + self.perceptual_weight * perceptual_loss + self.quant_weight * vq_out.loss
        if vq_out.entropy is not None:
            vq_loss = vq_loss + self.entropy_weight * vq_out.entropy
        # always calculate GAN loss even when not including in total loss, since it may be used elsewhere
        score_fake = self.discriminator(x_rec_logit[:, :x.shape[1]])
        gan_loss = -score_fake.mean()
        if use_gan_loss:
            gan_weight = self.gan_weight
            if self.adaptive_gan_weight:
                gan_weight *= calculate_adaptive_weight(
                    vq_loss.as_subclass(torch.Tensor),
                    gan_loss.as_subclass(torch.Tensor),
                    gan_weight_ref_param,
                    fabric=fabric,
                )
        else:
            gan_weight = 0
        loss = vq_loss + gan_weight * gan_loss
        log_dict = {
            'loss': loss,
            'rec_loss': rec_loss,
            'diff': diff,
            'perceptual_loss': perceptual_loss,
            'quant_loss': vq_out.loss,
            'util_var': vq_out.util_var,
            'vq_loss': vq_loss,
            'gan_loss': gan_loss,
            'gan_weight': gan_weight,
        }
        if vq_out.entropy is not None:
            log_dict['entropy'] = vq_out.entropy
        return loss, log_dict

    def forward_disc(
        self,
        x_logit: spadop.SpatialTensor,
        x_rec_logit: spadop.SpatialTensor,
        log_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        score_real = self.discriminator(x_logit.detach())
        score_fake = self.discriminator(x_rec_logit[:, :x_logit.shape[1]].detach())
        disc_loss = 0.5 * hinge_loss(score_real, score_fake)
        log_dict['disc_loss'] = disc_loss
        return disc_loss, log_dict
