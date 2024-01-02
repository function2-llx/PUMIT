from dataclasses import dataclass
from typing import Literal

from lightning import Fabric
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.losses import SlicePerceptualLoss
from luolib.models import spadop

from .discriminator import PatchDiscriminatorBase
from .vq import VectorQuantizerOutput

def hinge_loss(score_real: torch.Tensor, score_fake: torch.Tensor):
    return (1 - score_real).relu().mean() + (1 + score_fake).relu().mean()

@dataclass
class ParameterWrapper:
    """use this class to prevent parameter being registered under nn.Module"""
    param: nn.Parameter | None

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
        gan_weight: float,
        adaptive_gan_weight: bool,
        grad_ema_decay: float,
        max_log_scale: float = 8,
    ):
        """
        Args:
            post_family: family of post distribution, laplace → l1, normal → l2
            rec_scale: whether reconstruction has the scale parameter
            adaptive_gan_weight: use adaptive weight according to VQGAN
            max_log_scale: log_scale lather than this value will be clamped. empirically, this value should be smaller
            with Laplace than normal
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
        self.max_log_scale = max_log_scale
        self.grad_ema_decay = grad_ema_decay

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

    def rec_loss(self, x_rec_logit: torch.Tensor, x_logit: torch.Tensor) -> tuple[torch.Tensor, dict]:
        mu = x_rec_logit[:, :x_logit.shape[1]]
        log_dict = {}
        if self.rec_scale:
            log_scale = x_rec_logit[:, x_logit.shape[1]:]
            with torch.no_grad():
                clamped_log_scale = log_scale.clamp(max=self.max_log_scale)
            log_scale = clamped_log_scale - log_scale.detach() + log_scale
            # scale=1/b for Laplace, 1/var for normal
            scale = log_scale.exp()
            diff = self.rec_loss_fn(mu, x_logit)
            log_scale = log_scale.mean()
            loss = -log_scale + (diff * scale).mean()
            log_dict['log-scale'] = log_scale
            with torch.no_grad():
                log_dict['scale'] = scale.mean()
        else:
            loss = self.rec_loss_fn(mu, x_logit)
        log_dict['rec_loss'] = loss
        return loss, log_dict

    def set_gan_ref_param(self, param: nn.Parameter):
        # it's not a good idea trying to use property setter in nn.Module: https://github.com/pytorch/pytorch/issues/52664
        self.gan_ref_param = ParameterWrapper(param)
        self.discriminator.register_buffer('grad_vq_ema', torch.zeros_like(param))
        self.discriminator.register_buffer('grad_gan_ema', torch.zeros_like(param))

    @torch.no_grad()
    def adapt_gan_weight(self, vq_loss: torch.Tensor, gan_loss: torch.Tensor, fabric: Fabric):
        grad_vq, = torch.autograd.grad(vq_loss, self.gan_ref_param.param, retain_graph=True)
        grad_gan, = torch.autograd.grad(gan_loss, self.gan_ref_param.param, retain_graph=True)
        grad_vq, grad_gan = fabric.all_reduce(torch.stack([grad_vq, grad_gan]).contiguous())
        self.discriminator.grad_vq_ema.mul_(self.grad_ema_decay).add_(grad_vq, alpha=1 - self.grad_ema_decay)
        self.discriminator.grad_gan_ema.mul_(self.grad_ema_decay).add_(grad_gan, alpha=1 - self.grad_ema_decay)
        scale = self.discriminator.grad_vq_ema.norm() / (self.discriminator.grad_gan_ema.norm() + 1e-6)
        scale.clamp_(max=1e4)
        return scale

    def forward_gen(
        self,
        x: spadop.SpatialTensor,
        x_logit: spadop.SpatialTensor,
        x_rec: spadop.SpatialTensor,
        x_rec_logit: spadop.SpatialTensor,
        vq_out: VectorQuantizerOutput,
        use_gan_loss: bool,
        fabric: Fabric | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            x_rec_logit: this may be (mu, scale) of the reconstruction logit
        """
        rec_loss, log_dict = self.rec_loss(x_rec_logit, x_logit)
        perceptual_loss = self.perceptual_loss(x_rec, x)
        vq_loss = rec_loss + self.perceptual_weight * perceptual_loss + self.quant_weight * vq_out.loss
        if vq_out.entropy is not None:
            vq_loss = vq_loss + self.entropy_weight * vq_out.entropy
        with torch.set_grad_enabled(self.adaptive_gan_weight or use_gan_loss):
            score_fake = self.discriminator(x_rec_logit[:, :x.shape[1]])
        gan_loss = -score_fake.mean()
        if self.adaptive_gan_weight:
            gan_loss_scale = self.adapt_gan_weight(vq_loss, gan_loss, fabric)
        else:
            gan_loss_scale = 1
        if use_gan_loss:
            gan_weight = self.gan_weight * gan_loss_scale
        else:
            gan_weight = 0
        loss = vq_loss + gan_weight * gan_loss
        log_dict.update({
            'loss': loss,
            'perceptual_loss': perceptual_loss,
            'quant_loss': vq_out.loss,
            'util_var': vq_out.util_var,
            'vq_loss': vq_loss,
            'gan_loss': gan_loss,
            'gan_weight': gan_weight,
            'gan_loss_scale': gan_loss_scale,
        })
        with torch.no_grad():
            log_dict['l1'] = nnf.l1_loss(x_rec, x)
            log_dict['l2'] = nnf.mse_loss(x_rec, x)
        if vq_out.entropy is not None:
            log_dict['entropy'] = vq_out.entropy
        return loss, log_dict

    def forward_disc(
        self,
        x_logit: spadop.SpatialTensor,
        x_rec_logit: spadop.SpatialTensor,
        log_dict: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict]:
        # discriminator is usually a small network, concat and make it a little faster
        batch_size = x_logit.shape[0]
        logits = torch.cat([x_logit, x_rec_logit[:, :x_logit.shape[1]].detach()], 0)
        scores = self.discriminator(logits)
        score_real, score_fake = scores[:batch_size], scores[batch_size:]
        real_loss = (1 - score_real).relu().mean()
        fake_loss = (1 + score_fake).relu().mean()
        disc_loss = 0.5 * (real_loss + fake_loss)
        log_dict.update({
            'disc_loss': disc_loss,
            'disc_loss_real': real_loss,
            'disc_loss_fake': fake_loss,
        })
        return disc_loss, log_dict
