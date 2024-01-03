from dataclasses import dataclass
from typing import Literal

from lightning import Fabric
import torch
from torch import nn
from torch.nn import functional as nnf

from luolib.losses import SlicePerceptualLoss
from luolib.models import spadop
from luolib.utils import ema_update

from .discriminator import PatchDiscriminatorBase, get_disc_scores
from .vq import VectorQuantizerOutput

# def hinge_loss(score_real: torch.Tensor, score_fake: torch.Tensor):
#     return (1 - score_real).relu().mean() + (1 + score_fake).relu().mean()

__all__ = [
    'hinge_gan_loss',
    'VQVTLoss',
]

@dataclass
class ParameterWrapper:
    """use this class to prevent parameter being registered under nn.Module"""
    param: nn.Parameter | None

def hinge_gan_loss(score_real: torch.Tensor, score_fake: torch.Tensor):
    real_loss = (1 - score_real).relu().mean()
    fake_loss = (1 + score_fake).relu().mean()
    return real_loss, fake_loss


class VQVTLoss(nn.Module):
    def __init__(
        self,
        quant_weight: float,
        entropy_weight: float,
        rec_loss: Literal['l1', 'l2', 'smooth_l1'],
        rec_loss_beta: float,
        rec_scale: bool,
        rec_weight: float,
        perceptual_loss: SlicePerceptualLoss,
        perceptual_weight: float,
        discriminator: PatchDiscriminatorBase,
        gan_weight: float,
        adaptive_gan_weight: bool,
        grad_ema_decay: float,
        gan_ema_decay: float,
        gan_start_th: float,
        gan_stop_th: float,
        max_log_scale: float = 8,
    ):
        """
        Args:
            rec_scale: whether reconstruction has the scale parameter
            adaptive_gan_weight: use adaptive weight according to VQGAN
            max_log_scale: log_scale lather than this value will be clamped. empirically, this value should be smaller
            with Laplace than normal
        """
        super().__init__()
        self.quant_weight = quant_weight
        self.entropy_weight = entropy_weight
        self.rec_weight = rec_weight
        self.rec_scale = rec_scale
        reduction = 'none' if rec_scale else 'mean'
        match rec_loss:
            case 'l1':
                self.rec_loss_fn = nn.L1Loss(reduction=reduction)
            case 'l2':
                self.rec_loss_fn = nn.MSELoss(reduction=reduction)
            case 'smooth_l1':
                self.rec_loss_fn = nn.SmoothL1Loss(reduction=reduction, beta=rec_loss_beta)
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
        self.gan_ema_decay = gan_ema_decay
        self.gan_start_th = gan_start_th
        self.gan_stop_th = gan_stop_th
        self.register_buffer('real_loss_ema', torch.tensor(1.))
        self.register_buffer('fake_loss_ema', torch.tensor(1.))
        self.eval()
        self.requires_grad_(False)

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

    # FIXME: current implementation of EMA is incompatible with gradient accumulation
    grad_vq_ema: torch.Tensor
    grad_gan_ema: torch.Tensor

    def set_gan_ref_param(self, param: nn.Parameter):
        # it's not a good idea trying to use property setter in nn.Module: https://github.com/pytorch/pytorch/issues/52664
        self.gan_ref_param = ParameterWrapper(param)
        self.register_buffer('grad_vq_ema', torch.zeros_like(param))
        self.register_buffer('grad_gan_ema', torch.zeros_like(param))

    @torch.no_grad()
    def adapt_gan_weight(self, vq_loss: torch.Tensor, gan_loss: torch.Tensor, fabric: Fabric):
        grad_vq, = torch.autograd.grad(vq_loss, self.gan_ref_param.param, retain_graph=True)
        grad_gan, = torch.autograd.grad(gan_loss, self.gan_ref_param.param, retain_graph=True)
        grad_vq, grad_gan = fabric.all_reduce(torch.stack([grad_vq, grad_gan]).contiguous())
        ema_update(self.grad_vq_ema, grad_vq, self.grad_ema_decay)
        ema_update(self.grad_gan_ema, grad_gan, self.grad_ema_decay)
        scale = self.grad_vq_ema.norm() / (self.grad_gan_ema.norm() + 1e-6)
        scale.clamp_(max=1e4)
        return scale

    real_loss_ema: torch.Tensor
    fake_loss_ema: torch.Tensor

    def forward_gen(
        self,
        x: spadop.SpatialTensor,
        x_logit: spadop.SpatialTensor,
        x_rec: spadop.SpatialTensor,
        x_rec_logit: spadop.SpatialTensor,
        vq_out: VectorQuantizerOutput,
        use_gan_loss: bool,
        fabric: Fabric | None = None,
    ) -> tuple[torch.Tensor, dict, bool]:
        """
        Args:
            x_rec_logit: this may be (mu, scale) of the reconstruction logit
        """
        # vq loss part
        rec_loss, log_dict = self.rec_loss(x_rec_logit, x_logit)
        perceptual_loss = self.perceptual_loss(x_rec, x)
        vq_loss = rec_loss + self.perceptual_weight * perceptual_loss + self.quant_weight * vq_out.loss
        if vq_out.entropy is not None:
            vq_loss = vq_loss + self.entropy_weight * vq_out.entropy
        # gan loss part, very abstract
        score_real, score_fake = get_disc_scores(self.discriminator, x_logit, x_rec_logit[:, :x.shape[1]])
        with torch.no_grad():
            real_loss, fake_loss = hinge_gan_loss(score_real, score_fake)
            real_loss, fake_loss = fabric.all_reduce(torch.stack([real_loss, fake_loss]))
        ema_update(self.real_loss_ema, real_loss, self.gan_ema_decay)
        ema_update(self.fake_loss_ema, fake_loss, self.gan_ema_decay)

        if self.real_loss_ema <= self.gan_start_th and self.fake_loss_ema <= self.gan_start_th:
            use_gan_loss = True
        if self.real_loss_ema > self.gan_stop_th or self.fake_loss_ema > self.gan_stop_th:
            use_gan_loss = False

        gan_loss = -score_fake.mean()
        if self.adaptive_gan_weight:
            gan_loss_scale = self.adapt_gan_weight(vq_loss, gan_loss, fabric)
        else:
            gan_loss_scale = 1
        if use_gan_loss:
            gan_weight = self.gan_weight * gan_loss_scale
        else:
            gan_weight = 0
        # combine both losses and log
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
            'disc/teacher-real': real_loss,
            'disc/teacher-real_ema': self.real_loss_ema,
            'disc/teacher-fake': fake_loss,
            'disc/teacher-fake_ema': self.fake_loss_ema,
        })
        with torch.no_grad():
            log_dict['l1'] = nnf.l1_loss(x_rec, x)
            log_dict['l2'] = nnf.mse_loss(x_rec, x)
        if vq_out.entropy is not None:
            log_dict['entropy'] = vq_out.entropy
        return loss, log_dict, use_gan_loss
