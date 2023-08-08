from collections.abc import Sequence
from functools import cached_property
from pathlib import Path

import einops
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from timm.scheduler.scheduler import Scheduler as TIMMScheduler
import torch
from torch import nn
from torch.nn import functional as nnf
from torch.utils import checkpoint
from torchvision.utils import save_image

from luolib.models import load_ckpt
from luolib.types import LRSchedulerConfig, NoWeightDecayParameter
from luolib.utils.grad import grad_norm
from monai.config import PathLike

from pumt import sac
from pumt.optim import build_lr_scheduler, build_optimizer
from pumt.tokenizer import VQTokenizer
from .vit import ViT

class ViTForMIM(ViT, LightningModule):
    # input_norm_mean: torch.Tensor
    # input_norm_std: torch.Tensor

    def __init__(
        self,
        *args,
        tokenizer: VQTokenizer,
        tokenizer_eval: bool = True,
        mask_ratio: float,
        mask_layer_ids: Sequence[int],
        optimizer: dict | None = None,
        lr_scheduler: LRSchedulerConfig | None = None,
        plot_image_every_n_steps: int = 400,
        eva02_pretrained_path: Path | None = None,
        **kwargs,
    ):
        """mask_layer_ids: layers that include mask tokens as input"""
        # note: intercept the `eva02_pretrained_path` parameter or mask_token will not be loaded
        super().__init__(*args, **kwargs)
        assert tokenizer.stride == self.patch_embed.patch_size
        self.tokenizer = tokenizer
        assert tokenizer.quantize.mode in ['gumbel', 'soft']
        tokenizer.requires_grad_(False)
        self.tokenizer_eval = tokenizer_eval
        if tokenizer_eval:
            tokenizer.eval()
        self.mask_token = NoWeightDecayParameter(torch.empty(1, 1, self.embed_dim))
        self.mask_ratio = mask_ratio
        self.mask_layer_ids = set(mask_layer_ids)
        self.mim_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.Linear(self.embed_dim, tokenizer.codebook_size),
        )

        self.mim_loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        assert self.patch_embed.adaptive
        self.register_buffer(
            'input_norm_mean',
            einops.rearrange(torch.tensor(IMAGENET_DEFAULT_MEAN), 'c -> c 1 1 1'),
            persistent=False,
        )
        self.register_buffer(
            'input_norm_std',
            einops.rearrange(torch.tensor(IMAGENET_DEFAULT_STD), 'c -> c 1 1 1'),
            persistent=False,
        )
        self.plot_image_every_n_steps = plot_image_every_n_steps
        if eva02_pretrained_path is not None:
            load_ckpt(self, eva02_pretrained_path, 'module')

    def train(self, mode: bool = True):
        super().train(mode)
        self.tokenizer.train(not self.tokenizer_eval)

    def input_norm(self, x: sac.SpatialTensor):
        return nnf.instance_norm(x)

    def state_dict(self, *args, **kwargs):
        return {
            k: v for k, v in super().state_dict(*args, **kwargs).items()
            if not k.startswith('tokenizer.')
        }

    @cached_property
    def run_dir(self) -> Path:
        logger: WandbLogger = self.logger
        if self.trainer.is_global_zero:
            run_dir = Path(logger.save_dir) / Path(logger.experiment.dir).parent.name
        else:
            run_dir = None
        run_dir = self.trainer.strategy.broadcast(run_dir)
        return run_dir

    def forward(self, x: sac.SpatialTensor, visible_idx: torch.Tensor | None = None):
        if visible_idx is None:
            return super().forward(x)
        x, spatial_shape = self.prepare_seq_input(x)
        x = x.as_tensor()
        self.rope.prepare(spatial_shape, visible_idx)
        batch_size, seq_len, dim = x.shape
        seq_len -= 1  # exclude cls token
        # cls token should always be visible
        visible_idx = torch.cat([visible_idx.new_zeros(batch_size, 1), visible_idx + 1], dim=1)
        visible_idx = einops.repeat(visible_idx, 'n l -> n l d', d=dim).contiguous()
        x = torch.cat([
                self.cls_token.expand(batch_size, 1, -1),
                self.mask_token.expand(batch_size, seq_len, -1),
            ],
            dim=1,
        ).scatter_(dim=1, index=visible_idx, src=x.gather(dim=1, index=visible_idx))
        for i, block in enumerate(self.blocks):
            x_layer = x if i in self.mask_layer_ids else x.gather(dim=1, index=visible_idx)
            if self.training and self.grad_ckpt:
                x_layer = checkpoint.checkpoint(block, x_layer)
            else:
                x_layer = block(x_layer)
            if i in self.mask_layer_ids:
                x = x_layer
            else:
                x = x.scatter(dim=1, index=visible_idx, src=x_layer)
        self.rope.reset()
        return self.norm(x)

    def configure_optimizers(self):
        return {
            'optimizer': (optimizer := build_optimizer(self, self.optimizer)),
            'lr_scheduler': vars(build_lr_scheduler(optimizer, self.lr_scheduler, self.trainer.max_steps)),
        }

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero:
            (self.run_dir / 'model.txt').write_text(repr(self))

    def lr_scheduler_step(self, scheduler: TIMMScheduler, metric=None):
        scheduler.step_update(self.global_step + 1, metric)

    def on_train_start(self) -> None:
        scheduler: TIMMScheduler = self.lr_schedulers()
        # https://github.com/Lightning-AI/lightning/issues/17972
        scheduler.step_update(0)

    def training_step(self, batch: tuple[torch.Tensor, int, list[PathLike]], batch_idx: int, *args, **kwargs):
        x = sac.SpatialTensor(*batch[:2])
        tokenizer_input = 2 * x - 1
        quant_out = self.tokenizer.tokenize(tokenizer_input)
        token_logits = einops.rearrange(quant_out.logits.as_tensor(), 'n ... ne -> n (...) ne')
        batch_size, seq_len = token_logits.shape[:2]
        num_visible_patches = int(seq_len * (1 - self.mask_ratio))
        visible_idx, _ = token_logits.new_ones(batch_size, seq_len).multinomial(num_visible_patches).sort()
        hidden_states = self(self.input_norm(x), visible_idx)[:, 1:]
        masked_mask = hidden_states.new_ones(batch_size, seq_len, dtype=torch.bool).scatter(dim=1, index=visible_idx, value=False)
        masked_token_logits = self.mim_head(hidden_states[masked_mask])
        loss = self.mim_loss(
            masked_token_logits,
            token_logits[masked_mask].softmax(dim=-1),
        )
        self.log('train/loss', loss)
        with torch.no_grad():
            kld = nnf.kl_div(
                masked_token_logits.log_softmax(dim=-1),
                token_logits[masked_mask].log_softmax(dim=-1),
                reduction='batchmean',
                log_target=True,
            )
            self.log('train/kld', kld)
        if self.trainer.is_global_zero and (optimized_steps := self.global_step + 1) % self.plot_image_every_n_steps == 0:
            plot_dir = self.run_dir / 'plot' / f'step-{optimized_steps}'
            plot_dir.mkdir(parents=True)
            mim_token_logits = token_logits[0:1].detach().clone()
            mim_token_logits[masked_mask[0:1]] = masked_token_logits[0:1].to(token_logits)
            d, h, w = quant_out.z_q.shape[2:]
            mim_z_q = einops.rearrange(
                einops.einsum(
                    mim_token_logits.softmax(dim=-1), self.tokenizer.quantize.embedding.weight,
                    '... ne, ne c -> ... c',
                ),
                'n (d h w) c -> n c d h w', d=d, h=h, w=w,
            )
            mim_z_q = sac.SpatialTensor(mim_z_q, quant_out.z_q.aniso_d, quant_out.z_q.num_downsamples)
            mim_x_rec = (self.tokenizer.decode(mim_z_q) + 1) / 2
            x_rec = (self.tokenizer.decode(quant_out.z_q[0:1]) + 1) / 2
            (plot_dir / 'path.txt').write_text(str(batch[-1][0]))
            for i in range(x.shape[2]):
                save_image(x[0, :, i], plot_dir / f'{i}-origin.png')
                save_image(x_rec[0, :, i], plot_dir / f'{i}-rec.png')
                save_image(mim_x_rec[0, :, i], plot_dir / f'{i}-mim-rec.png')
        return loss

    def on_before_optimizer_step(self, *args, **kwargs):
        self.log('grad-norm/patch_embed', grad_norm(self.patch_embed))
        for i, block in enumerate(self.blocks):
            self.log(f'grad-norm/block-{i}', grad_norm(block))
        self.log('grad-norm/mim_head', grad_norm(self.mim_head))
        self.log('grad-norm/total', grad_norm(self))
