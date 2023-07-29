from collections.abc import Sequence
from pathlib import Path

import einops
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from timm.scheduler.scheduler import Scheduler as TIMMScheduler
import torch
from torch import nn
from torch.utils import checkpoint

from luolib.models import load_ckpt
from luolib.types import LRSchedulerConfig, NoWeightDecayParameter

from pumt import sac
from pumt.optim import build_lr_scheduler, build_optimizer
from pumt.tokenizer import VQTokenizer
from .vit import ViT

class ViTForMIM(ViT, LightningModule):
    # tokenizer typed as dict https://github.com/omni-us/jsonargparse/issues/330
    def __init__(
        self,
        *args,
        tokenizer: VQTokenizer,
        mask_ratio: float,
        mask_layer_ids: Sequence[int],
        optimizer: dict | None = None,
        lr_scheduler: LRSchedulerConfig | None = None,
        eva02_pretrained_path: Path | None = None,
        **kwargs,
    ):
        """mask_layer_ids: layers that include mask tokens as input"""
        super().__init__(*args, **kwargs)
        assert tokenizer.stride == self.patch_embed.patch_size
        self.tokenizer = tokenizer
        tokenizer.requires_grad_(False)
        self.mask_token = NoWeightDecayParameter(torch.empty(1, 1, self.embed_dim))
        self.mask_ratio = mask_ratio
        self.mask_layer_ids = set(mask_layer_ids)
        self.mim_head = nn.Linear(self.embed_dim, tokenizer.codebook_size)
        self.mim_loss = nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        if eva02_pretrained_path is not None:
            load_ckpt(self, eva02_pretrained_path, 'module')

    def state_dict(self, *args, **kwargs):
        return {
            k: v for k, v in super().state_dict(*args, **kwargs)
            if k.startswith('tokenizer.')
        }

    @property
    def run_dir(self):
        logger: WandbLogger = self.logger
        return Path(logger.save_dir) / Path(logger.experiment.dir).parent.name

    def forward(self, x: sac.SpatialTensor, visible_idx: torch.Tensor | None = None):
        if visible_idx is None:
            return super().forward(x)
        x, spatial_shape = self.prepare_seq_input(x)
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
            if self.grad_ckpt:
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

    def lr_scheduler_step(self, scheduler: TIMMScheduler, metric=None):
        scheduler.step_update(self.global_step + 1, metric)

    def on_train_start(self) -> None:
        scheduler: TIMMScheduler = self.lr_schedulers()
        # https://github.com/Lightning-AI/lightning/issues/17972
        scheduler.step_update(0)

    def training_step(self, batch: tuple[torch.Tensor, int, Path], *args, **kwargs):
        x = sac.SpatialTensor(*batch[:2])
        token_ids = self.tokenizer.tokenize(x)
        batch_size, seq_len = token_ids.shape[:2]
        num_visible_patches = int(seq_len * (1 - self.mask_ratio))
        visible_idx, _ = token_ids.new_ones(batch_size, seq_len).multinomial(num_visible_patches).sort()
        hidden_states = self(x, visible_idx)[:, 1:]
        masked_mask = hidden_states.new_ones(batch_size, seq_len, dtype=torch.bool).scatter(dim=1, index=visible_idx, value=False)
        token_probs = self.mim_head(hidden_states[masked_mask])
        loss = self.mim_loss(token_probs, token_ids[masked_mask])
        self.log('train/loss', loss)
        return loss
