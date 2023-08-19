from pathlib import Path

import cytoolz
from lightning import LightningModule
from timm.scheduler.scheduler import Scheduler
import torch
from torch.optim import Optimizer

from monai.utils import ensure_tuple

from mylib.conf import ExpConfBase
from mylib.types import ParamGroup
from mylib.utils import partition_by_predicate
from mylib.optim import create_optimizer, param_groups_layer_decay
from mylib.scheduler import create_scheduler
from ..utils import split_weight_decay_keys

class ExpModelBase(LightningModule):
    def __init__(self, conf: ExpConfBase):
        super().__init__()
        self.conf = conf
        self.backbone = self.create_backbone()

    def create_backbone(self):
        return create_model(self.conf.backbone, backbone_registry)

    @torch.no_grad()
    def backbone_dummy(self):
        conf = self.conf
        self.backbone.eval()
        dummy_input = torch.zeros(1, conf.num_input_channels, *conf.sample_shape)
        dummy_output = self.backbone.forward(dummy_input)
        if conf.print_shape:
            print('backbone output shapes:')
            for x in dummy_output.feature_maps:
                print(x.shape[1:])
        return dummy_input, dummy_output

    @property
    def tta_flips(self):
        match self.conf.spatial_dims:
            case 2:
                return [[2], [3], [2, 3]]
            case 3:
                return [[2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]
            case _:
                raise ValueError

    @property
    def log_exp_dir(self) -> Path:
        assert self.trainer.is_global_zero
        from pytorch_lightning.loggers import WandbLogger
        logger: WandbLogger = self.trainer.logger   # type: ignore
        return Path(logger.experiment.dir)

    def on_fit_start(self):
        if not self.trainer.is_global_zero:
            return
        with open(self.log_exp_dir / 'fit-summary.txt', 'w') as f:
            print(self, file=f, end='\n\n\n')
            print('optimizers:\n', file=f)
            for optimizer in ensure_tuple(self.optimizers()):
                print(optimizer, file=f)
            print('\n\n', file=f)
            print('schedulers:\n', file=f)
            for scheduler in ensure_tuple(self.lr_schedulers()):
                print(scheduler, file=f)

    def get_param_groups(self) -> list[ParamGroup]:
        others_no_decay_keys, backbone_no_decay_keys = map(
            set,
            partition_by_predicate(lambda k: k.startswith('backbone.'), split_weight_decay_keys(self)),
        )
        backbone_optim = self.conf.backbone_optim
        optim = self.conf.optimizer
        param_groups = []
        if backbone_optim.lr == 0:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('fix backbone weights')
        backbone_param_groups: list[ParamGroup] = param_groups_layer_decay(
            self.backbone,
            backbone_optim.weight_decay,
            backbone_no_decay_keys,
            backbone_optim.layer_decay,
        )
        for param_group in backbone_param_groups:
            param_group['lr'] = backbone_optim.lr * param_group['lr_scale']
        param_groups.extend(backbone_param_groups)

        others_decay_param_group, others_no_decay_param_group = map(
            lambda named_parameters: {
                'param_names': [n for n, _ in named_parameters],
                'param': [p for p, _ in named_parameters],
            },
            partition_by_predicate(
                lambda np: np[0] in others_no_decay_keys,
                filter(lambda np: not np[0].startswith('backbone.'), self.named_parameters()),
            )
        )

        param_groups.extend([
            {
                **others_decay_param_group,
                'weight_decay': optim.weight_decay,
            },
            {
                **others_no_decay_param_group,
                'weight_decay': 0.,
            }
        ])

        return param_groups

    def configure_optimizers(self):
        conf = self.conf
        param_groups = self.get_param_groups()
        named_parameters = dict(self.named_parameters())
        for param_group in param_groups:
            if 'params' not in param_group:
                param_group['params'] = list(cytoolz.get(param_group['param_names'], named_parameters))

        optimizer = create_optimizer(conf.optimizer, param_groups)
        scheduler = create_scheduler(conf.scheduler, optimizer)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': conf.scheduler.interval,
                'frequency': conf.scheduler.frequency,
                'reduce_on_plateau': conf.scheduler.reduce_on_plateau,
                'monitor': conf.monitor,
            },
        }

    def lr_scheduler_step(self, scheduler: Scheduler, metric):
        # make compatible with timm scheduler
        conf = self.conf
        match conf.scheduler.interval:
            case 'epoch':
                scheduler.step(self.current_epoch + 1, metric)
            case 'step':
                from timm.scheduler import PlateauLRScheduler
                if isinstance(scheduler, PlateauLRScheduler):
                    scheduler.step(self.global_step // conf.scheduler.frequency, metric)
                else:
                    scheduler.step_update(self.global_step, metric)

    def optimizer_zero_grad(self, _epoch, _batch_idx, optimizer: Optimizer):
        optimizer.zero_grad(set_to_none=self.conf.optimizer_set_to_none)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        grad_norm = torch.linalg.vector_norm(
            torch.stack([
                torch.linalg.vector_norm(g.detach())
                for p in self.parameters() if (g := p.grad) is not None
            ])
        )
        self.log('train/grad_norm', grad_norm)

    @property
    def interpolate_mode(self):
        match self.conf.spatial_dims:
            case 2:
                return 'bilinear'
            case 3:
                return 'trilinear'
            case _:
                raise ValueError
