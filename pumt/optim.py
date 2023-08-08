from collections.abc import Mapping

from lightning.pytorch.cli import instantiate_class
from torch import nn
from torch.optim import Optimizer

from luolib.models.utils import create_param_groups, split_weight_decay_keys
from luolib.types import LRSchedulerConfig

def build_optimizer(model: nn.Module, optimizer_conf: Mapping):
    decay_keys, no_decay_keys = split_weight_decay_keys(model)
    return instantiate_class(
        create_param_groups(model.named_parameters(), decay_keys, no_decay_keys),
        optimizer_conf,
    )

def build_lr_scheduler(optimizer: Optimizer, config: LRSchedulerConfig, max_steps: int, ensure_timm: bool = True) -> LRSchedulerConfig:
    config.scheduler['init_args']['t_initial'] = max_steps
    config.scheduler = instantiate_class(optimizer, config.scheduler)
    if ensure_timm:
        from timm.scheduler.scheduler import Scheduler as TIMMScheduler
        assert isinstance(config.scheduler, TIMMScheduler)
    return config
