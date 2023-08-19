from collections.abc import Mapping
from functools import cached_property
from pathlib import Path

from lightning import LightningModule as LightningModuleBase, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint as ModelCheckpointBase
from lightning.pytorch.cli import (
    LightningCLI as LightningCLIBase,
    SaveConfigCallback as SaveConfigCallbackBase,
)
from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from timm.scheduler.scheduler import Scheduler as TIMMScheduler
from torch.optim import Optimizer

from mylib.types import LRSchedulerConfig
from pumt.optim import build_lr_scheduler, build_optimizer

class LightningModule(LightningModuleBase):
    optimizer: Mapping
    lr_scheduler_config: LRSchedulerConfig

    @cached_property
    def run_dir(self) -> Path:
        logger: WandbLogger = self.logger
        if self.trainer.is_global_zero:
            run_dir = Path(logger.save_dir) / logger.experiment.name / f'{Path(logger.experiment.dir).parent.name}'
        else:
            run_dir = None
        run_dir = self.trainer.strategy.broadcast(run_dir)
        return run_dir

    def configure_optimizers(self):
        return {
            'optimizer': (optimizer := build_optimizer(self, self.optimizer)),
            'lr_scheduler': vars(build_lr_scheduler(optimizer, self.lr_scheduler_config, self.trainer.max_steps)),
        }

    def on_fit_start(self) -> None:
        if self.trainer.is_global_zero:
            (self.run_dir / 'model.txt').write_text(repr(self))

    def on_train_start(self) -> None:
        scheduler: TIMMScheduler = self.lr_schedulers()
        # https://github.com/Lightning-AI/lightning/issues/17972
        scheduler.step_update(0)

    def lr_scheduler_step(self, scheduler: TIMMScheduler, metric=None):
        scheduler.step_update(self.global_step + 1, metric)

class SaveConfigCallback(SaveConfigCallbackBase):
    @rank_zero_only
    def setup(self, trainer: Trainer, model: LightningModule, *args, **kwargs):
        assert not self.already_saved
        save_dir = model.run_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / 'conf.yaml'
        self.parser.save(
            self.config, save_path,
            skip_none=False, overwrite=self.overwrite, multifile=self.multifile,
        )

class ModelCheckpoint(ModelCheckpointBase):
    def __resolve_ckpt_dir(self, trainer):
        model: LightningModule = trainer.model
        return model.run_dir / 'checkpoint'

class LightningCLI(LightningCLIBase):
    model: LightningModule

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            model_class=LightningModule,
            save_config_callback=SaveConfigCallback,
            subclass_mode_model=True,
            auto_configure_optimizers=False,
            **kwargs,
        )

    def add_arguments_to_parser(self, parser) -> None:
        parser.add_subclass_arguments(Optimizer, 'optimizer', instantiate=False, skip={'params'})
        parser.add_subclass_arguments(TIMMScheduler, 'lr_scheduler', instantiate=False, skip={'optimizer'})
        parser.link_arguments('trainer.max_steps', 'lr_scheduler.init_args.t_initial')
        parser.add_dataclass_arguments(LRSchedulerConfig, 'lr_scheduler_config')

    def before_instantiate_classes(self):
        # https://github.com/wandb/wandb/issues/714#issuecomment-565870686
        save_dir = self.config[self.subcommand].trainer.logger.init_args.save_dir
        Path(save_dir).mkdir(exist_ok=True, parents=True)

    def instantiate_classes(self) -> None:
        super().instantiate_classes()
        if self.subcommand == 'fit':
            self.model.optimizer = self.config.fit.optimizer
            self.model.lr_scheduler_config = self.config.fit.lr_scheduler_config
            self.model.lr_scheduler_config.scheduler = self.config.fit.lr_scheduler
