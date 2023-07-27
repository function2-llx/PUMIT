from pathlib import Path

from lightning import Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback as LightningSaveConfigCallback
from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
import torch

class SaveConfigCallback(LightningSaveConfigCallback):
    @rank_zero_only
    def setup(self, trainer: Trainer, *args, **kwargs):
        assert not self.already_saved
        logger: WandbLogger = trainer.logger
        save_dir = Path(logger.save_dir) / Path(logger.experiment.dir).parent.name / 'conf'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / 'conf.yaml'
        # https://github.com/omni-us/jsonargparse/issues/332
        self.parser.save(
            self.config, save_path, skip_none=False, skip_check=True, overwrite=self.overwrite, multifile=self.multifile
        )

class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('seed_everything', 'data.init_args.seed')
        parser.link_arguments('trainer.max_steps', 'data.init_args.dl_conf.num_train_batches')

    def before_instantiate_classes(self):
        if self.subcommand == 'fit':
            save_dir = self.config.fit.trainer.logger.init_args.save_dir
            Path(save_dir).mkdir(exist_ok=True, parents=True)

def main():
    torch.autograd.set_detect_anomaly(True)
    torch.set_float32_matmul_precision('high')
    CLI(save_config_callback=SaveConfigCallback)

if __name__ == "__main__":
    main()
