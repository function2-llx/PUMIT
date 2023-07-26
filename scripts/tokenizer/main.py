from dataclasses import dataclass
from pathlib import Path
from typing import cast

from jsonargparse import ActionConfigFile, ArgumentParser
from lightning.fabric import Fabric as LightningFabric
from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.loggers import WandbLogger
import math
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from luolib.models.utils import create_param_groups, load_ckpt, split_weight_decay_keys
from luolib.types import LRSchedulerConfig
from pumt.conv import SpatialTensor
from pumt.tokenizer import TokenizerDataModule, VQVAEModel
from pumt.tokenizer.loss import VQGANLoss

class Fabric(LightningFabric):
    # https://github.com/Lightning-AI/lightning/issues/18106
    def _setup_dataloader(
        self, dataloader: DataLoader, use_distributed_sampler: bool = True, move_to_device: bool = True
    ) -> DataLoader:
        assert use_distributed_sampler is False
        # add worker_init_fn for correct seeding in worker processes
        from lightning.fabric.utilities.data import _auto_add_worker_init_fn
        _auto_add_worker_init_fn(dataloader, self.global_rank)

        dataloader = self._strategy.process_dataloader(dataloader)
        from lightning.fabric.wrappers import _FabricDataLoader
        lite_dataloader = _FabricDataLoader(dataloader=dataloader, device=self.device)
        lite_dataloader = cast(DataLoader, lite_dataloader)
        return lite_dataloader

@dataclass(kw_only=True)
class TrainingArguments:
    max_steps: int
    seed: int = 42
    log_every_n_steps: int = 10
    save_last_every_n_steps: int = 1000
    plot_image_every_n_steps: int = 100
    save_every_n_steps: int = 10000
    max_norm_g: int | float | None = None
    max_norm_d: int | float | None = None
    resume_ckpt_path: Path | None = None
    pretrained_ckpt_path: Path | None = None
    output_dir: Path = Path('output/tokenizer')
    disc_loss_ema_init: float = 1.
    disc_loss_momentum: float = 0.85
    use_gan_th: float = 0.15

def build_optimizer(model: nn.Module, optimizer_conf: dict):
    decay_keys, no_decay_keys = split_weight_decay_keys(model)
    return instantiate_class(
        create_param_groups(model.named_parameters(), decay_keys, no_decay_keys),
        optimizer_conf,
    )

def build_lr_scheduler(optimizer: Optimizer, config: LRSchedulerConfig, max_steps: int) -> LRSchedulerConfig:
    config.scheduler['init_args']['t_initial'] = max_steps
    config.scheduler = instantiate_class(optimizer, config.scheduler)
    from timm.scheduler.scheduler import Scheduler as TIMMScheduler
    assert isinstance(config.scheduler, TIMMScheduler)
    return config

def get_parser():
    parser = ArgumentParser(parser_mode='omegaconf')
    parser.add_argument('-c', '--config', action=ActionConfigFile)
    parser.add_subclass_arguments(VQVAEModel, 'vqvae')
    parser.add_argument('--optimizer_g', type=dict)
    parser.add_argument('--lr_scheduler_g', type=LRSchedulerConfig)
    parser.add_class_arguments(VQGANLoss, 'loss')
    parser.add_argument('--optimizer_d', type=dict)
    parser.add_argument('--lr_scheduler_d', type=LRSchedulerConfig)
    parser.add_class_arguments(TokenizerDataModule, 'data')
    parser.add_dataclass_arguments(TrainingArguments, 'training')
    parser.link_arguments('training.max_steps', 'data.dl_conf.num_train_batches')
    parser.link_arguments('training.seed', 'data.seed')
    return parser

class MetricDict(dict):
    num_updates: int = 0

    @torch.no_grad()
    def update_metrics(self, metrics: dict):
        for k, v in metrics.items():
            if k in self:
                self[k] += v
            else:
                self[k] = v
        self.num_updates += 1

    @torch.no_grad()
    def reduce(self):
        ret = {
            k: v / self.num_updates
            for k, v in self.items()
        }
        self.clear()
        self.num_updates = 0
        return ret

def log_dict_split(fabric: Fabric, split: str, metric_dict: MetricDict, step: int | None = None):
    metric_dict = fabric.all_reduce(metric_dict.reduce())
    metric_dict = {
        f'{split}/{k}': v
        for k, v in metric_dict.items()
    }
    fabric.log_dict(metric_dict, step)

class Timer:
    @staticmethod
    def get_time():
        from time import monotonic_ns
        return monotonic_ns() / 1e6

    def __init__(self):
        self.t = self.get_time()
        self.step = 0

    def update(self, info: str):
        elapsed = (t := self.get_time()) - self.t
        self.t = t
        print(f'step {self.step} {info}: {elapsed:.2f} ms')

def main():
    torch.set_float32_matmul_precision('high')
    parser = get_parser()
    raw_args = parser.parse_args()
    args = parser.instantiate_classes(raw_args)
    training_args: TrainingArguments = args.training
    training_args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger('tokenizer', training_args.output_dir, project='PUMT')
    fabric = Fabric(precision='16-mixed', loggers=logger)
    fabric.seed_everything(training_args.seed)
    fabric.launch()
    save_dir = training_args.output_dir / Path(logger.experiment.dir).parent.name if fabric.is_global_zero else None
    save_dir = fabric.broadcast(save_dir)
    img_save_dir = save_dir / 'images'
    ckpt_save_dir = save_dir / 'checkpoints'
    conf_save_dir = save_dir / 'conf'
    if fabric.is_global_zero:
        save_dir.mkdir(parents=True)
        ckpt_save_dir.mkdir()
        conf_save_dir.mkdir()
        parser.save(raw_args, conf_save_dir / 'main.yaml')

    # the shape of our data varies, but enabling this still seems to be faster
    torch.backends.cudnn.benchmark = True
    vqvae: VQVAEModel = args.vqvae
    optimizer_g: Optimizer = build_optimizer(vqvae, args.optimizer_g)
    lr_scheduler_g = build_lr_scheduler(optimizer_g, args.lr_scheduler_g, training_args.max_steps)
    loss_module: VQGANLoss = args.loss
    optimizer_d: Optimizer = build_optimizer(loss_module.discriminator, args.optimizer_d)
    lr_scheduler_d = build_lr_scheduler(optimizer_d, args.lr_scheduler_d, training_args.max_steps)

    optimized_steps = 0
    state = {
        'vqvae': vqvae,
        'discriminator': loss_module.discriminator,
        'optimizer_g': optimizer_g,
        'optimizer_d': optimizer_d,
        'step': 0,
    }
    if training_args.resume_ckpt_path is None:
        ckpt = torch.load(training_args.pretrained_ckpt_path, map_location='cpu')
        if 'state_dict' in ckpt:
            print(f'[rank {fabric.global_rank}] load discriminator')
            load_ckpt(loss_module, ckpt, key_prefix='loss.')
            ckpt['state_dict'] = {
                k: v
                for k, v in ckpt['state_dict'].items() if not k.startswith('loss.')
            }
            print(f'[rank {fabric.global_rank}] load vqvae')
            load_ckpt(vqvae, ckpt)
        else:
            vqvae.load_state_dict(ckpt['vqvae'])
            loss_module.discriminator.load_state_dict(ckpt['discriminator'])
    else:
        fabric.load(training_args.resume_ckpt_path, state)
        print(f'resumed from {training_args.resume_ckpt_path}')
        optimized_steps = state['step']

    # setup model and optimizer after checkpoint loading, or optimizer.param_groups[i] will be different object
    vqvae, optimizer_g = fabric.setup(vqvae, optimizer_g)
    loss_module = fabric.to_device(loss_module)
    loss_module.discriminator, optimizer_d = fabric.setup(loss_module.discriminator, optimizer_d)
    datamodule: TokenizerDataModule = args.data
    train_loader, val_loader = fabric.setup_dataloaders(
        datamodule.train_dataloader(fabric.world_size, fabric.global_rank, optimized_steps),
        datamodule.val_dataloader(),
        use_distributed_sampler=False,
    )

    metric_dict = MetricDict()
    disc_loss_ema = training_args.disc_loss_ema_init
    disc_loss_item = math.inf
    for step, (x, aniso_d, paths) in enumerate(
        tqdm(train_loader, desc='training', ncols=80, disable=fabric.local_rank != 0, initial=optimized_steps),
        start=optimized_steps,
    ):
        x = 2 * x - 1
        x = SpatialTensor(x, aniso_d)
        vqvae.quantize.adjust_temperature(step, training_args.max_steps)
        x_rec, quant_out = vqvae(x)
        loss_module.discriminator.requires_grad_(False)
        loss, log_dict = loss_module.forward_gen(
            x, x_rec, quant_out.loss, max(disc_loss_ema, disc_loss_item) <= training_args.use_gan_th, vqvae.decoder.conv_out.weight, fabric
        )
        fabric.backward(loss)
        if training_args.max_norm_g is not None:
            fabric.clip_gradients(vqvae, optimizer_g, max_norm=training_args.max_norm_g)
        if step % lr_scheduler_g.frequency == 0:
            lr_scheduler_g.scheduler.step_update(step)
            fabric.log('lr-g', optimizer_g.param_groups[0]['lr'], step)
        optimizer_g.step()
        optimizer_g.zero_grad()
        loss_module.discriminator.requires_grad_(True)
        disc_loss, log_dict = loss_module.forward_disc(x, x_rec, log_dict)
        fabric.backward(disc_loss)
        if training_args.max_norm_d is not None:
            fabric.clip_gradients(loss_module.discriminator, optimizer_d, max_norm=training_args.max_norm_d)
        if step % lr_scheduler_d.frequency == 0:
            lr_scheduler_d.scheduler.step_update(step)
            fabric.log('lr-d', optimizer_d.param_groups[0]['lr'], step)
        optimizer_d.step()
        optimizer_d.zero_grad()
        disc_loss = fabric.all_reduce(disc_loss)
        disc_loss_ema = disc_loss_ema * training_args.disc_loss_momentum + (disc_loss_item := disc_loss.item()) * (1 - training_args.disc_loss_momentum)
        # fabric.all_reduce will modify value inplace with sum
        log_dict['disc_loss'] = disc_loss
        log_dict['disc_loss_ema'] = disc_loss_ema
        metric_dict.update_metrics(log_dict)
        optimized_steps = step + 1
        if optimized_steps % training_args.log_every_n_steps == 0 or optimized_steps == training_args.max_steps:
            log_dict_split(fabric, 'train', metric_dict, optimized_steps)
        if optimized_steps % training_args.plot_image_every_n_steps == 0 and fabric.is_global_zero:
            step_save_dir = img_save_dir / f'step-{optimized_steps}'
            step_save_dir.mkdir(parents=True)
            (step_save_dir / 'path.txt').write_text(str(paths[0]))
            for i in range(x.shape[2]):
                save_image((x[0, :, i] + 1) / 2, step_save_dir / f'{i}-origin.png')
                save_image((x_rec[0, :, i] + 1) / 2, step_save_dir / f'{i}-rec.png')
        state['step'] = optimized_steps
        if optimized_steps % training_args.save_every_n_steps == 0:
            fabric.save(ckpt_save_dir / f'step={optimized_steps}.ckpt', state)
        if optimized_steps % training_args.save_last_every_n_steps == 0:
            fabric.save(ckpt_save_dir / f'last.ckpt', state)

if __name__ == '__main__':
    main()
