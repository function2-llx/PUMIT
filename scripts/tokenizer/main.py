from dataclasses import dataclass
from pathlib import Path

import cytoolz
from jsonargparse import ActionConfigFile, ArgumentParser
from lightning import Fabric
from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import nn
from torch.optim import Optimizer
from torchvision.utils import save_image
from tqdm import tqdm

from luolib.models.utils import create_param_groups, load_ckpt, split_weight_decay_keys
from luolib.types import LRSchedulerConfig
from luolib.utils import DataKey
from pumt.tokenizer import TokenizerDataModule, VQVAEModel
from pumt.tokenizer.loss import VQGANLoss

@dataclass(kw_only=True)
class TrainingArguments:
    max_steps: int
    seed: int = 42
    log_every_n_steps: int = 50
    save_every_n_steps: int = 1000
    max_norm_g: int | float | None = None
    max_norm_d: int | float | None = None
    ckpt_path: Path
    output_dir: Path = Path('output/tokenizer')

def build_optimizer(model: nn.Module, optimizer_conf: dict):
    decay_keys, no_decay_keys = split_weight_decay_keys(model)
    return instantiate_class(
        create_param_groups(model.named_parameters(), decay_keys, no_decay_keys),
        optimizer_conf,
    )

def build_lr_scheduler(optimizer: Optimizer, config: LRSchedulerConfig) -> LRSchedulerConfig:
    config.scheduler = instantiate_class(optimizer, config.scheduler)
    from timm.scheduler.scheduler import Scheduler as TIMMScheduler
    assert isinstance(config.scheduler, TIMMScheduler)
    return config

def parse_args():
    parser = ArgumentParser(parser_mode='omegaconf')
    parser.add_argument('-c', '--config', action=ActionConfigFile)
    parser.add_class_arguments(VQVAEModel, 'vqvae')
    parser.add_argument('--optimizer_g', type=dict)
    parser.add_argument('--lr_scheduler_g', type=LRSchedulerConfig)
    parser.add_class_arguments(VQGANLoss, 'loss')
    parser.add_argument('--optimizer_d', type=dict)
    parser.add_argument('--lr_scheduler_d', type=LRSchedulerConfig)
    parser.add_class_arguments(TokenizerDataModule, 'data')
    parser.add_dataclass_arguments(TrainingArguments, 'training')
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    return args

def log_dict_split(fabric: Fabric, split: str, log_dict: dict, step: int | None = None):
    log_dict = {
        f'{split}/{k}': v
        for k, v in log_dict.items()
    }
    fabric.log_dict(log_dict, step)

def main():
    torch.set_float32_matmul_precision('high')
    args = parse_args()
    training_args: TrainingArguments = args.training
    logger = WandbLogger(
        'tokenizer',
        training_args.output_dir,
        project='PUMT',
    )
    fabric = Fabric(
        precision='16-mixed',
        # plugins=MixedPrecision(
        #     '16-mixed',
        #     'cuda',
        #     # GradScaler(init_scale=4096),
        # ),
        loggers=logger,
    )
    fabric.seed_everything(training_args.seed)
    ckpt = torch.load(training_args.ckpt_path, map_location='cpu')
    vqvae: VQVAEModel = args.vqvae
    optimizer_g: Optimizer = build_optimizer(vqvae, args.optimizer_g)
    lr_scheduler_g = build_lr_scheduler(optimizer_g, args.lr_scheduler_g)
    loss_module: VQGANLoss = args.loss
    load_ckpt(loss_module, ckpt, key_prefix='loss.')
    ckpt['state_dict'] = {
        k: v
        for k, v in ckpt['state_dict'].items() if not k.startswith('loss.')
    }
    load_ckpt(vqvae, ckpt)
    optimizer_d: Optimizer = build_optimizer(loss_module.discriminator, args.optimizer_d)
    lr_scheduler_d = build_lr_scheduler(optimizer_d, args.lr_scheduler_d)
    datamodule: TokenizerDataModule = args.data
    fabric.launch()
    if fabric.is_global_zero:
        save_dir = training_args.output_dir / logger.version
        img_save_dir = save_dir / 'images'
    datamodule.world_size = fabric.world_size
    vqvae, optimizer_g = fabric.setup(vqvae, optimizer_g)
    loss_module = fabric.to_device(loss_module)
    loss_module.discriminator, optimizer_d = fabric.setup(loss_module.discriminator, optimizer_d)
    train_loader, val_loader = fabric.setup_dataloaders(datamodule.train_dataloader(), datamodule.val_dataloader())
    for step, batch in enumerate(tqdm(train_loader, ncols=80, desc='training')):
        x, spacing = cytoolz.get([DataKey.IMG, DataKey.SPACING], batch)
        vqvae.quantize.adjust_temperature(step, training_args.max_steps)
        x_rec, quant_out = vqvae(x, spacing)
        loss_module.discriminator.requires_grad_(False)
        loss_module.adjust_gan_weight(step)
        loss, log_dict = loss_module.forward_gen(x, x_rec, spacing, quant_out.loss)
        fabric.backward(loss)
        if training_args.max_norm_g is not None:
            fabric.clip_gradients(vqvae, optimizer_g, max_norm=training_args.max_norm_g)
        if step % lr_scheduler_g.frequency == 0:
            lr_scheduler_g.scheduler.step_update(step)
            fabric.log('lr-g', optimizer_g.param_groups[0]['lr'], step)
        optimizer_g.step()
        optimizer_g.zero_grad()
        loss_module.discriminator.requires_grad_(True)
        disc_loss, log_dict = loss_module.forward_disc(x, x_rec, spacing, log_dict)
        fabric.backward(disc_loss)
        if training_args.max_norm_d is not None:
            fabric.clip_gradients(loss_module.discriminator, optimizer_d, max_norm=training_args.max_norm_d)
        if step % lr_scheduler_d.frequency == 0:
            lr_scheduler_d.scheduler.step_update(step)
            fabric.log('lr-d', optimizer_d.param_groups[0]['lr'], step)
        optimizer_d.step()
        optimizer_d.zero_grad()
        if (optimized_steps := step + 1) % training_args.log_every_n_steps == 0:
            log_dict = fabric.all_reduce(log_dict)
            log_dict_split(fabric, 'train', log_dict, optimized_steps)
            if fabric.is_global_zero:
                step_dir = img_save_dir / f'step-{optimized_steps}'
                step_dir.mkdir(parents=True)
                for i in range(x.shape[2]):
                    save_image((x[0, :, i] + 1) / 2, step_dir / f'{i}-origin.png')
                    save_image((x_rec[0, :, i] + 1) / 2, step_dir / f'{i}-rec.png')

if __name__ == '__main__':
    main()
