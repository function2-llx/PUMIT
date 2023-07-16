from dataclasses import dataclass

import cytoolz
from jsonargparse import ActionConfigFile, ArgumentParser
from lightning import Fabric, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.cli import instantiate_class
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins import MixedPrecisionPlugin
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from tqdm import tqdm

from luolib.models.utils import get_no_weight_decay_keys, split_by_weight_decay
from luolib.utils import DataKey
from pumt.tokenizer import PatchDiscriminator, TokenizerDataModule, VQVAEModel

@dataclass
class TrainingArguments:
    max_steps: int
    seed: int = 42

def build_optimizer(model: nn.Module, optimizer_conf: dict):
    no_weight_decay_keys = get_no_weight_decay_keys(model)
    return instantiate_class(
        split_by_weight_decay(model.named_parameters(), no_weight_decay_keys),
        optimizer_conf,
    )

def parse_args():
    parser = ArgumentParser(parser_mode='omegaconf')
    parser.add_argument('-c', '--config', action=ActionConfigFile)
    parser.add_class_arguments(VQVAEModel, 'vqvae')
    parser.add_argument('--optimizer_g', type=dict)
    parser.add_class_arguments(PatchDiscriminator, 'discriminator')
    parser.add_argument('--optimizer_d', type=dict)
    parser.add_class_arguments(TokenizerDataModule, 'data')
    parser.add_dataclass_arguments(TrainingArguments, 'training')
    args = parser.parse_args()
    args = parser.instantiate_classes(args)
    return args

def main():
    args = parse_args()
    training_args: TrainingArguments = args.training
    seed_everything(training_args.seed)
    vqvae: VQVAEModel = args.vqvae
    optimizer_g = build_optimizer(vqvae, args.optimizer_g)
    discriminator: PatchDiscriminator = args.discriminator
    optimizer_d = build_optimizer(discriminator, args.optimizer_d)
    datamodule: TokenizerDataModule = args.data
    fabric = Fabric(
        plugins=MixedPrecisionPlugin(
            '16-mixed',
            'cuda',
            GradScaler(init_scale=4096),
        ),
        loggers=WandbLogger('tokenizer', 'output/tokenizer', 'PUMT'),
    )
    fabric.launch()
    fabric.setup(vqvae, optimizer_g)
    fabric.setup(discriminator, optimizer_d)
    train_loader, val_loader = fabric.setup_dataloaders(datamodule.train_dataloader(), datamodule.val_dataloader())
    for step, batch in enumerate(tqdm(train_loader, ncols=80, desc='training')):
        x, spacing = cytoolz.get([DataKey.IMG, DataKey.SPACING], batch)
        x_rec, quant_out = vqvae(x, spacing)
        # self.toggle_optimizer(optimizer)
        # x_rec, quant_out = self(x, spacing)


if __name__ == '__main__':
    main()
