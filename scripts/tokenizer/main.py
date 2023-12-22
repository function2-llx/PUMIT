from dataclasses import dataclass
import math
from pathlib import Path

from jsonargparse import ActionConfigFile, ArgumentParser
from lightning.fabric import Fabric
from lightning.pytorch.loggers import WandbLogger
import torch
from torchvision.utils import save_image
from tqdm import tqdm

from luolib.lightning import OptimizationConf, build_hybrid_optimization
from luolib.models import spadop
from luolib.models.utils import load_ckpt
from luolib.utils.grad import grad_norm

from pumit.datamodule import PUMITDataModule
from pumit.tokenizer import VQVTLoss, VQVisualTokenizer
from pumit.tokenizer.vq import GumbelVQ

@dataclass(kw_only=True)
class TrainingArguments:
    max_steps: int
    seed: int = 42
    log_every_n_steps: int = 50
    save_last_every_n_steps: int = 1000
    plot_image_every_n_steps: int = 100
    save_every_n_steps: int = 10000
    max_norm_g: float | None = None
    max_norm_d: float | None = None
    resume_ckpt_path: Path | None = None
    pretrained_ckpt_path: Path | None = None
    output_dir: Path
    exp_name: str
    disc_loss_ema_init: float = 0.1
    disc_loss_momentum: float = 0.9
    use_gan_th: float = 0.03
    benchmark: bool = False
    pretrained_codebook: Path | None = None
    fix_codebook: bool = False

def get_parser():
    parser = ArgumentParser(parser_mode='omegaconf')
    parser.add_argument('-c', '--config', action=ActionConfigFile)
    parser.add_subclass_arguments(VQVisualTokenizer, 'model')
    parser.add_argument('--optim_g', type=list[OptimizationConf], enable_path=True)
    parser.add_class_arguments(VQVTLoss, 'loss')
    parser.add_argument('--optim_d', type=list[OptimizationConf], enable_path=True)
    parser.add_class_arguments(PUMITDataModule, 'data')
    parser.add_dataclass_arguments(TrainingArguments, 'training')
    parser.link_arguments('training.max_steps', 'data.dataloader.num_train_batches')
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

def check_loss(loss: torch.Tensor, state: dict, batch, save_dir: Path, fabric: Fabric):
    if loss.isfinite() or save_dir.exists():
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    fabric.save(save_dir / 'checkpoint.ckpt', state)
    torch.save(batch, save_dir / 'batch.pt')

def main():
    torch.set_float32_matmul_precision('high')
    torch.multiprocessing.set_start_method('spawn')
    parser = get_parser()
    raw_args = parser.parse_args()
    args = parser.instantiate_classes(raw_args)
    training_args: TrainingArguments = args.training
    training_args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(training_args.exp_name, training_args.output_dir, project='pumit')
    fabric = Fabric(precision='16-mixed', loggers=logger)
    fabric.seed_everything(training_args.seed)
    fabric.launch()
    save_dir = training_args.output_dir / Path(logger.experiment.dir).parent.name if fabric.is_global_zero else None
    save_dir = fabric.broadcast(save_dir)
    img_save_dir = save_dir / 'images'
    ckpt_save_dir = save_dir / 'checkpoints'
    if fabric.is_global_zero:
        save_dir.mkdir(parents=True)
        ckpt_save_dir.mkdir()
        parser.save(raw_args, save_dir / 'conf.yaml', multifile=False)

    # the shape of our data varies, but enabling this still seems to be faster
    torch.backends.cudnn.benchmark = training_args.benchmark
    model: VQVisualTokenizer = args.model
    optimizer_g, lr_scheduler_g = build_hybrid_optimization(model, args.optim_g)
    loss_module: VQVTLoss = args.loss
    optimizer_d, lr_scheduler_d = build_hybrid_optimization(loss_module.discriminator, args.optim_d)

    if fabric.is_global_zero:
        Path(save_dir / 'model.txt').write_text(repr(model))
        Path(save_dir / 'loss.txt').write_text(repr(loss_module))

    optimized_steps = 0
    state = {
        'model': model,
        'discriminator': loss_module.discriminator,
        'optimizer_g': optimizer_g,
        'optimizer_d': optimizer_d,
        'step': 0,
    }
    if training_args.resume_ckpt_path is None:
        if training_args.pretrained_ckpt_path is not None:
            ckpt = torch.load(training_args.pretrained_ckpt_path, map_location='cpu')
            if 'state_dict' in ckpt:
                print(f'[rank {fabric.global_rank}] load discriminator')
                load_ckpt(loss_module, ckpt, key_prefix='loss.')
                ckpt['state_dict'] = {
                    k: v
                    for k, v in ckpt['state_dict'].items() if not k.startswith('loss.')
                }
                print(f'[rank {fabric.global_rank}] load tokenizer')
                load_ckpt(model, ckpt)
            else:
                load_ckpt(model, ckpt, 'model')
                load_ckpt(loss_module.discriminator, ckpt, 'discriminator')
        if training_args.pretrained_codebook is not None:
            model.quantize.load_state_dict(torch.load(training_args.pretrained_codebook))
            print(f'load pre-trained codebook from {training_args.pretrained_codebook}')
    else:
        fabric.load(training_args.resume_ckpt_path, state)
        print(f'resumed from {training_args.resume_ckpt_path}')
        optimized_steps = state['step']
    if training_args.fix_codebook:
        model.quantize.requires_grad_(False)

    # setup model and optimizer after checkpoint loading, or optimizer.param_groups[i] will be different object
    model, optimizer_g = fabric.setup(model, optimizer_g)
    loss_module = fabric.to_device(loss_module)
    loss_module.discriminator, optimizer_d = fabric.setup(loss_module.discriminator, optimizer_d)
    datamodule: PUMITDataModule = args.data
    datamodule.setup_ddp(fabric.local_rank, fabric.global_rank, fabric.world_size)
    train_loader, val_loader = fabric.setup_dataloaders(
        datamodule.train_dataloader(optimized_steps),
        datamodule.val_dataloader(),
        use_distributed_sampler=False,
    )
    datamodule.dataset_info.to_csv(save_dir / 'dataset-info.csv')
    datamodule.dataset_info.to_excel(save_dir / 'dataset-info.xlsx')

    metric_dict = MetricDict()
    disc_loss_ema = training_args.disc_loss_ema_init
    disc_loss_item = math.inf
    for step, batch in enumerate(
        tqdm(train_loader, desc='training', ncols=80, disable=fabric.local_rank != 0, initial=optimized_steps),
        start=optimized_steps,
    ):
        x, not_rgb, aniso_d, paths = batch
        x = 2 * x - 1
        x = spadop.SpatialTensor(x, aniso_d)
        if isinstance(model.quantize, GumbelVQ):
            model.quantize.adjust_temperature(step, training_args.max_steps)
        x_rec, vq_out = model(x, autoencode=True, fabric=fabric)
        loss_module.discriminator.requires_grad_(False)
        loss, log_dict = loss_module.forward_gen(
            x, x_rec, not_rgb, vq_out,
            max(disc_loss_ema, disc_loss_item) <= training_args.use_gan_th, model.get_ref_param(),
            fabric,
        )
        fabric.backward(loss)
        fabric.log('grad-norm-g', grad_norm(model), step)
        check_loss(loss, state, batch, save_dir / 'bad-g', fabric)
        if training_args.max_norm_g is not None:
            fabric.clip_gradients(model, optimizer_g, max_norm=training_args.max_norm_g)
        if step % lr_scheduler_g.frequency == 0:
            lr_scheduler_g.scheduler.step(step)
            fabric.log('lr-g', optimizer_g.param_groups[0]['lr'], step)
        optimizer_g.step()
        optimizer_g.zero_grad()
        loss_module.discriminator.requires_grad_(True)
        disc_loss, log_dict = loss_module.forward_disc(x, x_rec, not_rgb, log_dict)
        fabric.backward(disc_loss)
        fabric.log('grad-norm-d', grad_norm(loss_module.discriminator), step)
        check_loss(disc_loss, state, batch, save_dir / 'bad-d', fabric)
        if training_args.max_norm_d is not None:
            fabric.clip_gradients(loss_module.discriminator, optimizer_d, max_norm=training_args.max_norm_d)
        if step % lr_scheduler_d.frequency == 0:
            lr_scheduler_d.scheduler.step(step)
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
            if isinstance(model.quantize, GumbelVQ):
                fabric.log('train/temperature', model.quantize.temperature, optimized_steps)
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
