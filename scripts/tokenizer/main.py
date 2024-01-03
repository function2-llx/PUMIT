from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path

from jsonargparse import ActionConfigFile, ArgumentParser
from lightning.fabric import Fabric as FabricBase
from lightning.fabric.plugins import Precision
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer, _unwrap_objects
from lightning.pytorch.loggers import WandbLogger
import torch
from torch import nn
from torch.optim import Optimizer
from torchvision.utils import save_image
from tqdm import trange

from luolib.lightning import OptimizationConf, build_hybrid_optimization
from luolib.models.spadop import SpatialTensor
from luolib.models.utils import load_ckpt
from luolib.utils import ema_update
from luolib.utils.grad import grad_norm

from pumit.datamodule import PUMITDataModule
from pumit.tokenizer import LOGIT_EPS, VQVTLoss, VQVisualTokenizer
from pumit.tokenizer.discriminator import PatchDiscriminatorBase, get_disc_scores
from pumit.tokenizer.loss import hinge_gan_loss
from pumit.tokenizer.vq import GumbelVQ
from pumit.transforms import rgb_to_gray

@dataclass(kw_only=True)
class TrainingArguments:
    max_steps: int
    seed: int = 42
    log_every_n_steps: int = 20
    save_last_every_n_steps: int = 1000
    plot_image_every_n_steps: int = 100
    save_every_n_steps: int = 10000
    accumulate_grad_batches: int = 1
    grad_norm_g: float | None = None
    grad_norm_d: float | None = None
    resume_ckpt_path: Path | None = None
    pretrained_ckpt_path: Path | None = None
    output_root: Path = 'output'
    output_dir: Path | None = None
    exp_name: str
    use_gan_th: float
    gan_start_step: int
    benchmark: bool = False
    pretrained_codebook: Path | None = None
    fix_codebook: bool = False
    precision: str = '32-true'

    @dataclass
    class TeacherUpdate:
        base_momentum: float
        schedule: bool
    teacher_update: TeacherUpdate

    def get_teacher_update_momentum(self, step: int) -> float:
        if not self.teacher_update.schedule:
            return self.teacher_update.base_momentum
        return 1 - (1 - self.teacher_update.base_momentum) * (math.cos(step / self.max_steps * math.pi) + 1) / 2

def get_parser():
    parser = ArgumentParser(parser_mode='omegaconf')
    parser.add_argument('-c', '--config', action=ActionConfigFile)
    parser.add_subclass_arguments(VQVisualTokenizer, 'model')
    parser.add_argument('--optim_g', type=list[OptimizationConf], enable_path=True)
    parser.add_class_arguments(VQVTLoss, 'loss')
    parser.add_argument('--optim_d', type=list[OptimizationConf], enable_path=True)
    parser.add_class_arguments(PUMITDataModule, 'data')
    parser.add_dataclass_arguments(TrainingArguments, 'training')
    parser.link_arguments(
        ('training.max_steps', 'training.accumulate_grad_batches'),
        'data.dataloader.num_train_batches',
        lambda max_steps, accumulate_grad_batches: max_steps * accumulate_grad_batches,
    )
    return parser

class MetricDict(dict):
    def update_metrics(self, metrics: dict):
        for k, v in metrics.items():
            self.setdefault(k, []).append(v)

    @torch.no_grad()
    def reduce(self):
        ret = {
            k: sum(v) / len(v)
            for k, v in self.items()
        }
        self.clear()
        return ret

def reduce_log(fabric: FabricBase, prefix: str, metric_dict: MetricDict, step: int | None = None):
    metric_dict = fabric.all_reduce(metric_dict.reduce())
    if prefix != '':
        prefix = f'{prefix}/'
    log_dict = {
        f'{prefix}{k}': v
        for k, v in metric_dict.items()
    }
    fabric.log_dict(log_dict, step)

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

def check_loss(loss: torch.Tensor, state: dict, batch, save_dir: Path, fabric: FabricBase):
    if save_dir.exists():
        return
    if not fabric.all_gather(is_finite := loss.isfinite()).all():
        if fabric.is_global_zero:
            save_dir.mkdir(parents=True, exist_ok=True)
        fabric.save(save_dir / 'checkpoint.ckpt', state)
        if not is_finite:
            torch.save(batch, save_dir / f'batch-{fabric.local_rank}.pt')

class Fabric(FabricBase):
    def unscale_gradients(self, optimizer: _FabricOptimizer | Optimizer):
        optimizer = _unwrap_objects(optimizer)
        self.strategy.precision.unscale_gradients(optimizer)

    def clip_gradients(
        self,
        module: nn.Module | _FabricModule,
        optimizer: Optimizer | _FabricOptimizer,
        clip_val: float | int | None = None,
        max_norm: float | int | None = None,
        norm_type: float | int = 2.0,
        error_if_nonfinite: bool = True,
        *,
        unscale_gradients: bool = True,
        return_norm: bool = False,
    ) -> torch.Tensor | None:
        """
        clip gradients for both value and norm
        Args:
            return_norm: always return gradient norm after clipping by value
        Returns:
        """
        if unscale_gradients:
            self.unscale_gradients(optimizer)
        if clip_val is None and max_norm is None:
            return grad_norm(module) if return_norm else None
        optimizer = _unwrap_objects(optimizer)
        module = _unwrap_objects(module)
        parameters = self.strategy.precision.main_params(optimizer)
        if clip_val is not None:
            torch.nn.utils.clip_grad_value_(parameters, clip_value=clip_val)
            if max_norm is None:
                return grad_norm(module) if return_norm else None
            return torch.nn.utils.clip_grad_norm_(
                self.strategy.precision.main_params(optimizer),
                max_norm, norm_type, error_if_nonfinite,
            )
        else:
            return self.strategy.clip_gradients_norm(
                module, optimizer,
                max_norm, norm_type, error_if_nonfinite,
            )

def main():
    torch.set_float32_matmul_precision('medium')
    torch.multiprocessing.set_start_method('spawn')
    parser = get_parser()
    raw_args = parser.parse_args()
    args = parser.instantiate_classes(raw_args)
    training_args: TrainingArguments = args.training
    if training_args.output_dir is None:
        training_args.output_dir = training_args.output_root / training_args.exp_name
    training_args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(training_args.exp_name, training_args.output_dir, project='pumit')
    fabric = Fabric(precision=training_args.precision, loggers=logger)
    print('precision:', fabric.strategy.precision.precision)
    fabric.seed_everything(training_args.seed)
    fabric.launch()
    save_dir = training_args.output_dir / Path(logger.experiment.dir).parent.name if fabric.is_global_zero else None
    save_dir = fabric.broadcast(save_dir)
    img_save_dir = save_dir / 'images'
    ckpt_save_dir = save_dir / 'checkpoints'
    if fabric.is_global_zero:
        save_dir.mkdir(parents=True)
        ckpt_save_dir.mkdir()
        parser.save(raw_args, save_dir / 'conf.yaml', multifile=False, skip_none=False)

    # the shape of our data varies, but enabling this still seems to be faster
    torch.backends.cudnn.benchmark = training_args.benchmark
    model: VQVisualTokenizer | _FabricModule = args.model
    optimizer_g, lr_scheduler_g = build_hybrid_optimization(model, args.optim_g)
    loss_module: VQVTLoss = args.loss
    disc_student: PatchDiscriminatorBase | _FabricModule = deepcopy(loss_module.discriminator)
    disc_student.train()
    disc_student.requires_grad_(True)
    loss_module.set_gan_ref_param(model.get_ref_param())
    optimizer_d, lr_scheduler_d = build_hybrid_optimization(disc_student, args.optim_d)

    if fabric.is_global_zero:
        Path(save_dir / 'model.txt').write_text(repr(model))
        Path(save_dir / 'loss.txt').write_text(repr(loss_module))

    # setup optimizer before loading the checkpoint: https://github.com/Lightning-AI/pytorch-lightning/issues/19225
    model, optimizer_g = fabric.setup(model, optimizer_g)
    loss_module = fabric.to_device(loss_module)
    disc_student, optimizer_d = fabric.setup(disc_student, optimizer_d)
    # override 16-mixed precision plugin, Fabric should have provided such flexible interface in .setup()
    disc_student._precision = Precision()
    state = {
        'model': model,
        'disc': disc_student,
        'loss': loss_module,
        'optimizer_g': optimizer_g,
        'optimizer_d': optimizer_d,
        'step': 0,
    }
    # NOTE: learning rate update frequency should divide the step to recover the learning rate
    load_state(model, loss_module, state, training_args, fabric)
    optimization_step = state['step']

    datamodule: PUMITDataModule = args.data
    datamodule.setup_ddp(fabric.local_rank, fabric.global_rank, fabric.world_size)
    train_loader, val_loader = fabric.setup_dataloaders(
        datamodule.train_dataloader(optimization_step * training_args.accumulate_grad_batches),
        datamodule.val_dataloader(),
        use_distributed_sampler=False,
    )
    datamodule.dataset_info.to_csv(save_dir / 'dataset-info.csv')
    datamodule.dataset_info.to_excel(save_dir / 'dataset-info.xlsx')

    metric_dict = MetricDict()
    grad_norm_dict = MetricDict()
    train_loader_iter = iter(train_loader)
    for step in trange(
        training_args.max_steps,
        desc='training', dynamic_ncols=True, disable=fabric.local_rank != 0, initial=optimization_step
    ):
        optimization_step = step + 1
        if isinstance(model.quantize, GumbelVQ):
            model.quantize.adjust_temperature(step, training_args.max_steps)
        for batch_idx in range(training_args.accumulate_grad_batches):
            batch = next(train_loader_iter)
            accumulating = batch_idx + 1 < training_args.accumulate_grad_batches
            # 0. prepare input
            x, not_rgb, aniso_d, paths = batch
            x = SpatialTensor(x, aniso_d)
            x_logit = x.logit(LOGIT_EPS)
            # 1. generator part
            with fabric.no_backward_sync(model, accumulating):
                x_rec, x_rec_logit, vq_out = model(x_logit, autoencode=True, fabric=fabric)
                loss, log_dict = loss_module.forward_gen(
                    x, x_logit, x_rec, x_rec_logit, vq_out,
                    training_args.use_gan_th, fabric,
                )
                fabric.backward(loss / training_args.accumulate_grad_batches)
            check_loss(loss, state, batch, save_dir / 'bad-g', fabric)
            # 2. discriminator part
            with fabric.no_backward_sync(disc_student, accumulating):
                score_real, score_fake = get_disc_scores(disc_student, x_logit, x_rec_logit[:, :x.shape[1]].detach())
                real_loss, fake_loss = hinge_gan_loss(score_real, score_fake)
                disc_loss = 0.5 * (real_loss + fake_loss)
                log_dict.update({
                    'disc/loss': disc_loss,
                    'disc/real': real_loss,
                    'disc/fake': fake_loss,
                })
                fabric.backward(disc_loss / training_args.accumulate_grad_batches)
            check_loss(disc_loss, state, batch, save_dir / 'bad-d', fabric)
            metric_dict.update_metrics(log_dict)
            if accumulating:
                continue
            # 3. complete gradient accumulation, perform optimization
            # 3.1 optimize generator
            # use `step` here instead of `optimization_step` to adjust the learning rate in the beginning of the training
            if step % lr_scheduler_g.frequency == 0:
                lr_scheduler_g.scheduler.step(step)
                fabric.log('lr-g', optimizer_g.param_groups[0]['lr'], step)
            fabric.unscale_gradients(optimizer_g)
            grad_norm_dict.update_metrics({'gen': grad_norm(model)})
            fabric.clip_gradients(model, optimizer_g, max_norm=training_args.grad_norm_g, unscale_gradients=False)
            grad_norm_dict.update_metrics({'gen-clipped': grad_norm(model)})
            optimizer_g.step()
            optimizer_g.zero_grad()
            # 3.2 optimize discriminator
            if step % lr_scheduler_d.frequency == 0:
                lr_scheduler_d.scheduler.step(step)
                fabric.log('lr-d', optimizer_d.param_groups[0]['lr'], step)
            fabric.unscale_gradients(optimizer_d)
            grad_norm_dict.update_metrics({'disc': grad_norm(disc_student)})
            fabric.clip_gradients(disc_student, optimizer_d, max_norm=training_args.grad_norm_d, unscale_gradients=False)
            grad_norm_dict.update_metrics({'disc-clipped': grad_norm(disc_student)})
            optimizer_d.step()
            optimizer_d.zero_grad()
            with torch.no_grad():
                teacher_update_momentum = training_args.get_teacher_update_momentum(step)
                metric_dict.update_metrics({'disc/teacher_update_momentum': teacher_update_momentum})
                # FIXME: order?
                for p_teacher, p_student in zip(loss_module.discriminator.parameters(), disc_student.parameters()):
                    ema_update(p_teacher, p_student, teacher_update_momentum)
            # -1. logging, plotting, saving, etc.
            if optimization_step % training_args.log_every_n_steps == 0 or optimization_step == training_args.max_steps:
                reduce_log(fabric, 'train', metric_dict, optimization_step)
                reduce_log(fabric, 'grad-norm', grad_norm_dict, optimization_step)
                if isinstance(model.quantize, GumbelVQ):
                    fabric.log('train/temperature', model.quantize.temperature, optimization_step)
            if optimization_step % training_args.plot_image_every_n_steps == 0 and fabric.is_global_zero:
                plot_rec(img_save_dir, optimization_step, paths, x, x_rec)
            state['step'] = optimization_step
            if optimization_step % training_args.save_every_n_steps == 0:
                fabric.save(ckpt_save_dir / f'step={optimization_step}.ckpt', state)
            if optimization_step % training_args.save_last_every_n_steps == 0:
                fabric.save(ckpt_save_dir / f'last.ckpt', state)
            break

def load_state(
    model: VQVisualTokenizer,
    loss_module: VQVTLoss,
    state: dict,
    training_args: TrainingArguments,
    fabric: FabricBase,
):
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
    if training_args.fix_codebook:
        model.quantize.requires_grad_(False)

def plot_rec(img_save_dir: Path, optimized_steps: int, paths: list[Path], x: torch.Tensor, x_rec: torch.Tensor):
    step_save_dir = img_save_dir / f'step-{optimized_steps}'
    step_save_dir.mkdir(parents=True)
    (step_save_dir / 'path.txt').write_text(str(paths[0]))
    x_rec_gray = rgb_to_gray(x_rec[0])
    for i in range(x.shape[2]):
        save_image(x[0, :, i], step_save_dir / f'{i}-origin.png')
        save_image(x_rec[0, :, i], step_save_dir / f'{i}-rec.png')
        save_image(x_rec_gray[:, i], step_save_dir / f'{i}-rec-gray.png')

if __name__ == '__main__':
    main()
