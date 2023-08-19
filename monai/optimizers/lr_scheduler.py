# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, _LRScheduler

__all__ = ["LinearLR", "ExponentialLR"]


class _LRSchedulerMONAI(_LRScheduler):
    """Base class for increasing the learning rate between two boundaries over a number
    of iterations"""

    def __init__(self, optimizer: Optimizer, end_lr: float, num_iter: int, last_epoch: int = -1) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            end_lr: the final learning rate.
            num_iter: the number of iterations over which the test occurs.
            last_epoch: the index of last epoch.
        Returns:
            None
        """
        self.end_lr = end_lr
        self.num_iter = num_iter
        super().__init__(optimizer, last_epoch)


class LinearLR(_LRSchedulerMONAI):
    """Linearly increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr + r * (self.end_lr - base_lr) for base_lr in self.base_lrs]


class ExponentialLR(_LRSchedulerMONAI):
    """Exponentially increases the learning rate between two boundaries over a number of
    iterations.
    """

    def get_lr(self):
        r = self.last_epoch / (self.num_iter - 1)
        return [base_lr * (self.end_lr / base_lr) ** r for base_lr in self.base_lrs]


class WarmupCosineSchedule(LambdaLR):
    """Linear warmup and then cosine decay.
    Based on https://huggingface.co/ implementation.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        t_total: int,
        end_lr: float = 0.0,
        cycles: float = 0.5,
        last_epoch: int = -1,
        warmup_multiplier: float = 0,
    ) -> None:
        """
        Args:
            optimizer: wrapped optimizer.
            warmup_steps: number of warmup iterations.
            t_total: total number of training iterations.
            end_lr: the final learning rate. Defaults to 0.0.
            cycles: cosine cycles parameter.
            last_epoch: the index of last epoch.
            warmup_multiplier: if provided, starts the linear warmup from this fraction of the initial lr.
                Must be in 0..1 interval. Defaults to 0
        Returns:
            None
        """
        self.warmup_steps = min(max(warmup_steps, 0), t_total)
        self.warmup_multiplier = warmup_multiplier
        self.t_total = t_total
        self.cycles = cycles
        self.end_lr = end_lr
        if warmup_multiplier < 0 or warmup_multiplier > 1:
            raise ValueError("warmup_multiplier must be in 0..1 range")
        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            f = float(step) / float(max(1.0, self.warmup_steps))
            return self.warmup_multiplier + (1 - self.warmup_multiplier) * f
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))

    def get_lr(self):
        current_lr = [base_lr * lmbda(self.last_epoch) for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]
        if self.last_epoch < self.warmup_steps:
            return current_lr
        else:
            return [max(self.end_lr, _current_lr) for _current_lr in current_lr]
