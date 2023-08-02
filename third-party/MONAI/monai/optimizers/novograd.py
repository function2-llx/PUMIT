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

from collections.abc import Callable, Iterable
from typing import TypeVar

import torch
from torch.optim import Optimizer

T = TypeVar("T")


class Novograd(Optimizer):
    """
    Novograd based on `Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks
    <https://arxiv.org/pdf/1905.11286.pdf>`_.
    The code is adapted from the implementations in `Jasper for PyTorch
    <https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechRecognition/Jasper/common/optimizers.py>`_,
    and `OpenSeq2Seq <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/optimizers/novograd.py>`_.

    Args:
        params: iterable of parameters to optimize or dicts defining parameter groups.
        lr: learning rate. Defaults to 1e-3.
        betas: coefficients used for computing running averages of gradient and its square. Defaults to (0.9, 0.98).
        eps: term added to the denominator to improve numerical stability. Defaults to 1e-8.
        weight_decay: weight decay (L2 penalty). Defaults to 0.
        grad_averaging: gradient averaging. Defaults to ``False``.
        amsgrad: whether to use the AMSGrad variant of this algorithm from the paper
            `On the Convergence of Adam and Beyond <https://arxiv.org/pdf/1904.09237.pdf>`_. Defaults to ``False``.
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.98),
        eps: float = 1e-8,
        weight_decay: float = 0,
        grad_averaging: bool = False,
        amsgrad: bool = False,
    ):
        if 0.0 > lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if 0.0 > eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if 0.0 > weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, grad_averaging=grad_averaging, amsgrad=amsgrad
        )

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure: Callable[[], T] | None = None) -> T | None:
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss. Defaults to ``None``.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Sparse gradients are not supported.")
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros([]).to(state["exp_avg"].device)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros([]).to(state["exp_avg"].device)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                norm = torch.sum(torch.pow(grad, 2))

                if exp_avg_sq == 0:
                    exp_avg_sq.copy_(norm)
                else:
                    exp_avg_sq.mul_(beta2).add_(norm, alpha=1 - beta2)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                grad.div_(denom)
                if group["weight_decay"] != 0:
                    grad.add_(p.data, alpha=group["weight_decay"])
                if group["grad_averaging"]:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                p.data.add_(exp_avg, alpha=-group["lr"])

        return loss
