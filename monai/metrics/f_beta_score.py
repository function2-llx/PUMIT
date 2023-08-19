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

from collections.abc import Sequence

import torch

from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction

from .metric import CumulativeIterationMetric


class FBetaScore(CumulativeIterationMetric):
    def __init__(
        self,
        beta: float = 1.0,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if y_pred.ndimension() < 2:
            raise ValueError("y_pred should have at least two dimensions.")

        return get_f_beta_score(y_pred=y_pred, y=y, include_background=self.include_background)

    def aggregate(
        self, compute_sample: bool = False, reduction: MetricReduction | str | None = None
    ) -> Sequence[torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        results: list[torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = []
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        f = compute_f_beta_score(f, self.beta)
        if self.get_not_nans:
            results.append((f, not_nans))
        else:
            results.append(f)

        return results


def get_f_beta_score(y_pred: torch.Tensor, y: torch.Tensor, include_background: bool = True) -> torch.Tensor:
    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    # get confusion matrix related metric
    batch_size, n_class = y_pred.shape[:2]
    # convert to [BNS], where S is the number of pixels for one sample.
    # As for classification tasks, S equals to 1.
    y_pred = y_pred.view(batch_size, n_class, -1)
    y = y.view(batch_size, n_class, -1)
    tp = ((y_pred + y) == 2).float()
    tn = ((y_pred + y) == 0).float()

    tp = tp.sum(dim=[2])
    tn = tn.sum(dim=[2])
    p = y.sum(dim=[2])
    n = y.shape[-1] - p

    fn = p - tp
    fp = n - tn

    return torch.stack([tp, fp, tn, fn], dim=-1)


def compute_f_beta_score(confusion_matrix: torch.Tensor, beta: float) -> torch.Tensor:
    input_dim = confusion_matrix.ndimension()
    if input_dim == 1:
        confusion_matrix = confusion_matrix.unsqueeze(dim=0)
    if confusion_matrix.shape[-1] != 4:
        raise ValueError("the size of the last dimension of confusion_matrix should be 4.")

    tp = confusion_matrix[..., 0]
    fp = confusion_matrix[..., 1]
    # tn = confusion_matrix[..., 2]
    fn = confusion_matrix[..., 3]

    nan_tensor = torch.tensor(float("nan"), device=confusion_matrix.device)
    numerator, denominator = (1.0 + beta**2) * tp, ((1.0 + beta**2) * tp + beta**2 * fn + fp)

    if isinstance(denominator, torch.Tensor):
        return torch.where(denominator != 0, numerator / denominator, nan_tensor)
    return numerator / denominator
