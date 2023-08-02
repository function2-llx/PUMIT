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

import torch

from monai.metrics.utils import do_metric_reduction, ignore_background
from monai.utils import MetricReduction, deprecated

from .metric import CumulativeIterationMetric


class MeanIoU(CumulativeIterationMetric):
    """
    Compute average Intersection over Union (IoU) score between two tensors.
    It supports both multi-classes and multi-labels tasks.
    Input `y_pred` is compared with ground truth `y`.
    `y_pred` is expected to have binarized predictions and `y` should be in one-hot format. You can use suitable transforms
    in ``monai.transforms.post`` first to achieve binarized values.
    The `include_background` parameter can be set to ``False`` to exclude
    the first category (channel index 0) which is by convention assumed to be background. If the non-background
    segmentations are small compared to the total image size they can get overwhelmed by the signal from the
    background.
    `y_pred` and `y` can be a list of channel-first Tensor (CHW[D]) or a batch-first Tensor (BCHW[D]).

    Example of the typical execution steps of this metric class follows :py:class:`monai.metrics.metric.Cumulative`.

    Args:
        include_background: whether to include IoU computation on the first channel of
            the predicted output. Defaults to ``True``.
        reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
            available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
            ``"mean_channel"``, ``"sum_channel"``}, default to ``"mean"``. if "none", will not do reduction.
        get_not_nans: whether to return the `not_nans` count, if True, aggregate() returns (metric, not_nans).
            Here `not_nans` count the number of not nans for the metric, thus its shape equals to the shape of the metric.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.

    """

    def __init__(
        self,
        include_background: bool = True,
        reduction: MetricReduction | str = MetricReduction.MEAN,
        get_not_nans: bool = False,
        ignore_empty: bool = True,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.reduction = reduction
        self.get_not_nans = get_not_nans
        self.ignore_empty = ignore_empty

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
                should be binarized.
            y: ground truth to compute mean IoU metric. It must be one-hot format and first dim is batch.
                The values should be binarized.

        Raises:
            ValueError: when `y_pred` has less than three dimensions.
        """
        dims = y_pred.ndimension()
        if dims < 3:
            raise ValueError(f"y_pred should have at least 3 dimensions (batch, channel, spatial), got {dims}.")
        # compute IoU (BxC) for each channel for each batch
        return compute_iou(
            y_pred=y_pred, y=y, include_background=self.include_background, ignore_empty=self.ignore_empty
        )

    def aggregate(
        self, reduction: MetricReduction | str | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Execute reduction logic for the output of `compute_meaniou`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, not_nans = do_metric_reduction(data, reduction or self.reduction)
        return (f, not_nans) if self.get_not_nans else f


def compute_iou(
    y_pred: torch.Tensor, y: torch.Tensor, include_background: bool = True, ignore_empty: bool = True
) -> torch.Tensor:
    """Computes Intersection over Union (IoU) score metric from a batch of predictions.

    Args:
        y_pred: input data to compute, typical segmentation model output.
            It must be one-hot format and first dim is batch, example shape: [16, 3, 32, 32]. The values
            should be binarized.
        y: ground truth to compute mean IoU metric. It must be one-hot format and first dim is batch.
            The values should be binarized.
        include_background: whether to include IoU computation on the first channel of
            the predicted output. Defaults to True.
        ignore_empty: whether to ignore empty ground truth cases during calculation.
            If `True`, NaN value will be set for empty ground truth cases.
            If `False`, 1 will be set if the predictions of empty ground truth cases are also empty.

    Returns:
        IoU scores per batch and per class, (shape [batch_size, num_classes]).

    Raises:
        ValueError: when `y_pred` and `y` have different shapes.

    """

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    y = y.float()
    y_pred = y_pred.float()

    if y.shape != y_pred.shape:
        raise ValueError(f"y_pred and y should have same shapes, got {y_pred.shape} and {y.shape}.")

    # reducing only spatial dimensions (not batch nor channels)
    n_len = len(y_pred.shape)
    reduce_axis = list(range(2, n_len))
    intersection = torch.sum(y * y_pred, dim=reduce_axis)

    y_o = torch.sum(y, reduce_axis)
    y_pred_o = torch.sum(y_pred, dim=reduce_axis)
    union = y_o + y_pred_o - intersection

    if ignore_empty:
        return torch.where(y_o > 0, (intersection) / union, torch.tensor(float("nan"), device=y_o.device))
    return torch.where(union > 0, (intersection) / union, torch.tensor(1.0, device=y_o.device))


@deprecated(since="1.0.0", msg_suffix="use `compute_iou` instead.")
def compute_meaniou(*args, **kwargs):
    return compute_iou(*args, **kwargs)
