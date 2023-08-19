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

import warnings
from typing import Any

import torch

from monai.metrics.utils import ignore_background
from monai.utils import MetricReduction

from .metric import Metric


class VarianceMetric(Metric):
    """
    Compute the Variance of a given T-repeats N-dimensional array/tensor. The primary usage is as an uncertainty based
    metric for Active Learning.

    It can return the spatial variance/uncertainty map based on user choice or a single scalar value via mean/sum of the
    variance for scoring purposes

    Args:
        include_background: Whether to include the background of the spatial image or channel 0 of the 1-D vector
        spatial_map: Boolean, if set to True, spatial map of variance will be returned corresponding to i/p image dimensions
        scalar_reduction: reduction type of the metric, either 'sum' or 'mean' can be used
        threshold: To avoid NaN's a threshold is used to replace zero's

    """

    def __init__(
        self,
        include_background: bool = True,
        spatial_map: bool = False,
        scalar_reduction: str = "sum",
        threshold: float = 0.0005,
    ) -> None:
        super().__init__()
        self.include_background = include_background
        self.spatial_map = spatial_map
        self.scalar_reduction = scalar_reduction
        self.threshold = threshold

    def __call__(self, y_pred: Any) -> Any:
        """
        Args:
            y_pred: Predicted segmentation, typically segmentation model output.
                It must be N-repeats, repeat-first tensor [N,C,H,W,D].

        Returns:
            Pytorch tensor of scalar value of variance as uncertainty or a spatial map of uncertainty

        """
        return compute_variance(
            y_pred=y_pred,
            include_background=self.include_background,
            spatial_map=self.spatial_map,
            scalar_reduction=self.scalar_reduction,
            threshold=self.threshold,
        )


class LabelQualityScore(Metric):
    """
    The assumption is that the DL model makes better predictions than the provided label quality, hence the difference
    can be treated as a label quality score

    It can be combined with variance/uncertainty for active learning frameworks to factor in the quality of label along
    with uncertainty
    Args:
        include_background: Whether to include the background of the spatial image or channel 0 of the 1-D vector
        spatial_map: Boolean, if set to True, spatial map of variance will be returned corresponding to i/p image
        dimensions
        scalar_reduction: reduction type of the metric, either 'sum' or 'mean' can be used

    """

    def __init__(self, include_background: bool = True, scalar_reduction: str = "sum") -> None:
        super().__init__()
        self.include_background = include_background
        self.scalar_reduction = scalar_reduction

    def __call__(self, y_pred: Any, y: Any) -> torch.Tensor | None:
        """
        Args:
            y_pred: Predicted segmentation, typically segmentation model output.
                It must be N-repeats, repeat-first tensor [N,C,H,W,D].

        Returns:
            Pytorch tensor of scalar value of variance as uncertainty or a spatial map of uncertainty

        """
        return label_quality_score(
            y_pred=y_pred, y=y, include_background=self.include_background, scalar_reduction=self.scalar_reduction
        )


def compute_variance(
    y_pred: torch.Tensor,
    include_background: bool = True,
    spatial_map: bool = False,
    scalar_reduction: str = "mean",
    threshold: float = 0.0005,
) -> torch.Tensor | None:
    """
    Args:
        y_pred: [N, C, H, W, D] or [N, C, H, W] or [N, C, H] where N is repeats, C is channels and H, W, D stand for
            Height, Width & Depth
        include_background: Whether to include the background of the spatial image or channel 0 of the 1-D vector
        spatial_map: Boolean, if set to True, spatial map of variance will be returned corresponding to i/p image
            dimensions
        scalar_reduction: reduction type of the metric, either 'sum' or 'mean' can be used
        threshold: To avoid NaN's a threshold is used to replace zero's
    Returns:
        A single scalar uncertainty/variance value or the spatial map of uncertainty/variance
    """

    # The background utils is only applicable here because instead of Batch-dimension we have repeats here
    y_pred = y_pred.float()

    if not include_background:
        y = y_pred
        # TODO If this utils is made to be optional for 'y' it would be nice
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    # Set any values below 0 to threshold
    y_pred[y_pred <= 0] = threshold

    n_len = len(y_pred.shape)

    if n_len < 4 and spatial_map:
        warnings.warn("Spatial map requires a 2D/3D image with N-repeats and C-channels")
        return None

    # Create new shape list
    # The N-repeats are multiplied by channels
    n_shape = y_pred.shape
    new_shape = [n_shape[0] * n_shape[1]]
    for each_dim_idx in range(2, n_len):
        new_shape.append(n_shape[each_dim_idx])

    y_reshaped = torch.reshape(y_pred, new_shape)
    variance = torch.var(y_reshaped, dim=0, unbiased=False)

    if spatial_map:
        return variance

    if scalar_reduction == MetricReduction.MEAN:
        return torch.mean(variance)
    if scalar_reduction == MetricReduction.SUM:
        return torch.sum(variance)
    raise ValueError(f"scalar_reduction={scalar_reduction} not supported.")


def label_quality_score(
    y_pred: torch.Tensor, y: torch.Tensor, include_background: bool = True, scalar_reduction: str = "mean"
) -> torch.Tensor | None:
    """
    The assumption is that the DL model makes better predictions than the provided label quality, hence the difference
    can be treated as a label quality score

    Args:
        y_pred: Input data of dimension [B, C, H, W, D] or [B, C, H, W] or [B, C, H] where B is Batch-size, C is
            channels and H, W, D stand for Height, Width & Depth
        y: Ground Truth of dimension [B, C, H, W, D] or [B, C, H, W] or [B, C, H] where B is Batch-size, C is channels
            and H, W, D stand for Height, Width & Depth
        include_background: Whether to include the background of the spatial image or channel 0 of the 1-D vector
        scalar_reduction: reduction type of the metric, either 'sum' or 'mean' can be used to retrieve a single scalar
            value, if set to 'none' a spatial map will be returned

    Returns:
        A single scalar absolute difference value as score with a reduction based on sum/mean or the spatial map of
        absolute difference
    """

    # The background utils is only applicable here because instead of Batch-dimension we have repeats here
    y_pred = y_pred.float()
    y = y.float()

    if not include_background:
        y_pred, y = ignore_background(y_pred=y_pred, y=y)

    n_len = len(y_pred.shape)
    if n_len < 4 and scalar_reduction == "none":
        warnings.warn("Reduction set to None, Spatial map return requires a 2D/3D image of B-Batchsize and C-channels")
        return None

    abs_diff_map = torch.abs(y_pred - y)

    if scalar_reduction == MetricReduction.NONE:
        return abs_diff_map

    if scalar_reduction == MetricReduction.MEAN:
        return torch.mean(abs_diff_map, dim=list(range(1, n_len)))
    if scalar_reduction == MetricReduction.SUM:
        return torch.sum(abs_diff_map, dim=list(range(1, n_len)))
    raise ValueError(f"scalar_reduction={scalar_reduction} not supported.")
