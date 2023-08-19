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

from torch import Tensor

from monai.config.type_definitions import NdarrayOrTensor


def root_sum_of_squares_t(x: Tensor, spatial_dim: int) -> Tensor:
    """
    Compute the root sum of squares (rss) of the data (typically done for multi-coil MRI samples)

    Args:
        x: Input tensor
        spatial_dim: dimension along which rss is applied

    Returns:
        rss of x along spatial_dim

    Example:
        .. code-block:: python

            import numpy as np
            x = torch.ones([2,3])
            # the following line prints Tensor([1.41421356, 1.41421356, 1.41421356])
            print(rss(x,spatial_dim=0))
    """
    rss_x: Tensor = (x**2).sum(spatial_dim) ** 0.5
    return rss_x


def root_sum_of_squares(x: NdarrayOrTensor, spatial_dim: int) -> NdarrayOrTensor:
    """
    Compute the root sum of squares (rss) of the data (typically done for multi-coil MRI samples)

    Args:
        x: Input array/tensor
        spatial_dim: dimension along which rss is applied

    Returns:
        rss of x along spatial_dim

    Example:
        .. code-block:: python

            import numpy as np
            x = np.ones([2,3])
            # the following line prints array([1.41421356, 1.41421356, 1.41421356])
            print(rss(x,spatial_dim=0))
    """
    rss_x: NdarrayOrTensor = root_sum_of_squares_t(x, spatial_dim)  # type: ignore
    return rss_x
