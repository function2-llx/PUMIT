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
from itertools import chain

import numpy as np
import torch

from monai.config import KeysCollection
from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset
from monai.data.meta_tensor import MetaTensor
from monai.data.utils import affine_to_spacing
from monai.transforms import concatenate
from monai.utils import PostFix, convert_data_type, convert_to_tensor

DEFAULT_POST_FIX = PostFix.meta()


class DatasetSummary:
    """
    This class provides a way to calculate a reasonable output voxel spacing according to
    the input dataset. The achieved values can used to resample the input in 3d segmentation tasks
    (like using as the `pixdim` parameter in `monai.transforms.Spacingd`).
    In addition, it also supports to compute the mean, std, min and max intensities of the input,
    and these statistics are helpful for image normalization
    (as parameters of `monai.transforms.ScaleIntensityRanged` and `monai.transforms.NormalizeIntensityd`).

    The algorithm for calculation refers to:
    `Automated Design of Deep Learning Methods for Biomedical Image Segmentation <https://arxiv.org/abs/1904.08128>`_.

    """

    def __init__(
        self,
        dataset: Dataset,
        image_key: str | None = "image",
        label_key: str | None = "label",
        meta_key: KeysCollection | None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        num_workers: int = 0,
        **kwargs,
    ):
        """
        Args:
            dataset: dataset from which to load the data.
            image_key: key name of images (default: ``image``).
            label_key: key name of labels (default: ``label``).
            meta_key: explicitly indicate the key of the corresponding metadata dictionary.
                for example, for data with key `image`, the metadata by default is in `image_meta_dict`.
                the metadata is a dictionary object which contains: filename, affine, original_shape, etc.
                if None, will try to construct meta_keys by `{image_key}_{meta_key_postfix}`.
                This is not required if `data[image_key]` is a MetaTensor.
            meta_key_postfix: use `{image_key}_{meta_key_postfix}` to fetch the metadata from dict,
                the metadata is a dictionary object (default: ``meta_dict``).
            num_workers: how many subprocesses to use for data loading.
                ``0`` means that the data will be loaded in the main process (default: ``0``).
            kwargs: other parameters (except `batch_size` and `num_workers`) for DataLoader,
                this class forces to use ``batch_size=1``.

        """

        self.data_loader = DataLoader(dataset=dataset, batch_size=1, num_workers=num_workers, **kwargs)

        self.image_key = image_key
        self.label_key = label_key
        self.meta_key = meta_key or f"{image_key}_{meta_key_postfix}"
        self.all_meta_data: list = []

    def collect_meta_data(self):
        """
        This function is used to collect the metadata for all images of the dataset.
        """

        for data in self.data_loader:
            if isinstance(data[self.image_key], MetaTensor):
                meta_dict = data[self.image_key].meta
            elif self.meta_key in data:
                meta_dict = data[self.meta_key]
            else:
                warnings.warn(f"To collect metadata for the dataset, `{self.meta_key}` or `data.meta` must exist.")
            self.all_meta_data.append(meta_dict)

    def get_target_spacing(self, spacing_key: str = "affine", anisotropic_threshold: int = 3, percentile: float = 10.0):
        """
        Calculate the target spacing according to all spacings.
        If the target spacing is very anisotropic,
        decrease the spacing value of the maximum axis according to percentile.
        The spacing is computed from `affine_to_spacing(data[spacing_key][0], 3)` if `data[spacing_key]` is a matrix,
        otherwise, the `data[spacing_key]` must be a vector of pixdim values.

        Args:
            spacing_key: key of the affine used to compute spacing in metadata (default: ``affine``).
            anisotropic_threshold: threshold to decide if the target spacing is anisotropic (default: ``3``).
            percentile: for anisotropic target spacing, use the percentile of all spacings of the anisotropic axis to
                replace that axis.

        """
        if len(self.all_meta_data) == 0:
            self.collect_meta_data()
        if spacing_key not in self.all_meta_data[0]:
            raise ValueError("The provided spacing_key is not in self.all_meta_data.")
        spacings = []
        for data in self.all_meta_data:
            spacing_vals = convert_to_tensor(data[spacing_key][0], track_meta=False, wrap_sequence=True)
            if spacing_vals.ndim == 1:  # vector
                spacings.append(spacing_vals[:3][None])
            elif spacing_vals.ndim == 2:  # matrix
                spacings.append(affine_to_spacing(spacing_vals, 3)[None])
            else:
                raise ValueError("data[spacing_key] must be a vector or a matrix.")
        all_spacings = concatenate(to_cat=spacings, axis=0)
        all_spacings, *_ = convert_data_type(data=all_spacings, output_type=np.ndarray, wrap_sequence=True)

        target_spacing = np.median(all_spacings, axis=0)
        if max(target_spacing) / min(target_spacing) >= anisotropic_threshold:
            largest_axis = np.argmax(target_spacing)
            target_spacing[largest_axis] = np.percentile(all_spacings[:, largest_axis], percentile)

        output = list(target_spacing)

        return tuple(output)

    def calculate_statistics(self, foreground_threshold: int = 0):
        """
        This function is used to calculate the maximum, minimum, mean and standard deviation of intensities of
        the input dataset.

        Args:
            foreground_threshold: the threshold to distinguish if a voxel belongs to foreground, this parameter
                is used to select the foreground of images for calculation. Normally, `label > 0` means the corresponding
                voxel belongs to foreground, thus if you need to calculate the statistics for whole images, you can set
                the threshold to ``-1`` (default: ``0``).

        """
        voxel_sum = torch.as_tensor(0.0)
        voxel_square_sum = torch.as_tensor(0.0)
        voxel_max, voxel_min = [], []
        voxel_ct = 0

        for data in self.data_loader:
            if self.image_key and self.label_key:
                image, label = data[self.image_key], data[self.label_key]
            else:
                image, label = data
            image, *_ = convert_data_type(data=image, output_type=torch.Tensor)
            label, *_ = convert_data_type(data=label, output_type=torch.Tensor)

            image_foreground = image[torch.where(label > foreground_threshold)]

            voxel_max.append(image_foreground.max().item())
            voxel_min.append(image_foreground.min().item())
            voxel_ct += len(image_foreground)
            voxel_sum += image_foreground.sum()
            voxel_square_sum += torch.square(image_foreground).sum()

        self.data_max, self.data_min = max(voxel_max), min(voxel_min)
        self.data_mean = (voxel_sum / voxel_ct).item()
        self.data_std = (torch.sqrt(voxel_square_sum / voxel_ct - self.data_mean**2)).item()

    def calculate_percentiles(
        self,
        foreground_threshold: int = 0,
        sampling_flag: bool = True,
        interval: int = 10,
        min_percentile: float = 0.5,
        max_percentile: float = 99.5,
    ):
        """
        This function is used to calculate the percentiles of intensities (and median) of the input dataset. To get
        the required values, all voxels need to be accumulated. To reduce the memory used, this function can be set
        to accumulate only a part of the voxels.

        Args:
            foreground_threshold: the threshold to distinguish if a voxel belongs to foreground, this parameter
                is used to select the foreground of images for calculation. Normally, `label > 0` means the corresponding
                voxel belongs to foreground, thus if you need to calculate the statistics for whole images, you can set
                the threshold to ``-1`` (default: ``0``).
            sampling_flag: whether to sample only a part of the voxels (default: ``True``).
            interval: the sampling interval for accumulating voxels (default: ``10``).
            min_percentile: minimal percentile (default: ``0.5``).
            max_percentile: maximal percentile (default: ``99.5``).

        """
        all_intensities = []
        for data in self.data_loader:
            if self.image_key and self.label_key:
                image, label = data[self.image_key], data[self.label_key]
            else:
                image, label = data
            image, *_ = convert_data_type(data=image, output_type=torch.Tensor)
            label, *_ = convert_data_type(data=label, output_type=torch.Tensor)

            intensities = image[torch.where(label > foreground_threshold)].tolist()
            if sampling_flag:
                intensities = intensities[::interval]
            all_intensities.append(intensities)

        all_intensities = list(chain(*all_intensities))
        self.data_min_percentile, self.data_max_percentile = np.percentile(
            all_intensities, [min_percentile, max_percentile]
        )
        self.data_median = np.median(all_intensities)
