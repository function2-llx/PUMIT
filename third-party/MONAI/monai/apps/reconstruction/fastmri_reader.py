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

import os
from collections.abc import Sequence

import numpy as np
from numpy import ndarray

from monai.config import PathLike
from monai.data.image_reader import ImageReader
from monai.data.utils import is_supported_format
from monai.utils import FastMRIKeys, optional_import, require_pkg

h5py, has_h5py = optional_import("h5py")


@require_pkg(pkg_name="h5py")
class FastMRIReader(ImageReader):
    """
    Load fastMRI files with '.h5' suffix. fastMRI files, when loaded with "h5py",
    are HDF5 dictionary-like datasets. The keys are:

    - kspace: contains the fully-sampled kspace
    - reconstruction_rss: contains the root sum of squares of ifft of kspace. This
        is the ground-truth image.

    It also has several attributes with the following keys:

    - acquisition (str): acquisition mode of the data (e.g., AXT2 denotes T2 brain MRI scans)
    - max (float): dynamic range of the data
    - norm (float): norm of the kspace
    - patient_id (str): the patient's id whose measurements were recorded
    """

    def verify_suffix(self, filename: Sequence[PathLike] | PathLike) -> bool:
        """
         Verify whether the specified file format is supported by h5py reader.

        Args:
             filename: file name
        """
        suffixes: Sequence[str] = [".h5"]
        return has_h5py and is_supported_format(filename, suffixes)

    def read(self, data: Sequence[PathLike] | PathLike) -> dict:  # type: ignore
        """
        Read data from specified h5 file.
        Note that the returned object is a dictionary.

        Args:
            data: file name to read.
        """
        if isinstance(data, (tuple, list)):
            data = data[0]

        with h5py.File(data, "r") as f:
            # extract everything from the ht5 file
            dat = dict(
                [(key, f[key][()]) for key in f]
                + [(key, f.attrs[key]) for key in f.attrs]
                + [(FastMRIKeys.FILENAME, os.path.basename(data))]  # type: ignore
            )
        f.close()

        return dat

    def get_data(self, dat: dict) -> tuple[ndarray, dict]:
        """
        Extract data array and metadata from the loaded data and return them.
        This function returns two objects, first is numpy array of image data, second is dict of metadata.

        Args:
            dat: a dictionary loaded from an h5 file
        """
        header = self._get_meta_dict(dat)
        data: ndarray = np.array(dat[FastMRIKeys.KSPACE])
        header[FastMRIKeys.MASK] = (
            np.expand_dims(np.array(dat[FastMRIKeys.MASK]), 0)[None, ..., None]
            if FastMRIKeys.MASK in dat.keys()
            else np.zeros(data.shape)
        )
        return data, header

    def _get_meta_dict(self, dat: dict) -> dict:
        """
        Get all the metadata of the loaded dict and return the meta dict.

        Args:
            dat: a dictionary object loaded from an h5 file.
        """
        return {k.value: dat[k.value] for k in FastMRIKeys if k.value in dat}
