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

from abc import ABC, abstractmethod

import monai
from monai.config import PathLike
from monai.data.utils import create_file_basename

__all__ = ["FolderLayoutBase", "FolderLayout", "default_name_formatter"]


def default_name_formatter(metadict: dict, saver: monai.transforms.Transform) -> dict:
    """Returns a kwargs dict for :py:meth:`FolderLayout.filename`,
    according to the input metadata and SaveImage transform."""
    subject = (
        metadict.get(monai.utils.ImageMetaKey.FILENAME_OR_OBJ, getattr(saver, "_data_index", 0))
        if metadict
        else getattr(saver, "_data_index", 0)
    )
    patch_index = metadict.get(monai.utils.ImageMetaKey.PATCH_INDEX, None) if metadict else None
    return {"subject": f"{subject}", "idx": patch_index}


class FolderLayoutBase(ABC):
    """
    Abstract base class to define a common interface for FolderLayout and derived classes
    Mainly, defines the ``filename(**kwargs) -> PathLike`` function, which must be defined
    by the deriving class.

    Example:

    .. code-block:: python

        from monai.data import FolderLayoutBase

        class MyFolderLayout(FolderLayoutBase):
            def __init__(
                self,
                basepath: Path,
                extension: str = "",
                makedirs: bool = False
            ):
                self.basepath = basepath
                if not extension:
                    self.extension = ""
                elif extension.startswith("."):
                    self.extension = extension:
                else:
                    self.extension = f".{extension}"
                self.makedirs = makedirs

            def filename(self, patient_no: int, image_name: str, **kwargs) -> Path:
                sub_path = self.basepath / patient_no
                if not sub_path.exists():
                    sub_path.mkdir(parents=True)

                file = image_name
                for k, v in kwargs.items():
                    file += f"_{k}-{v}"

                file +=  self.extension
                return sub_path / file

    """

    @abstractmethod
    def filename(self, **kwargs) -> PathLike:
        """
        Create a filename with path based on the input kwargs.
        Abstract method, implement your own.
        """
        raise NotImplementedError


class FolderLayout(FolderLayoutBase):
    """
    A utility class to create organized filenames within ``output_dir``. The
    ``filename`` method could be used to create a filename following the folder structure.

    Example:

    .. code-block:: python

        from monai.data import FolderLayout

        layout = FolderLayout(
            output_dir="/test_run_1/",
            postfix="seg",
            extension="nii",
            makedirs=False)
        layout.filename(subject="Sub-A", idx="00", modality="T1")
        # return value: "/test_run_1/Sub-A_seg_00_modality-T1.nii"

    The output filename is a string starting with a ``subject`` ID, and
    includes additional information about a customized index and image
    modality.  This utility class doesn't alter the underlying image data, but
    provides a convenient way to create filenames.
    """

    def __init__(
        self,
        output_dir: PathLike,
        postfix: str = "",
        extension: str = "",
        parent: bool = False,
        makedirs: bool = False,
        data_root_dir: PathLike = "",
    ):
        """
        Args:
            output_dir: output directory.
            postfix: a postfix string for output file name appended to ``subject``.
            extension: output file extension to be appended to the end of an output filename.
            parent: whether to add a level of parent folder to contain each image to the output filename.
            makedirs: whether to create the output parent directories if they do not exist.
            data_root_dir: an optional `PathLike` object to preserve the folder structure of the input `subject`.
                Please see :py:func:`monai.data.utils.create_file_basename` for more details.
        """
        self.output_dir = output_dir
        self.postfix = postfix
        self.ext = extension
        self.parent = parent
        self.makedirs = makedirs
        self.data_root_dir = data_root_dir

    def filename(self, subject: PathLike = "subject", idx=None, **kwargs) -> PathLike:
        """
        Create a filename based on the input ``subject`` and ``idx``.

        The output filename is formed as:

            ``output_dir/[subject/]subject[_postfix][_idx][_key-value][ext]``

        Args:
            subject: subject name, used as the primary id of the output filename.
                When a `PathLike` object is provided, the base filename will be used as the subject name,
                the extension name of `subject` will be ignored, in favor of ``extension``
                from this class's constructor.
            idx: additional index name of the image.
            kwargs: additional keyword arguments to be used to form the output filename.
                The key-value pairs will be appended to the output filename as ``f"_{k}-{v}"``.
        """
        full_name = create_file_basename(
            postfix=self.postfix,
            input_file_name=subject,
            folder_path=self.output_dir,
            data_root_dir=self.data_root_dir,
            separate_folder=self.parent,
            patch_index=idx,
            makedirs=self.makedirs,
        )
        for k, v in kwargs.items():
            full_name += f"_{k}-{v}"
        if self.ext is not None:
            ext = f"{self.ext}"
            full_name += f".{ext}" if ext and not ext.startswith(".") else f"{ext}"
        return full_name
