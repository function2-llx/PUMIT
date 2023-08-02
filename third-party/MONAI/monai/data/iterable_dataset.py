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

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any

from torch.utils.data import IterableDataset as _TorchIterableDataset
from torch.utils.data import get_worker_info

from monai.data.utils import convert_tables_to_dicts
from monai.transforms import apply_transform
from monai.transforms.transform import Randomizable
from monai.utils import optional_import

pd, _ = optional_import("pandas")


class IterableDataset(_TorchIterableDataset):
    """
    A generic dataset for iterable data source and an optional callable data transform
    when fetching a data sample. Inherit from PyTorch IterableDataset:
    https://pytorch.org/docs/stable/data.html?highlight=iterabledataset#torch.utils.data.IterableDataset.
    For example, typical input data can be web data stream which can support multi-process access.

    To accelerate the loading process, it can support multi-processing based on PyTorch DataLoader workers,
    every process executes transforms on part of every loaded data.
    Note that the order of output data may not match data source in multi-processing mode.
    And each worker process will have a different copy of the dataset object, need to guarantee
    process-safe from data source or DataLoader.

    """

    def __init__(self, data: Iterable[Any], transform: Callable | None = None) -> None:
        """
        Args:
            data: input data source to load and transform to generate dataset for model.
            transform: a callable data transform on input data.
        """
        self.data = data
        self.transform = transform
        self.source: Iterator[Any] | None = None

    def __iter__(self):
        info = get_worker_info()
        num_workers = info.num_workers if info is not None else 1
        id = info.id if info is not None else 0

        self.source = iter(self.data)
        for i, item in enumerate(self.source):
            if i % num_workers == id:
                if self.transform is not None:
                    item = apply_transform(self.transform, item)
                yield item


class ShuffleBuffer(Randomizable, IterableDataset):
    """
    Extend the IterableDataset with a buffer and randomly pop items.

    Args:
        data: input data source to load and transform to generate dataset for model.
        transform: a callable data transform on input data.
        buffer_size: size of the buffer to store items and randomly pop, default to 512.
        seed: random seed to initialize the random state of all workers, set `seed += 1` in
            every iter() call, refer to the PyTorch idea:
            https://github.com/pytorch/pytorch/blob/v1.10.0/torch/utils/data/distributed.py#L98.
        epochs: number of epochs to iterate over the dataset, default to 1, -1 means infinite epochs.

    Note:
        Both ``monai.data.DataLoader`` and ``torch.utils.data.DataLoader`` do not seed this class (as a subclass of
        ``IterableDataset``) at run time. ``persistent_workers=True`` flag (and pytorch>1.8) is therefore required
        for multiple epochs of loading when ``num_workers>0``. For example::

            import monai

            def run():
                dss = monai.data.ShuffleBuffer([1, 2, 3, 4], buffer_size=30, seed=42)

                dataloader = monai.data.DataLoader(
                    dss, batch_size=1, num_workers=2, persistent_workers=True)
                for epoch in range(3):
                    for item in dataloader:
                        print(f"epoch: {epoch} item: {item}.")

            if __name__ == '__main__':
                run()

    """

    def __init__(self, data, transform=None, buffer_size: int = 512, seed: int = 0, epochs: int = 1) -> None:
        super().__init__(data=data, transform=transform)
        self.size = buffer_size
        self.seed = seed
        self.epochs = epochs
        self._idx = 0

    def randomized_pop(self, buffer):
        """Return the item at a randomized location `self._idx` in `buffer`."""
        self.randomize(len(buffer))
        ret, buffer[self._idx] = buffer[self._idx], buffer[-1]
        buffer.pop()
        return ret

    def generate_item(self):
        """Fill a `buffer` list up to `self.size`, then generate randomly popped items."""
        buffer: list[Any] = []
        for item in iter(self.data):
            if len(buffer) >= self.size:
                yield self.randomized_pop(buffer)
            buffer.append(item)
        while buffer:
            yield self.randomized_pop(buffer)

    def __iter__(self):
        """
        Randomly pop buffered items from `self.data`.
        Multiple dataloader workers sharing this dataset will generate identical item sequences.
        """
        self.seed += 1
        super().set_random_state(seed=self.seed)  # make all workers in sync
        for _ in range(self.epochs) if self.epochs >= 0 else iter(int, 1):
            yield from IterableDataset(self.generate_item(), transform=self.transform)

    def randomize(self, size: int) -> None:
        self._idx = self.R.randint(size)


class CSVIterableDataset(IterableDataset):
    """
    Iterable dataset to load CSV files and generate dictionary data.
    It is particularly useful when data come from a stream, inherits from PyTorch IterableDataset:
    https://pytorch.org/docs/stable/data.html?highlight=iterabledataset#torch.utils.data.IterableDataset.

    It also can be helpful when loading extremely big CSV files that can't read into memory directly,
    just treat the big CSV file as stream input, call `reset()` of `CSVIterableDataset` for every epoch.
    Note that as a stream input, it can't get the length of dataset.

    To effectively shuffle the data in the big dataset, users can set a big buffer to continuously store
    the loaded data, then randomly pick data from the buffer for following tasks.

    To accelerate the loading process, it can support multi-processing based on PyTorch DataLoader workers,
    every process executes transforms on part of every loaded data.
    Note: the order of output data may not match data source in multi-processing mode.

    It can load data from multiple CSV files and join the tables with additional `kwargs` arg.
    Support to only load specific columns.
    And it can also group several loaded columns to generate a new column, for example,
    set `col_groups={"meta": ["meta_0", "meta_1", "meta_2"]}`, output can be::

        [
            {"image": "./image0.nii", "meta_0": 11, "meta_1": 12, "meta_2": 13, "meta": [11, 12, 13]},
            {"image": "./image1.nii", "meta_0": 21, "meta_1": 22, "meta_2": 23, "meta": [21, 22, 23]},
        ]

    Args:
        src: if provided the filename of CSV file, it can be a str, URL, path object or file-like object to load.
            also support to provide iter for stream input directly, will skip loading from filename.
            if provided a list of filenames or iters, it will join the tables.
        chunksize: rows of a chunk when loading iterable data from CSV files, default to 1000. more details:
            https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html.
        buffer_size: size of the buffer to store the loaded chunks, if None, set to `2 x chunksize`.
        col_names: names of the expected columns to load. if None, load all the columns.
        col_types: `type` and `default value` to convert the loaded columns, if None, use original data.
            it should be a dictionary, every item maps to an expected column, the `key` is the column
            name and the `value` is None or a dictionary to define the default value and data type.
            the supported keys in dictionary are: ["type", "default"]. for example::

                col_types = {
                    "subject_id": {"type": str},
                    "label": {"type": int, "default": 0},
                    "ehr_0": {"type": float, "default": 0.0},
                    "ehr_1": {"type": float, "default": 0.0},
                    "image": {"type": str, "default": None},
                }

        col_groups: args to group the loaded columns to generate a new column,
            it should be a dictionary, every item maps to a group, the `key` will
            be the new column name, the `value` is the names of columns to combine. for example:
            `col_groups={"ehr": [f"ehr_{i}" for i in range(10)], "meta": ["meta_1", "meta_2"]}`
        transform: transform to apply on the loaded items of a dictionary data.
        shuffle: whether to shuffle all the data in the buffer every time a new chunk loaded.
        seed: random seed to initialize the random state for all the workers if `shuffle` is True,
            set `seed += 1` in every iter() call, refer to the PyTorch idea:
            https://github.com/pytorch/pytorch/blob/v1.10.0/torch/utils/data/distributed.py#L98.
        kwargs_read_csv: dictionary args to pass to pandas `read_csv` function. Default to ``{"chunksize": chunksize}``.
        kwargs: additional arguments for `pandas.merge()` API to join tables.

    .. deprecated:: 0.8.0
        ``filename`` is deprecated, use ``src`` instead.

    """

    def __init__(
        self,
        src: str | Sequence[str] | Iterable | Sequence[Iterable],
        chunksize: int = 1000,
        buffer_size: int | None = None,
        col_names: Sequence[str] | None = None,
        col_types: dict[str, dict[str, Any] | None] | None = None,
        col_groups: dict[str, Sequence[str]] | None = None,
        transform: Callable | None = None,
        shuffle: bool = False,
        seed: int = 0,
        kwargs_read_csv: dict | None = None,
        **kwargs,
    ):
        self.src = src
        self.chunksize = chunksize
        self.buffer_size = 2 * chunksize if buffer_size is None else buffer_size
        self.col_names = col_names
        self.col_types = col_types
        self.col_groups = col_groups
        self.shuffle = shuffle
        self.seed = seed
        self.kwargs_read_csv = kwargs_read_csv or {"chunksize": chunksize}
        # in case treating deprecated arg `filename` as kwargs, remove it from `kwargs`
        kwargs.pop("filename", None)
        self.kwargs = kwargs

        self.iters: list[Iterable] = self.reset()
        super().__init__(data=None, transform=transform)  # type: ignore

    def reset(self, src: str | Sequence[str] | Iterable | Sequence[Iterable] | None = None):
        """
        Reset the pandas `TextFileReader` iterable object to read data. For more details, please check:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html?#iteration.

        Args:
            src: if not None and provided the filename of CSV file, it can be a str, URL, path object
                or file-like object to load. also support to provide iter for stream input directly,
                will skip loading from filename. if provided a list of filenames or iters, it will join the tables.
                default to `self.src`.

        """
        src = self.src if src is None else src
        srcs = (src,) if not isinstance(src, (tuple, list)) else src
        self.iters = []
        for i in srcs:
            if isinstance(i, str):
                self.iters.append(pd.read_csv(i, **self.kwargs_read_csv))
            elif isinstance(i, Iterable):
                self.iters.append(i)
            else:
                raise ValueError("`src` must be file path or iterable object.")
        return self.iters

    def close(self):
        """
        Close the pandas `TextFileReader` iterable objects.
        If the input src is file path, TextFileReader was created internally, need to close it.
        If the input src is iterable object, depends on users requirements whether to close it in this function.
        For more details, please check:
        https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html?#iteration.

        """
        for i in self.iters:
            i.close()  # type: ignore

    def _flattened(self):
        for chunks in zip(*self.iters):
            yield from convert_tables_to_dicts(
                dfs=chunks,
                col_names=self.col_names,
                col_types=self.col_types,
                col_groups=self.col_groups,
                **self.kwargs,
            )

    def __iter__(self):
        if self.shuffle:
            self.seed += 1
            buffer = ShuffleBuffer(
                data=self._flattened(), transform=self.transform, buffer_size=self.buffer_size, seed=self.seed
            )
            yield from buffer
        yield from IterableDataset(data=self._flattened(), transform=self.transform)
