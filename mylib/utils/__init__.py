from collections.abc import Hashable
import os
from typing import Callable, Iterable, Union

import cytoolz
import einops
from einops import rearrange
from einops.layers.torch import Rearrange
import torch

from .enums import DataSplit, DataKey
from .index_tracker import IndexTracker

PathLike = Union[str, bytes, os.PathLike]

class ChannelFirst(Rearrange):
    def __init__(self):
        super().__init__('n ... c -> n c ...')

def channel_first(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, 'n ... c -> n c ...')

class ChannelLast(Rearrange):
    def __init__(self):
        super().__init__('n c ... -> n ... c')

def channel_last(x: torch.Tensor) -> torch.Tensor:
    return rearrange(x, 'n c ... -> n ... c')

def flatten(x: torch.Tensor) -> torch.Tensor:
    return einops.rearrange(x, 'n c ... -> n (...) c')

def partition_by_predicate(pred: Callable | Hashable, seq: Iterable):
    groups = cytoolz.groupby(pred, seq)
    return tuple(groups.get(k, []) for k in [False, True])

class SimpleReprMixin(object):
    """A mixin implementing a simple __repr__."""
    def __repr__(self):
        return "<{cls} @{id:x} {attrs}>".format(
            cls=self.__class__.__name__,
            id=id(self) & 0xFFFFFF,
            attrs=", ".join("{}={!r}".format(k, v) for k, v in self.__dict__.items()),
        )
