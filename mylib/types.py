from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, TypeVar, TypedDict

from lightning.pytorch.utilities.types import LRSchedulerConfig as LRSchedulerConfigBase
from torch import nn

T = TypeVar('T')
tuple2_t: TypeAlias = tuple[T, T]
param2_t: TypeAlias = T | tuple2_t[T]
tuple3_t: TypeAlias = tuple[T, T, T]
param3_t: TypeAlias = T | tuple3_t[T]
spatial_param_t: TypeAlias = T | tuple2_t[T] | tuple3_t[T]
spatial_param_seq_t: TypeAlias = Sequence[param2_t[T]] | Sequence[param3_t[T]]

def check_tuple(obj, n: int, t: type):
    if not isinstance(obj, tuple):
        return False
    if len(obj) != n:
        return False
    return all(isinstance(x, t) for x in obj)


# set total=False to comfort IDE
class ParamGroup(TypedDict, total=False):
    params: list[nn.Parameter] | None
    names: list[str]
    lr_scale: float  # inserted by timm
    lr: float
    weight_decay: float

@dataclass
class LRSchedulerConfig(LRSchedulerConfigBase):
    scheduler: Any = None  # bypass jsonargparse check

@dataclass
class RangeTuple:
    min: float | int
    max: float | int

    def __iter__(self):
        yield self.min
        yield self.max

class NoWeightDecayParameter(nn.Parameter):
    pass

F = TypeVar('F', bound=type)
partial_t: TypeAlias = type[F] | tuple[type[F], dict]

def call_partial(partial: partial_t[F], *args, **kwargs):
    if not isinstance(partial, tuple):
        partial = (partial, {})
    return partial[0](*args, **partial[1], **kwargs)
