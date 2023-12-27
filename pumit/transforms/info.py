from typing import TypedDict

from luolib.types import tuple3_t

__all__ = [
    'TransInfo',
]

class TransInfo(TypedDict):
    aniso_d: int
    scale: tuple3_t[float]
    patch_size: tuple3_t[int]
