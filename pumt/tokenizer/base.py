from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import einops
from jsonargparse import class_from_function
import torch
from torch import nn

from luolib.models import load_ckpt
from luolib.types import tuple3_t

from pumt import sac
from .quantize import VectorQuantizer, VectorQuantizerOutput

class VQTokenizer(ABC, nn.Module):
    is_pretrained: bool = False

    def __init__(self, vq_kwargs: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize = VectorQuantizer(**vq_kwargs)

    @property
    @abstractmethod
    def stride(self) -> tuple3_t[int]:
        # watch out: https://github.com/pytorch/pytorch/issues/49726
        pass

    @property
    def codebook_size(self):
        return self.quantize.num_embeddings

    @abstractmethod
    def encode(self, x: sac.SpatialTensor) -> sac.SpatialTensor:
        pass

    @abstractmethod
    def decode(self, z_q: sac.SpatialTensor) -> sac.SpatialTensor:
        pass

    def forward(self, x: sac.SpatialTensor) -> tuple[sac.SpatialTensor, VectorQuantizerOutput]:
        z = self.encode(x)
        quant_out: VectorQuantizerOutput = self.quantize(z)
        x_rec = self.decode(quant_out.z_q)
        return x_rec, quant_out

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if self.is_pretrained:
            # make pytorch happy
            state_dict.update(self.state_dict(prefix=prefix))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def tokenize(self, x: sac.SpatialTensor) -> VectorQuantizerOutput:
        z = self.encode(x)
        quant_out: VectorQuantizerOutput = self.quantize(z)
        return quant_out

    def get_ref_param(self) -> nn.Parameter | None:
        return None

    @classmethod
    def from_pretrained(cls, model: VQTokenizer, path: Path) -> VQTokenizer:
        # not annotated as Self here for the recognition of jsonargparse
        load_ckpt(model, path)
        model.is_pretrained = True
        return model

# https://github.com/omni-us/jsonargparse/issues/309
from_pretrained = class_from_function(VQTokenizer.from_pretrained)
