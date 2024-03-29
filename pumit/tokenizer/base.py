from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import einops
from jsonargparse import class_from_function
from lightning import Fabric
import torch
from torch import nn

from mylib.models import load_ckpt
from mylib.types import tuple3_t

from pumit import sac
from .quantize import VectorQuantizer, VectorQuantizerOutput

class VQTokenizer(ABC, nn.Module):
    is_pretrained: bool = False

    def __init__(self, quantize: VectorQuantizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize = quantize

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

    def forward(self, x: sac.SpatialTensor, fabric: Fabric | None = None) -> tuple[sac.SpatialTensor, VectorQuantizerOutput]:
        z = self.encode(x)
        vq_out: VectorQuantizerOutput = self.quantize(z, fabric)
        x_rec = self.decode(vq_out.z_q)
        return x_rec, vq_out

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if self.is_pretrained:
            # make pytorch happy
            state_dict.update(self.state_dict(prefix=prefix))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def tokenize(self, x: sac.SpatialTensor) -> VectorQuantizerOutput:
        z = self.encode(x)
        vq_out: VectorQuantizerOutput = self.quantize(z)
        return vq_out

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
