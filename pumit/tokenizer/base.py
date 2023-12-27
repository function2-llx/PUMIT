from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from jsonargparse import class_from_function
from lightning import Fabric
import torch
from torch import nn

from luolib.models import spadop
from luolib.models.utils import load_ckpt

from .utils import logit_inv
from .vq import VectorQuantizer, VectorQuantizerOutput

class VQVisualTokenizer(ABC, nn.Module):
    is_pretrained: bool = False

    def __init__(self, *args, quantize: VectorQuantizer, **kwargs):
        super().__init__(*args, **kwargs)
        self.quantize = quantize

    @property
    def codebook_size(self):
        return self.quantize.num_embeddings

    @abstractmethod
    def encode(self, x: spadop.SpatialTensor) -> spadop.SpatialTensor:
        pass

    @abstractmethod
    def decode(self, z_q: spadop.SpatialTensor) -> spadop.SpatialTensor:
        pass

    def autoencode(self, x_logit: spadop.SpatialTensor, fabric: Fabric | None = None) -> tuple[spadop.SpatialTensor, spadop.SpatialTensor, VectorQuantizerOutput]:
        z = self.encode(x_logit)
        vq_out: VectorQuantizerOutput = self.quantize(z, fabric)
        x_rec_logit = self.decode(vq_out.z_q)
        x_rec = logit_inv(x_rec_logit[:, :x_logit.shape[1]])
        return x_rec, x_rec_logit, vq_out

    def forward(self, x_logit: spadop.SpatialTensor, *, autoencode: bool = False, fabric: Fabric | None = None) -> tuple[spadop.SpatialTensor, VectorQuantizerOutput] | VectorQuantizerOutput:
        if autoencode:
            return self.autoencode(x_logit, fabric)
        z = self.encode(x_logit)
        vq_out: VectorQuantizerOutput = self.quantize(z)
        return vq_out

    def _load_from_state_dict(self, state_dict: dict[str, torch.Tensor], prefix: str, *args, **kwargs):
        if self.is_pretrained:
            # make pytorch happy
            state_dict.update(self.state_dict(prefix=prefix))
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def tokenize(self, x: spadop.SpatialTensor) -> VectorQuantizerOutput:
        # deprecated
        return self(x)

    def get_ref_param(self) -> nn.Parameter | None:
        return None

    @classmethod
    def from_pretrained(cls, model: VQVisualTokenizer, path: Path) -> VQVisualTokenizer:
        # not annotated as Self here for the recognition of jsonargparse
        load_ckpt(model, path)
        model.is_pretrained = True
        return model

# https://github.com/omni-us/jsonargparse/issues/309
from_pretrained = class_from_function(VQVisualTokenizer.from_pretrained)
