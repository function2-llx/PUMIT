from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

class Backbone(nn.Module):
    def forward(self, img: torch.Tensor, *args, **kwargs) -> BackboneOutput:
        raise NotImplementedError

@dataclass
class BackboneOutput:
    cls_feature: torch.Tensor = field(default=None)
    feature_maps: list[torch.Tensor] = field(default_factory=list)

class Decoder(nn.Module):
    def forward(self, backbone_feature_maps: list[torch.Tensor], x_in: torch.Tensor) -> DecoderOutput:
        raise not NotImplementedError

@dataclass
class DecoderOutput:
    # low->high resolution
    feature_maps: list[torch.Tensor]
