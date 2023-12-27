import torch

__all__ = [
    'LOGIT_EPS',
    'logit_inv',
]

LOGIT_EPS = 1e-3

def logit_inv(x: torch.Tensor) -> torch.Tensor:
    x = x.sigmoid()
    return ((x - LOGIT_EPS) / (1 - 2 * LOGIT_EPS)).clamp_(min=0, max=1)
