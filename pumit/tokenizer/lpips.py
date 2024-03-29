# a cleaner (at least for me) re-implementation of: https://github.com/CompVis/taming-transformers/blob/master/taming/modules/losses/lpips.py

from pathlib import Path

import einops
import torch
import torch.nn as nn
from torchvision import models as tvm

# convert [-1, 1] to ImageNet normalized
class InputNormLayer(nn.Module):
    mean: torch.Tensor
    std: torch.Tensor

    def __init__(self):
        super().__init__()
        from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        self.register_buffer(
            'mean',
            einops.rearrange(1 - 2 * torch.tensor(IMAGENET_DEFAULT_MEAN), 'c -> c 1 1'),
            persistent=False,
        )
        self.register_buffer(
            'std',
            einops.rearrange(2 * torch.tensor(IMAGENET_DEFAULT_STD), 'c -> c 1 1'),
            persistent=False,
        )

    def forward(self, x: torch.Tensor):
        return (x - self.mean) / self.std

def normalize_tensor(x: torch.Tensor, eps: float = 1e-10):
    return x / (x.norm(dim=1, keepdim=True) + eps)

class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self):
        super().__init__()
        self.input_norm_layer = InputNormLayer()
        vgg_features: nn.Sequential = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        self.features = nn.ModuleList([
            vgg_features[s]
            for s in [slice(4), slice(4, 9), slice(9, 16), slice(16, 23), slice(23, 30)]
        ])
        num_channels = [64, 128, 256, 512, 512]
        self.aggregate_layers = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(),
                nn.Conv2d(c, 1, kernel_size=1, bias=False),
            )
            for c in num_channels
        ])
        weights_path = Path(__file__).parent / 'vgg.pth'
        state_dict = torch.load(weights_path)
        for i in range(len(num_channels)):
            conv: nn.Conv2d = self.aggregate_layers[i][1]
            conv.load_state_dict({'weight': state_dict[f'lin{i}.model.1.weight']})
        self.eval()
        self.requires_grad_(False)
        print(f'load LPIPS from {weights_path}')

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        x, y = self.input_norm_layer(x), self.input_norm_layer(y)
        ret = None
        for feature, aggregate_layer in zip(self.features, self.aggregate_layers):
            x = feature(x)
            y = feature(y)
            diff = (normalize_tensor(x) - normalize_tensor(y)) ** 2
            aggregated = aggregate_layer(diff).mean()
            if ret is None:
                ret = aggregated
            else:
                ret += aggregated
        return 2 * ret
