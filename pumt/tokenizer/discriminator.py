import torch
from torch import nn

from pumt.conv import AdaptiveConvDownsample, InflatableConv3d, InflatableInputConv3d

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int, num_downsample_layers: int = 3, base_channels: int = 64):
        super().__init__()

        layer_channels = [
            base_channels << min(i, 3)
            for i in range(num_downsample_layers + 1)
        ]
        self.main = nn.ModuleList()
        self.main.extend([
            AdaptiveConvDownsample(in_channels, layer_channels[0], (3, 4, 4), InflatableInputConv3d),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        for i in range(1, num_downsample_layers + 1):  # gradually increase the number of filters
            self.main.extend([
                AdaptiveConvDownsample(layer_channels[i - 1], layer_channels[i], (3, 4, 4), bias=False) if i < num_downsample_layers
                else InflatableConv3d(layer_channels[i - 1], layer_channels[i], kernel_size=(3, 4, 4), stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(layer_channels[i], affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        self.main.append(InflatableConv3d(layer_channels[-1], 1, kernel_size=(3, 4, 4), stride=1, padding=1))

    def forward(self, x: torch.Tensor, spacing: torch.Tensor):
        for module in self.main:
            if isinstance(module, AdaptiveConvDownsample):
                x, spacing, _ = module(x, spacing)
            else:
                x = module(x)
        return x
