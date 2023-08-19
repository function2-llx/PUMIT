# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
from torch import nn

from monai.utils import optional_import

if optional_import("torch.nn.functional", name="mish")[1]:

    def monai_mish(x, inplace: bool = False):
        return torch.nn.functional.mish(x, inplace=inplace)

else:

    def monai_mish(x, inplace: bool = False):
        return x * torch.tanh(torch.nn.functional.softplus(x))


if optional_import("torch.nn.functional", name="silu")[1]:

    def monai_swish(x, inplace: bool = False):
        return torch.nn.functional.silu(x, inplace=inplace)

else:

    def monai_swish(x, inplace: bool = False):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Swish}(x) = x * \text{Sigmoid}(\alpha * x) ~~~~\text{for constant value}~ \alpha.

    Citation: Searching for Activation Functions, Ramachandran et al., 2017, https://arxiv.org/abs/1710.05941.


    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input


    Examples::

        >>> import torch
        >>> from monai.networks.layers.factories import Act
        >>> m = Act['swish']()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(self.alpha * input)


class SwishImplementation(torch.autograd.Function):
    r"""Memory efficient implementation for training
    Follows recommendation from:
    https://github.com/lukemelas/EfficientNet-PyTorch/issues/18#issuecomment-511677853

    Results in ~ 30% memory saving during training as compared to Swish()
    """

    @staticmethod
    def forward(ctx, input):
        result = input * torch.sigmoid(input)
        ctx.save_for_backward(input)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        sigmoid_input = torch.sigmoid(input)
        return grad_output * (sigmoid_input * (1 + input * (1 - sigmoid_input)))


class MemoryEfficientSwish(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Swish}(x) = x * \text{Sigmoid}(\alpha * x) ~~~~\text{for constant value}~ \alpha=1.

    Memory efficient implementation for training following recommendation from:
    https://github.com/lukemelas/EfficientNet-PyTorch/issues/18#issuecomment-511677853

    Results in ~ 30% memory saving during training as compared to Swish()

    Citation: Searching for Activation Functions, Ramachandran et al., 2017, https://arxiv.org/abs/1710.05941.

    From Pytorch 1.7.0+, the optimized version of `Swish` named `SiLU` is implemented,
    this class will utilize `torch.nn.functional.silu` to do the calculation if meets the version.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input


    Examples::

        >>> import torch
        >>> from monai.networks.layers.factories import Act
        >>> m = Act['memswish']()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        # inplace only works when using torch.nn.functional.silu
        self.inplace = inplace

    def forward(self, input: torch.Tensor):
        return monai_swish(input, self.inplace)


class Mish(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{Mish}(x) = x * tanh(\text{softplus}(x)).

    Citation: Mish: A Self Regularized Non-Monotonic Activation Function, Diganta Misra, 2019, https://arxiv.org/abs/1908.08681.

    From Pytorch 1.9.0+, the optimized version of `Mish` is implemented,
    this class will utilize `torch.nn.functional.mish` to do the calculation if meets the version.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input


    Examples::

        >>> import torch
        >>> from monai.networks.layers.factories import Act
        >>> m = Act['mish']()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def __init__(self, inplace: bool = False):
        super().__init__()
        # inplace only works when using torch.nn.functional.mish
        self.inplace = inplace

    def forward(self, input: torch.Tensor):
        return monai_mish(input, self.inplace)


class GEGLU(nn.Module):
    r"""Applies the element-wise function:

    .. math::
        \text{GEGLU}(x) = x_1 * \text{Sigmoid}(x_2)

    where :math:`x_1` and :math:`x_2` are split from the input tensor along the last dimension.

    Citation: GLU Variants Improve Transformer, Noam Shazeer, 2020, https://arxiv.org/abs/2002.05202.

    Shape:
        - Input: :math:`(N, *, 2 * D)`
        - Output: :math:`(N, *, D)`, where `*` means, any number of additional dimensions
    """

    def forward(self, input: torch.Tensor):
        x, gate = input.chunk(2, dim=-1)
        return x * nn.functional.gelu(gate)
