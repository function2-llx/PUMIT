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

from typing import Any

from monai.networks.nets import NetAdapter
from monai.utils import optional_import

models, _ = optional_import("torchvision.models")

__all__ = ["TorchVisionFCModel"]


class TorchVisionFCModel(NetAdapter):
    """
    Customize the fully connected layer of (pretrained) TorchVision model or replace it by convolutional layer.

    This class supports two primary use cases:

        - use ``pool=None`` to indicate no modification in the pooling layers. It should be used with ``fc_name``
          to locate the target FC layer to modify:
          In this case, the class will load a torchvision classification model,
          replace the last fully connected (FC) layer with a new FC layer with ``num_classes`` outputs,
          example input arguments: ``use_conv=False, pool=None, fc_name="heads.head"``.
          The ``heads.head`` specifies the target FC of the input model, could be found by ``model.named_modules()``,
          for example::

              from torchvision.models import vit_b_16
              print([name[0] for name in vit_b_16().named_modules()])

        - use ``pool=""`` or set it to a tuple of pooling parameters to indicate modifications of both
          the pooling and the FC layer. It should be used with ``node_name`` to locate the model feature outputs:
          In this case, the class will load a torchvision model, remove the existing pooling and FC layers, and

          - append an additional convolution layer:
            ``use_conv=True, pool="", node_name="permute"``
          - append an additional pooling and FC layers:
            ``use_conv=False, pool=("avg", {"kernel_size": 7, "stride": 1}), node_name="permute"``
          - append an additional pooling and convolution layers:
            ``use_conv=True, pool=("avg", {"kernel_size": 7, "stride": 1}), node_name="permute"``

          The ``permute`` in the example is the target feature extraction node of the input
          `model_name`, could be found by using the torchvision feature extraction utilities, for example::

              from torchvision.models.feature_extraction import get_graph_node_names
              from torchvision.models import swin_t
              print(get_graph_node_names(swin_t())[0])

    Args:
        model_name: name of any torchvision model with fully connected layer at the end.
            ``resnet18`` (default), ``resnet34``, ``resnet50``, ``resnet101``, ``resnet152``,
            ``resnext50_32x4d``, ``resnext101_32x8d``, ``wide_resnet50_2``, ``wide_resnet101_2``, ``inception_v3``.
            model details: https://pytorch.org/vision/stable/models.html.
        num_classes: number of classes for the last classification layer. Default to 1.
        dim: number of supported spatial dimensions in the specified model, depends on the model implementation.
            default to 2 as most Torchvision models are for 2D image processing.
        in_channels: number of the input channels of last layer. if None, get it from `in_features` of last layer.
        use_conv: whether to use convolutional layer to replace the last layer, default to False.
        pool: parameters for the pooling layer, when it's a tuple, the first item is name of the pooling layer,
            the second item is dictionary of the initialization args. If None, will not replace the `layers[-2]`.
            default to `("avg", {"kernel_size": 7, "stride": 1})`. ``""`` indicates not adding a pooling layer.
        bias: the bias value when replacing the last layer. if False, the layer will not learn an additive bias,
            default to True.
        pretrained: whether to use the imagenet pretrained weights. Default to False.
        fc_name: the corresponding layer attribute of the last fully connected layer. Defaults to ``"fc"``.
        node_name: the corresponding feature extractor node name of `model`. Defaults to "", not in use.
        weights: additional weights enum for the torchvision model.
        kwargs: additional parameters for the torchvision model.

    Example::

        import torch
        from torchvision.models.inception import Inception_V3_Weights

        from monai.networks.nets import TorchVisionFCModel

        model = TorchVisionFCModel(
            "inception_v3",
            num_classes=4,
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            use_conv=False,
            pool=None,
        )
        # model = TorchVisionFCModel("vit_b_16", num_classes=4, pool=None, in_channels=768, fc_name="heads")
        output = model.forward(torch.randn(2, 3, 299, 299))
        print(output.shape)  # torch.Size([2, 4])

    """

    def __init__(
        self,
        model_name: str = "resnet18",
        num_classes: int = 1,
        dim: int = 2,
        in_channels: int | None = None,
        use_conv: bool = False,
        pool: tuple[str, dict[str, Any]] | None = ("avg", {"kernel_size": 7, "stride": 1}),
        bias: bool = True,
        pretrained: bool = False,
        fc_name: str = "fc",
        node_name: str = "",
        weights=None,
        **kwargs,
    ):
        if weights is not None:
            model = getattr(models, model_name)(weights=weights, **kwargs)
        elif pretrained:
            model = getattr(models, model_name)(weights="DEFAULT", **kwargs)
        else:
            model = getattr(models, model_name)(weights=None, **kwargs)

        super().__init__(
            model=model,
            num_classes=num_classes,
            dim=dim,
            in_channels=in_channels,
            use_conv=use_conv,
            pool=pool,
            bias=bias,
            fc_name=fc_name,
            node_name=node_name,
        )
