from collections.abc import Iterable, Sequence

from timm.models import group_parameters
from timm.optim import create_optimizer_v2
from timm.optim.optim_factory import _layer_map
import torch
from torch import nn

from mylib.conf import OptimizerConf
from mylib.types import ParamGroup

# modified from `timm.optim.optim_factory.param_groups_layer_decay`, add param_names in results
def param_groups_layer_decay(
    model: nn.Module,
    weight_decay: float = 0.05,
    no_weight_decay_list: Sequence[str] = (),
    layer_decay: float = .75,
) -> list[ParamGroup]:

    no_weight_decay_list = set(no_weight_decay_list)
    param_group_names = {}  # NOTE for debugging
    param_groups = {}

    if hasattr(model, 'group_matcher'):
        # FIXME interface needs more work
        layer_map = group_parameters(model, model.group_matcher(coarse=False), reverse=True)
    else:
        # fallback
        layer_map = _layer_map(model)
    num_layers = max(layer_map.values()) + 1
    layer_max = num_layers - 1
    layer_scales = list(layer_decay ** (layer_max - i) for i in range(num_layers))

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        if param.ndim == 1 or name in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        layer_id = layer_map.get(name, layer_max)
        group_name = "layer_%d_%s" % (layer_id, g_decay)

        if group_name not in param_groups:
            this_scale = layer_scales[layer_id]
            param_groups[group_name] = {
                "param_names": [],
                "params": [],
                "lr_scale": this_scale,
                "weight_decay": this_decay,
            }

        param_groups[group_name]["param_names"].append(name)
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())

def create_optimizer(conf: OptimizerConf, param_groups: list[ParamGroup] | Iterable[torch.Tensor]):
    # timm's typing is really not so great
    return create_optimizer_v2(
        param_groups,
        conf.name,
        conf.lr,
        conf.weight_decay,
        **conf.kwargs,
    )
