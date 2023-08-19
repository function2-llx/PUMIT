from collections.abc import Iterable

import torch
from torch import nn

from mylib.types import NoWeightDecayParameter, ParamGroup
from monai.config import PathLike

def load_ckpt(model: nn.Module, ckpt_or_path: dict | PathLike | None, state_dict_key: str | None = None, key_prefix: str = ''):
    if ckpt_or_path is None:
        return
    if isinstance(ckpt_or_path, dict):
        ckpt = ckpt_or_path
    else:
        ckpt: dict = torch.load(ckpt_or_path, map_location='cpu')
    if state_dict_key is None:
        if 'state_dict' in ckpt:
            state_dict_key = 'state_dict'
        elif 'model' in ckpt:
            state_dict_key = 'model'
    from timm.models import clean_state_dict
    state_dict = clean_state_dict(ckpt if state_dict_key is None else ckpt[state_dict_key])
    state_dict = {
        k[len(key_prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(key_prefix)
    }
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if not isinstance(ckpt_or_path, dict):
        print(f'Loaded {state_dict_key} from checkpoint {ckpt_or_path}')
    print('missing keys:', missing_keys)
    print('unexpected keys:', unexpected_keys)

def split_weight_decay_keys(module: nn.Module):
    # modify from https://github.com/karpathy/minGPT/blob/master/mingpt/model.py, `configure_optimizers`
    from torch.nn.modules.conv import _ConvNd
    whitelist_weight_modules = (
        nn.Linear,
        _ConvNd,
    )
    from torch.nn.modules.batchnorm import _BatchNorm
    from torch.nn.modules.instancenorm import _InstanceNorm
    blacklist_weight_modules = (
        nn.LayerNorm,
        _BatchNorm,
        _InstanceNorm,
        nn.GroupNorm,
        nn.Embedding,
    )
    decay = set()
    no_decay = set()
    for mn, m in module.named_modules():
        if hasattr(m, 'no_weight_decay'):
            no_decay |= {f'{mn}.{pn}' if mn else pn for pn in m.no_weight_decay()}

        for pn, p in m.named_parameters(prefix=mn, recurse=False):
            if not p.requires_grad:
                continue
            if isinstance(p, NoWeightDecayParameter):
                no_decay.add(pn)
            elif pn.endswith('.bias'):
                # all biases will not be decayed
                no_decay.add(pn)
            elif pn.endswith('.weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(pn)
            elif isinstance(m, nn.MultiheadAttention):
                if pn.endswith('_proj_weight'):
                    # projection weights of MultiheadAttention modules will be weight decayed
                    decay.add(pn)
                elif pn.endswith('_proj_bias'):
                    no_decay.add(pn)
            elif pn not in no_decay:
                assert pn.endswith('.weight') and isinstance(m, blacklist_weight_modules)
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(pn)

    inter_params = decay & no_decay
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    return decay, no_decay

def create_param_groups(named_parameters: Iterable[tuple[str, nn.Parameter]], decay_keys: set[str], no_decay_keys: set[str]) -> list[ParamGroup]:
    no_decay_group, decay_group = [], []
    for name, param in named_parameters:
        if name in no_decay_keys:
            no_decay_group.append(param)
        elif name in decay_keys:
            decay_group.append(param)
    return [
        {
            'params': no_decay_group,
            'weight_decay': 0.,
        },
        {
            'params': decay_group,
        }
    ]
