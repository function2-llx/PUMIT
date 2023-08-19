import importlib
from collections.abc import Mapping

from mylib.conf import parse_node
from mylib.utils import PathLike

def get_obj_from_str(string: str, reload: bool = False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate(conf: Mapping | PathLike):
    if isinstance(conf, PathLike):
        conf = parse_node(conf)
    target_cls = get_obj_from_str(conf.target)
    return target_cls(**conf.get('kwargs', dict()))
