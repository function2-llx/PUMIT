from torch.optim import Optimizer

from mylib.conf import SchedulerConf
from mylib.utils import SimpleReprMixin

def create_scheduler(conf: SchedulerConf, optimizer: Optimizer):
    from timm.scheduler import create_scheduler_v2
    scheduler = create_scheduler_v2(optimizer, conf.name, **conf.kwargs)
    if type(scheduler).__repr__ == object.__repr__:
        type(scheduler).__repr__ = SimpleReprMixin.__repr__
    return scheduler
