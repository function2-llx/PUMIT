from collections.abc import Sequence

import numpy as np

from mylib.conf import ClsExpConf
from mylib.utils import DataKey
from .base import ExpDataModuleBase, DataSeq

class ClsDataModule(ExpDataModuleBase):
    conf: ClsExpConf

    def classes_counting_data(self) -> DataSeq:
        return self.train_data()

    def count_classes(self) -> np.ndarray:
        cls_cnt = np.zeros(self.conf.num_cls_classes, dtype=np.float64)
        for x in self.classes_counting_data():
            cls_cnt[x[DataKey.CLS]] += 1
        return cls_cnt

    def default_cls_weights(self) -> list[float]:
        return (1. / self.count_classes()).tolist()
