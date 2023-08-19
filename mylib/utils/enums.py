from monai.utils import StrEnum

class DataSplit(StrEnum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

class DataKey(StrEnum):
    CASE = 'case'
    SPACING = 'spacing'
    IMG = 'img'
    CLS = 'cls'
    SEG = 'seg'
    MASK = 'mask'
    # SEG_ORIGIN = 'seg-origin'
    # CLINICAL = 'clinical'
