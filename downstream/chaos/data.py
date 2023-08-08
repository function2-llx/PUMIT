from pathlib import Path
from PIL import Image
import numpy as np

from monai.config import PathLike

mr_mapping = {
    0: 0,
    63: 1,
    126: 2,
    189: 3,
    252: 4,
}

def read_label(label_dir: PathLike):
    label_dir = Path(label_dir)
    png_files = sorted(label_dir.glob('*.png'))
    img = np.stack(
        [np.array(Image.open(png_path)) for png_path in png_files],
        axis=-1,
    )
    img = np.vectorize(mr_mapping.get)(img)
    img = np.flip(np.rot90(img), 0)
    return img.astype(np.uint8)
