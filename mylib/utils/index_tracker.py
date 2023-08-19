from typing import Optional

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from numpy import typing as npt

class IndexTracker:
    def __init__(
        self,
        img: npt.NDArray,
        seg: Optional[npt.NDArray] = None,
        block: bool = True,
        title: str = "",
        zyx: bool = False,
        choose_max: bool = False,
    ):
        fig, ax = plt.subplots()
        fig: plt.Figure
        ax: plt.Axes
        ax.set_title('use scroll wheel to navigate images')
        self.ax = ax
        if zyx:
            img = img.transpose(2, 1, 0)
            if seg is not None:
                seg = seg.transpose(2, 1, 0)

        self.img = img
        self.seg = seg
        rows, cols, self.slices = img.shape
        self.ind = self.slices // 2

        self.ax_img = ax.imshow(np.rot90(self.img[:, :, self.ind]), cmap='gray')
        if self.seg is None:
            self.ax_seg = None
        else:
            seg = np.array(seg).astype(np.int64)
            num_classes = seg.max() + 1
            if choose_max:
                self.ind = (self.seg > 0).sum(axis=(0, 1)).argmax().item()
            self.ax_seg = ax.imshow(
                np.rot90(self.seg[:, :, self.ind]),
                vmax=num_classes,
                cmap=ListedColormap(['none', *matplotlib.colormaps['tab20'].colors[:num_classes - 1]]),
                alpha=0.5,
            )
            from matplotlib.colorbar import Colorbar
            cbar: matplotlib.colorbar.Colorbar = fig.colorbar(self.ax_seg)
            cbar.set_ticks(np.arange(num_classes))

        self.update()
        fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        plt.title(title)
        plt.show(block=block)

    def on_scroll(self, event):
        # print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = min(self.img.shape[-1] - 1, self.ind + 1)
        else:
            self.ind = max(0, self.ind - 1)
        self.update()

    def update(self):
        self.ax.set_ylabel('slice %s' % self.ind)
        self.ax_img.set_data(np.rot90(self.img[:, :, self.ind]))
        self.ax_img.axes.figure.canvas.draw()
        if self.ax_seg is not None:
            self.ax_seg.set_data(np.rot90(self.seg[:, :, self.ind]))
            self.ax_seg.axes.figure.canvas.draw()
