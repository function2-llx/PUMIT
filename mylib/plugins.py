from lightning import LightningModule
from lightning.pytorch.plugins import MixedPrecisionPlugin
from lightning.pytorch.trainer import call
from lightning.pytorch.utilities import GradClipAlgorithmType
from lightning_fabric.utilities.types import Steppable

__all__ = [
    'MOAGCMixedPrecisionPlugin',
]

# Manual Optimization with Automatic Gradient Clipping
# https://github.com/Lightning-AI/lightning/issues/18089
class MOAGCMixedPrecisionPlugin(MixedPrecisionPlugin):
    def _clip_gradients(
        self,
        model: LightningModule,
        optimizer: Steppable,
        clip_val: int | float | None = None,
        gradient_clip_algorithm: GradClipAlgorithmType | None = None,
    ) -> None:
        call._call_lightning_module_hook(
            model.trainer,
            "configure_gradient_clipping",
            optimizer,
            gradient_clip_val=clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
