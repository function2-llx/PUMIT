import torch
from torchmetrics.classification import MulticlassStatScores
from torchmetrics.utilities.compute import _safe_divide  # noqa


class MulticlassBinaryAccuracy(MulticlassStatScores):
    is_differentiable = False
    higher_is_better = True
    full_state_update: bool = False

    def compute(self) -> torch.Tensor:
        """Computes accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        average = self.average
        multidim_average = self.multidim_average
        match average:
            case 'binary':
                return _safe_divide(tp + tn, tp + tn + fp + fn)
            case 'micro':
                # micro average for this metric makes no sense
                tp = tp.sum(dim=0 if multidim_average == "global" else 1)
                tn = tn.sum(dim=0 if multidim_average == "global" else 1)
                fp = fp.sum(dim=0 if multidim_average == "global" else 1)
                fn = fn.sum(dim=0 if multidim_average == "global" else 1)
                return _safe_divide(tp + fn, tp + tn + fp + fn)
            case _:
                score = _safe_divide(tp + tn, tp + tn + fp + fn)
                if average is None or average == "none":
                    return score
                if average == "weighted":
                    weights = tp + fn
                else:
                    weights = torch.ones_like(score)
                return _safe_divide(weights * score, weights.sum(-1, keepdim=True)).sum(-1)
