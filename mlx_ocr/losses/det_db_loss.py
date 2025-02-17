from typing import Dict

import mlx.core as mx
import mlx.nn as nn

from .det_basic_loss import BalanceLoss, MaskL1Loss, DiceLoss

PRED_KEYS = ["shrink_maps", "thresh_maps", "binary_maps"]
TARGET_KEYS = ["shrink_maps", "shrink_masks", "thresh_maps", "thresh_masks"]


class DBLoss(nn.Module):
    """
    Composit Loss for Differentiable Binarization.
    """

    def __init__(self, alpha: float = 5, beta: float = 10, ohem_ratio: float = 3, eps: float = 1e-6):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.shrink_loss_fn = DiceLoss(eps=eps)
        self.thresh_loss_fn = MaskL1Loss(eps=eps)
        self.binary_loss_fn = DiceLoss(eps=eps)

    def __call__(self, preds: Dict[str, mx.array], targets: Dict[str, mx.array]) -> Dict[str, mx.array]:
        assert all(key in preds.keys() for key in PRED_KEYS)
        assert all(key in targets.keys() for key in TARGET_KEYS)

        # Shrink loss
        loss_shrink = self.shrink_loss_fn(preds["shrink_maps"], targets["shrink_maps"], targets["shrink_masks"])
        loss_shrink *= self.alpha

        # Threshold loss
        loss_thresh = self.thresh_loss_fn(preds["thresh_maps"], targets["thresh_maps"], targets["thresh_masks"])
        loss_thresh *= self.beta

        # Binary loss
        loss_binary = self.binary_loss_fn(preds["binary_maps"], targets["shrink_maps"], targets["shrink_maps"])

        # Sum all losses
        loss = loss_shrink + loss_thresh + loss_binary

        return {
            "loss": loss,
            "loss_shrink": loss_shrink,
            "loss_thresh": loss_thresh,
            "loss_binary": loss_binary,
        }
