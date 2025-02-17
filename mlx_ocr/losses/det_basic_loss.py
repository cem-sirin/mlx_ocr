"""I am not sure whether to make losses functions or classes"""

from typing import Literal, Optional

import mlx.core as mx
import mlx.nn as nn

Reduce = Literal["none", "mean"]  # I discluded sum
Reduction = Literal["none", "mean", "sum"]


def _check_shapes(preds: mx.array, targets: mx.array, mask: Optional[mx.array] = None):
    assert preds.shape == targets.shape
    if mask is not None:
        assert mask.shape == preds.shape


class BCELoss(nn.Module):
    def __init__(self, with_logits: bool = True, reduction: Reduce = "mean"):
        super(BCELoss, self).__init__()
        self.with_logits = with_logits
        self.reduction = reduction

    def __call__(self, preds: mx.array, targets: mx.array, mask: Optional[mx.array] = None):
        _check_shapes(preds, targets, mask)
        # Passing mask as the weights arg to binary_cross_entropy
        return nn.losses.binary_cross_entropy(preds, targets, mask, self.with_logits, self.reduction)


class MaskL1Loss(nn.Module):
    def __init__(self, reduction: Reduce = "mean", eps: float = 1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def __call__(self, preds: mx.array, targets: mx.array, mask: Optional[mx.array] = None):
        _check_shapes(preds, targets, mask)
        if mask is None:
            mask = mx.ones_like(preds)

        loss = (preds - targets).abs()

        if mask is not None:
            loss = mx.where(mask, loss, mx.zeros_like(loss))

        if self.reduction == "mean":
            return loss.sum() / (mask.sum() + self.eps)
        else:  # no reduction
            return loss


class DiceLoss(nn.Module):
    """
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    """

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def __call__(self, preds: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
        """Computes the dice loss between preds and targets.

        Args:
            preds (array): The predicted values.
            targets (array): Binary target values.
            mask (array): The mask for the loss computation.

        Returns:
            array: The computed dice loss.
        """

        # TODO: Add weights support
        assert (
            preds.shape == targets.shape == mask.shape
        ), f"Predictions, targets, and mask shapes {preds.shape}, {targets.shape}, and {mask.shape} do not match."

        intersection = (preds * targets * mask).sum()
        union = (preds * mask).sum() + (targets * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        return loss


# In contrary to PaddleOCR, DiceLoss can not be balanced. Intersection over union can not
# be calculated per pixel, thus the positive and negative balancing is senseless.
BalanceLossType = Literal["MaskL1Loss", "BCELoss"]
LOSS_DICT = {"MaskL1Loss": MaskL1Loss, "BCELoss": BCELoss}


class BalanceLoss(nn.Module):
    def __init__(
        self,
        # balance_loss: bool = True, Removed to avoid confusion
        main_loss_type: BalanceLossType = "MaskL1Loss",
        negative_ratio: int = 3,
        return_origin: bool = False,
        eps: float = 1e-6,
    ):
        """
        The BalanceLoss for Differentiable Binarization text detection.

        In text detection context, positive samples are where text exists and negative vice versa.
        Since negative samples are more likely to be noise, we want to balance the loss by penalizing
        the negative samples. In PaddleOCR configs, you may see Balanced DiceLoss, which does not
        make sense. I assume they wanted to use DiceLoss for the shrink maps, but did not want to mess
        change to codebase

        Args:
            main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss',
                'Euclidean','BCELoss', 'MaskL1Loss'], default is  'DiceLoss'.
            negative_ratio (int|float): float, default is 3.
            return_origin (bool): whether return unbalanced loss or not, default is False.
            eps (float): default is 1e-6.
        """
        super(BalanceLoss, self).__init__()
        self.main_loss_type = main_loss_type
        self.loss_fn = LOSS_DICT[main_loss_type](reduction="none")
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps

    def __call__(self, preds: mx.array, targets: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        _check_shapes(preds, targets, mask)
        if mask is None:
            mask = mx.ones_like(preds)

        loss = self.loss_fn(preds, targets, mask=mask)
        assert loss.shape == preds.shape, "Loss must not be reduced else plz don't use BalanceLoss"

        positive = targets * mask
        negative = (1 - targets) * mask

        # We cap the negative count to be no larger than negative_ratio * positive_count
        positive_count = int(positive.sum())
        negative_count = int(min(negative.sum(), positive_count * self.negative_ratio))

        positive_loss = positive * loss
        negative_loss = negative * loss

        negative_loss = mx.topk(negative_loss.flatten(), k=negative_count)

        return (positive_loss.sum() + negative_loss.sum()) / (positive_count + negative_count + self.eps)


if __name__ == "__main__":
    import mlx.core as mx
    import mlx.nn as nn

    mx.random.seed(0)
    preds = mx.random.uniform(shape=(2, 1, 10, 10))
    # targets = mx.random.randint(0, 2, shape=(2, 1, 10, 10))
    targets = mx.random.bernoulli(p=0.1, shape=(2, 1, 10, 10))

    mask = mx.random.uniform(shape=(2, 1, 10, 10)) > 0.1

    mask_l1_loss = MaskL1Loss()
    loss = mask_l1_loss(preds, targets, mask)
    print(loss)

    dice_loss = DiceLoss()
    loss = dice_loss(preds, targets, mask)
    print(loss)

    balanced_loss = BalanceLoss()

    balanced_loss(preds, targets, mask)
