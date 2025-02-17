import os
import mlx.nn as nn
from mlx.utils import tree_flatten

from .logger import setup_logger
from .ppocr.sanitize import load_and_process_weights

logger = setup_logger(__name__)


def load_weights(model: nn.Module, weights: dict):
    """Loads the preprocessed weights into the MLX model.

    Args:
        model (nn.Module): The MLX model.
        weights (dict): The preprocessed weights dictionary.
    """
    sd = dict(tree_flatten(model.parameters()))

    missing_keys = set(sd.keys()) - set(weights.keys())
    extra_keys = set(weights.keys()) - set(sd.keys())

    if not missing_keys:
        weights = [(k, v) for k, v in weights.items() if k in sd]
        model.load_weights(weights)
        # logger.info("Weights loaded successfully.")
    else:
        print("Cannot load weights. Missing weights found.")

    return model
