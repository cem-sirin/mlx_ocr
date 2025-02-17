from typing import Dict
import os.path as osp
import re
import pickle
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from ...models.modules.modules import ConvBNLayer

KEY_MAPPINGS = {"._mean": ".running_mean", "._variance": ".running_var", "._batch_norm": ".bn"}


def load_paddle_weights(weight_path: str) -> Dict[str, mx.array]:
    """Loads weights from a pickle file."""

    with open(weight_path, "rb") as f:
        weights = pickle.load(f)

    # Filter for only np.ndarray
    weights = {k: mx.array(v) for k, v in weights.items() if isinstance(v, np.ndarray)}
    return weights


def check_distillation_model(weights: Dict[str, mx.array], select_model: str = "Student") -> Dict[str, mx.array]:
    """Checks if the weights are from a distillation model and returns the keys."""
    top_hierachy_keys = set([k.split(".")[0] for k in weights.keys()])
    # check = dist_keys == top_hierachy_keys
    check = "Student" in top_hierachy_keys and "Teacher" in top_hierachy_keys

    if not check:
        return weights

    assert select_model in top_hierachy_keys, f"select_model must be one of {top_hierachy_keys}, got {select_model}."
    # Return only selected model keys
    if check:
        weights = {k: v for k, v in weights.items() if select_model in k}
        weights = {k.replace(f"{select_model}.", ""): v for k, v in weights.items()}

    return weights


def load_and_process_weights(
    model: nn.Module,
    weight_path: str,
    additional_key_mappings: Dict[str, str] = {},
    pattern_mappings: Dict[str, str] = {},
    select_model: str = "Student",
) -> Dict[str, mx.array]:
    """Loads weights from a pickle file, converts them to mx.array, and applies key mappings.

    Args:
        weight_path (str): Path to the pickle file containing the weights.
        additional_key_mappings (dict): Additional key mappings to apply.

    Returns:
        dict: The preprocessed weights dictionary.
    """
    weights = load_paddle_weights(weight_path)

    # Check if is a distillation model
    weights = check_distillation_model(weights)

    # Define key mappings
    key_mappings = KEY_MAPPINGS.copy()
    key_mappings.update(additional_key_mappings)

    # Apply key mappings
    for old, new in key_mappings.items():
        weights = {k.replace(old, new): v for k, v in weights.items()}

    for pattern, replacement in pattern_mappings.items():
        weights = {re.sub(pattern, replacement, k): v for k, v in weights.items()}

    weights = sanitize_weights_per_module(model, weights)
    weights = remove_lstm_weights(weights)
    return weights


def sanitize_weights_per_module(model: nn.Module, weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Adjusts the weights based on the model's modules.

    Args:
        model (nn.Module): The MLX model.
        weights (dict): The weights dictionary.

    Returns:
        dict: The adjusted weights dictionary.
    """

    for name, module in model.named_modules():
        if isinstance(module, ConvBNLayer):
            # If there is norm in the weight name, replace it with bn
            if any([f"{name}.norm" in k for k in weights.keys()]):
                weights = {k.replace(f"{name}.norm", f"{name}.bn"): v for k, v in weights.items()}

        # Below are common adjustments from Paddle to MLX
        if isinstance(module, nn.Linear):
            # (in_c, out_c) -> (out_c, in_c)
            weights[f"{name}.weight"] = weights[f"{name}.weight"].T

        if isinstance(module, nn.Conv2d):
            # (in_c, out_c, h, w) -> (in_c, h, w, out_c)
            weights[f"{name}.weight"] = weights[f"{name}.weight"].transpose((0, 2, 3, 1))

        if isinstance(module, nn.ConvTranspose2d):
            # (in_c, out_c, h, w) -> (out_c, h, w, in_c)
            weights[f"{name}.weight"] = weights[f"{name}.weight"].transpose((1, 2, 3, 0))
    return weights


def remove_lstm_weights(weights: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Paddle duplicates LSTM weights for each layer. This function removes them."""
    pattern = r"(.+\.weight|.+\.bias)_(ih|hh)_l(\d+)"
    for k in list(weights.keys()):
        if re.match(pattern, k):
            weights.pop(k)
    return weights
