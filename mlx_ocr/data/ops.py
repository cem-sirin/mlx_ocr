from typing import Literal, Tuple
from math import ceil

import cv2
import numpy as np
import mlx.core as mx
from PIL import Image

from ..utils.logger import setup_logger


logger = setup_logger(__name__)


def resize_image_test(img: Image.Image, resize: Tuple[int, int], divisible_by: int = 8, keep_ar: bool = True):
    # The image must be divisible by 8 or 16, not sure
    w, h = img.size
    w_new, h_new = resize
    if w_new % divisible_by != 0:
        add = divisible_by - (w_new % divisible_by)
        logger.info(f"Adding {add} pixels to width {w_new} so that it is divisible by {divisible_by}")
        w += add
    if h_new % divisible_by != 0:
        add = divisible_by - (h_new % divisible_by)
        logger.info(f"Adding {add} pixels to height {h_new} so that it is divisible by {divisible_by}")
        h += add

    if keep_ar:
        scale = min(w_new / w, h_new / h)
        if scale > 1:
            logger.info(
                f"The image is smaller than the target resize, scaling it up by ~{scale:.2f}. If extracting image from PDF, consider increasing the DPI."
            )

    # Scale the image
    img = img.resize((int(w * scale), int(h * scale)))

    # Paste the image on a new blank image
    new_img = Image.new("RGB", (w_new, h_new), (255, 255, 255))
    new_img.paste(img, (0, 0))
    assert new_img.size == (w_new, h_new)

    return new_img, scale


def decode_image(
    img: Image.Image,
    mode: Literal["RGB", "BGR", "GRAY"] = "BGR",
) -> np.ndarray:
    """Decode image to numpy array."""

    img = np.array(img.convert("RGB"))
    if mode == "BGR":
        img = img[:, :, ::-1]
    elif mode == "GRAY":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img


def resize_image(img: np.ndarray, size: tuple[int, int] = None) -> np.ndarray:
    """Resize image."""
    if size:
        return cv2.resize(img, size)
    else:
        return img


def normalize_image(
    img: np.ndarray,
    mean: tuple[float, float, float] = (0.406, 0.456, 0.485),
    std: tuple[float, float, float] = (0.225, 0.224, 0.229),
    scale: float = 1 / 255,
) -> np.ndarray:
    """Normalize image."""
    return (img * scale - mean) / std


def to_mlx_array(img: np.ndarray) -> mx.array:
    """Convert numpy array to mlx array."""
    img = mx.array(img)
    if len(img.shape) == 3:
        img = img.reshape(1, *img.shape)
    return img


def make_divisible(x: float, divisor: int) -> int:
    return int(ceil(x / divisor) * divisor)


def resize_width_right_pad(image: np.ndarray, resize: Tuple[int, int]) -> np.ndarray:
    oldH, oldW = image.shape[:2]
    newW, newH = resize
    scale = newH / oldH
    new_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    # Pad right
    pad = np.ones((newH, newW - new_image.shape[1], 3), dtype=np.uint8) * 0  # This is will be changed back to 255
    new_image = np.hstack([new_image, pad])
    return new_image
