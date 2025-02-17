from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import cv2

from shapely import Polygon
import pyclipper

from ..utils import polygon_to_array
from ...structures import TextBox
from ...utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MakeShrinkMapsConfig:
    min_text_size: int = 8
    shrink_ratio: float = 0.4
    shrink_step_size: float = 0.1


class MakeShrinkMaps:
    """Creates shrink versions of text polygons for text detection training. Used in
    detection networks like DB (Differentiable Binarization)."""

    def __init__(self, config: MakeShrinkMapsConfig = None):
        """
        The larger the shrink ratio, the smaller the distance, the larger the shrinked polygon.
        """
        if config is None:
            logger.info("No config provided for MakeShrinkMaps. Using default config.")
            config = MakeShrinkMapsConfig()
        self.config = config
        self.min_text_size = self.config.min_text_size
        self.shrink_ratio = self.config.shrink_ratio
        self.step_size = self.config.shrink_step_size

    def __call__(
        self, text_boxes: List[TextBox], img_size: Tuple[int, int]
    ) -> Tuple[Dict[str, np.ndarray], List[Polygon]]:

        h, w = img_size
        # self.validate_polygons(text_boxes, h, w)

        # Initialize ground truth maps and mask
        maps = np.zeros((h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)

        shrink_polys = []
        for tb in text_boxes:
            shrink_poly = self._draw_shrink_maps(tb, maps, mask)
            shrink_polys.append(shrink_poly)
        return {
            "shrink_maps": maps,
            "shrink_mask": mask,
        }, shrink_polys

    def _draw_shrink_maps(self, text_poly: TextBox, maps: np.ndarray, mask: np.ndarray):
        poly = text_poly.poly
        poly_arr = polygon_to_array(poly)

        # Calculate height and width of polygon
        # poly_h = max(poly_arr[:, 1]) - min(poly_arr[:, 1])
        # poly_w = max(poly_arr[:, 0]) - min(poly_arr[:, 0])

        # If the polygon is too small, fill the mask and return
        # TODO: the filtering logic for small polygons

        subject = [tuple(l) for l in poly_arr]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)

        # When ofsetting is unsuccessful, pyclipper returns an empty list, so
        # we initialize shrinked as an empty list
        shrinked = []

        for ratio in np.arange(self.shrink_ratio, 1 + self.step_size, self.step_size):
            dist = poly.area * (1 - np.power(ratio, 2)) / poly.length
            shrinked = padding.Execute(-dist)
            if len(shrinked) == 1:
                cv2.fillPoly(maps, [np.array(shrinked[0]).astype(np.int32)], 1)
                return Polygon(shrinked[0])

        if len(shrinked) == 0:
            # Theoretically, this should never happen, because we iterate
            # ratios till 1, that must give D=0 and thus, return the original
            # poly at worst case
            logger.critical(f"Shrinked polygon not found for {poly}")
            cv2.fillPoly(mask, [poly_arr.astype(np.int32)], 0)
