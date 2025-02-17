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
class MakeThreshMapsConfig:
    shrink_ratio: float = 0.4
    thresh_min: float = 0.3
    thresh_max: float = 0.7


class MakeThreshMaps:
    def __init__(self, config: MakeThreshMapsConfig = None):
        if config is None:
            logger.info("No config provided for MakeThreshMaps. Using default config.")
            config = MakeThreshMapsConfig()
        self.config = config

        self.shrink_ratio = self.config.shrink_ratio
        self.thresh_min = self.config.thresh_min
        self.thresh_max = self.config.thresh_max
        self._validate_config()

    def _validate_config(self) -> None:
        assert 0 <= self.shrink_ratio <= 1, "shrink_ratio must be between 0 and 1"
        assert 0 <= self.thresh_min < self.thresh_max <= 1, "thresh_min and thresh_max must be between 0 and 1"

    def __call__(
        self, text_polys: List[TextBox], img_size: Tuple[int, int]
    ) -> Tuple[Dict[str, np.ndarray], List[Polygon]]:

        maps = np.zeros(img_size, dtype=np.float32)
        mask = np.zeros(img_size, dtype=np.int8)  # 1s and 0s

        padded_polys = []
        for tp in text_polys:
            padded_poly = self._draw_thresh_maps(tp, maps, mask)
            padded_polys.append(padded_poly)

        # Scale the threshold maps between thresh_min and thresh_max
        maps = maps * (self.thresh_max - self.thresh_min) + self.thresh_min

        return {"thresh_maps": maps, "thresh_mask": mask}, padded_polys

    def _draw_thresh_maps(self, text_poly: TextBox, maps: np.ndarray, mask: np.ndarray):
        """Draws a border maps for text detection, creating a distance-based representation of text boundaries.

        Args:
            text_poly: TextBox object containing polygon information
            maps: Target numpy array where the border maps will be drawn
            mask: Binary mask array to mark the text region
        """

        poly = text_poly.poly
        poly_arr = polygon_to_array(poly)
        n_vertices = len(poly_arr)  # Number of vertices/edges in the polygon

        # Calculate padding distance based on polygon area and shrink ratio
        dist: float = poly.area * (1 - np.power(self.shrink_ratio, 2)) / poly.length
        assert dist > 0

        # Use Clipper library to create padded version of the polygon
        subject = [tuple(l) for l in poly_arr]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        padded_poly_arr = np.array(padding.Execute(dist)[0])

        # Fill the mask with the padded polygon
        cv2.fillPoly(mask, [padded_poly_arr.astype(np.int32)], 1)

        # Calculate bounding box coordinates of padded polygon
        x0, y0 = padded_poly_arr.min(axis=0)  # Top-left corner
        x1, y1 = padded_poly_arr.max(axis=0)  # Bottom-right corner

        # Calculate width and height of bounding box
        w = x1 - x0 + 1  # Width of bounding box
        h = y1 - y0 + 1  # Height of bounding box

        # Normalize polygon coordinates to bounding box of the padded polygon
        poly_arr[:, 0] = poly_arr[:, 0] - x0
        poly_arr[:, 1] = poly_arr[:, 1] - y0

        # Create coordinate grids for the bounding box
        xs = np.linspace(0, w - 1, num=w).reshape(1, w)
        ys = np.linspace(0, h - 1, num=h).reshape(h, 1)

        # Calculate distance maps for each edge of the polygon
        dist_maps = np.zeros((n_vertices, h, w), dtype=np.float32)
        for i in range(n_vertices):
            j = (i + 1) % n_vertices  # Next vertex (loops back to 0 at the end)
            abs_dist = self._distance(xs, ys, poly_arr[i], poly_arr[j])
            dist_maps[i] = np.clip(abs_dist / dist, 0, 1)

        # Take minimum distance across all edges
        dist_maps = dist_maps.min(axis=0)

        x0_valid = min(max(0, x0), maps.shape[1] - 1)
        x1_valid = min(max(0, x1), maps.shape[1] - 1)
        y0_valid = min(max(0, y0), maps.shape[0] - 1)
        y1_valid = min(max(0, y1), maps.shape[0] - 1)

        maps[y0_valid : y1_valid + 1, x0_valid : x1_valid + 1] = np.fmax(
            1
            - dist_maps[
                y0_valid - y0 : y1_valid - y1 + h,
                x0_valid - x0 : x1_valid - x1 + w,
            ],
            maps[y0_valid : y1_valid + 1, x0_valid : x1_valid + 1],
        )

        # Convert the padded polygon back to a shapely polygon
        padded_poly = Polygon(padded_poly_arr)
        return padded_poly

    def _distance(self, xs: np.ndarray, ys: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
        """Compute the perpendicular distance from each point (xs, ys) to a line segment (p1, p2)

        Args:
            xs: x-coordinates of points to check
            ys: y-coordinates of points to check
            p1: first endpoint of line segment
            p2: second endpoint of line segment
        """
        # Calculate squared distances from points to both endpoints
        square_dist_1 = np.square(xs - p1[0]) + np.square(ys - p1[1])
        square_dist_2 = np.square(xs - p2[0]) + np.square(ys - p2[1])
        # Calculate squared length of line segment
        square_dist = np.square(p1[0] - p2[0]) + np.square(p1[1] - p2[1])

        # Calculate cosine of angle between point-to-endpoint vectors
        cosin = (square_dist - square_dist_1 - square_dist_2) / (2 * np.sqrt(square_dist_1 * square_dist_2))
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_dist_1 * square_dist_2 * square_sin / square_dist)

        # If point projects outside segment, use distance to nearest endpoint
        result[cosin < 0] = np.sqrt(np.fmin(square_dist, square_dist_2))[cosin < 0]
        return result
