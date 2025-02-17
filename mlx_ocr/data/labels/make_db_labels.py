"""This is a work in progress to replace the pyclipper library. The goal is 
- to use shapely for the buffer operation,
- to use shapely functions to calculate the distance map.
"""

from typing import List, Tuple, Literal
from math import ceil, floor
import cv2
import numpy as np

from shapely.geometry import Polygon, JOIN_STYLE

from ..utils import polygon_to_array
from ...structures import TextBox, TextBoxDB


class MakeDBLabels:
    """Creates shrink and padded versions of text polygons for text detection training. Used in
    detection networks like DB (Differentiable Binarization)."""

    def __init__(
        self,
        min_text_size: int = 8,
        shrink_ratio: float = 0.4,
        resolution: int = 16,
        join_style: Literal["round", "miter", "bevel"] = "bevel",
        buffer_args: dict = {"mitre_limit": 2.0},
        thresh_min: float = 0.3,
        thresh_max: float = 0.7,
    ):
        """The larger the shrink ratio, the smaller the distance is."""
        self.min_text_size = min_text_size
        self.shrink_ratio = shrink_ratio
        self.resolution = resolution
        self.join_style = join_style
        self.buffer_args = buffer_args
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def buffer(self, poly: Polygon, d: float) -> Polygon:
        return poly.buffer(
            d,
            resolution=self.resolution,
            join_style=getattr(JOIN_STYLE, self.join_style),
            **self.buffer_args,
        )

    def __call__(self, text_boxes: List[TextBox], img_size: Tuple[int, int]):
        """
        Args:
            text_boxes: List of TextBoxes
            img_size: Tuple of image height and width. It should correspond to the polygons in text_boxes.
        """

        # Initialize ground truth maps and mask
        shrink_maps = np.zeros(img_size, dtype=np.int8)
        shrink_mask = np.ones(img_size, dtype=np.int8)
        thresh_maps = np.zeros(img_size, dtype=np.float32)
        thresh_mask = np.zeros(img_size, dtype=np.int8)

        text_box_dbs = []
        for tb in text_boxes:
            poly = tb.poly
            D = poly.area * (1 - np.power(self.shrink_ratio, 2)) / poly.length

            padded_poly = self.buffer(poly, D)
            shrink_poly = self.buffer(poly, -D)
            # Append the new TextBoxDB to the list
            text_box_dbs.append(TextBoxDB(tb.text, poly, shrink_poly, padded_poly))

            # Fill in the shrink maps and the mask
            cv2.fillPoly(shrink_maps, np.array([shrink_poly.exterior.coords], dtype=np.int32), 1)
            cv2.fillPoly(thresh_mask, np.array([padded_poly.exterior.coords], dtype=np.int32), 1)
            # print("thresh_mask")
            # print(thresh_mask[:15, :15])
            ### Threshold map calculation ###
            poly_arr = polygon_to_array(poly)
            padded_poly_arr = polygon_to_array(padded_poly)

            # print(padded_poly_arr)
            N = len(poly_arr)  # Number of vertices/edges
            x0, y0 = padded_poly_arr.min(axis=0)  # Top-left corner
            x1, y1 = padded_poly_arr.max(axis=0)  # Bottom-right corner
            x0, y0, x1, y1 = floor(x0), floor(y0), ceil(x1), ceil(y1)

            # The width and height of the bounding box
            w = x1 - x0 + 1
            h = y1 - y0 + 1

            # Normalize polygon coordinates to bounding box of the padded polygon
            poly_arr[:, 0] = poly_arr[:, 0] - x0
            poly_arr[:, 1] = poly_arr[:, 1] - y0

            # print("bounds", padded_poly.exterior.bounds)
            print(f"w = {w}, h = {h}, x0 = {x0}, y0 = {y0}, x1 = {x1}, y1 = {y1}")
            # Create coordinate grids for the bounding box
            xs = np.linspace(0, w - 1, num=w).reshape(1, w)
            ys = np.linspace(0, h - 1, num=h).reshape(h, 1)

            # Calculate the distance for each pixel to the nearest polygon edge
            dist_map = np.zeros((N, h, w), dtype=np.float32)
            for i in range(N):
                j = (i + 1) % N  # Next vertex (loops back to 0 at the end)
                abs_dist = self._distance(xs, ys, poly_arr[i], poly_arr[j])
                dist_map[i] = np.clip(abs_dist / D, 0, 1)

            # Take the minimum distance across all edges
            dist_map = dist_map.min(axis=0)

            x0_valid = min(max(0, x0), thresh_maps.shape[1] - 1)
            x1_valid = min(max(0, x1), thresh_maps.shape[1] - 1)
            y0_valid = min(max(0, y0), thresh_maps.shape[0] - 1)
            y1_valid = min(max(0, y1), thresh_maps.shape[0] - 1)

            thresh_maps[y0_valid : y1_valid + 1, x0_valid : x1_valid + 1] = np.fmax(
                1
                - dist_map[
                    y0_valid - y0 : y1_valid - y1 + h,
                    x0_valid - x0 : x1_valid - x1 + w,
                ],
                thresh_maps[y0_valid : y1_valid + 1, x0_valid : x1_valid + 1],
            )

        thresh_maps = np.clip(thresh_maps, self.thresh_min, self.thresh_max)
        return {
            "shrink_maps": shrink_maps,
            "shrink_mask": shrink_mask,
            "thresh_maps": thresh_maps,
            "thresh_mask": thresh_mask,
            "text_box_dbs": text_box_dbs,
        }

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


if __name__ == "__main__":
    np.set_printoptions(linewidth=200, precision=2)
    from shapely.geometry import Polygon
    import matplotlib.pyplot as plt

    text_boxes = [
        TextBox(
            text="hello",
            poly=Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
        ),
        TextBox(
            text="world",
            poly=Polygon([(20, 20), (30, 20), (30, 40), (20, 40)]),
        ),
        TextBox(
            text="!",
            poly=Polygon([(50, 50), (60, 50), (60, 60), (50, 60)]),
        ),
    ]

    img_size = (100, 100)
    make_shrink_border_map = MakeDBLabels()
    out = make_shrink_border_map(text_boxes, img_size)

    # Plot the shrink maps and mask side by side
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].imshow(out["shrink_maps"], vmin=0, vmax=1)
    axs[0, 1].imshow(out["shrink_mask"], vmin=0, vmax=1)
    axs[1, 0].imshow(out["thresh_maps"], vmin=0, vmax=1)
    axs[1, 1].imshow(out["thresh_mask"], vmin=0, vmax=1)
    plt.show()
