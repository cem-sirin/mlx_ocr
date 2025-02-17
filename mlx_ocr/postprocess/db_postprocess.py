from typing import Dict, Literal, List, Tuple

import numpy as np
import cv2
import pyclipper
from shapely.geometry import Polygon

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DBPostProcess:
    """The post process for Differentiable Binarization (DB)."""

    def __init__(
        self,
        thresh: float = 0.2,
        box_thresh: float = 0.6,
        max_candidates: int = 1000,
        unclip_ratio: float = 2.0,
        min_size: int = 3,
        use_dilation: bool = False,
        score_mode: Literal["slow", "fast"] = "fast",
        box_type: Literal["quad", "poly"] = "quad",
    ):
        assert score_mode in ["slow", "fast"], f"Score mode must be in [slow, fast] but got: {score_mode}"
        assert box_type in ["quad", "poly"], f"Box type must be in [quad, poly] but got: {box_type}"

        if box_type == "poly":
            logger.warning(
                "The 'poly' box type has not been tested by the authors, use it at your own risk. Consider increasing unclip_ratio :)"
            )

        self.thresh = thresh
        self.box_thresh = box_thresh
        self.max_candidates = max_candidates
        self.unclip_ratio = unclip_ratio
        self.min_size = min_size
        self.score_mode = score_mode
        self.box_type = box_type

        self.dilation_kernel = None if not use_dilation else np.array([[1, 1], [1, 1]])

    def polygons_from_bitmap(
        self, shrink_maps: np.ndarray, bitmap: np.ndarray, dst_size: Tuple[int, int], ratio: Tuple[float, float]
    ):
        contours, _ = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        boxes, scores = [], []
        for contour in contours[: self.max_candidates]:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue

            score = self.box_score_fast(shrink_maps, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            if points.shape[0] > 2:
                box = self.unclip(points)
                if len(box) > 1:
                    continue
            else:
                continue
            box = np.array(box).reshape(-1, 2)
            if len(box) == 0:
                continue

            _, sside = self.get_mini_boxes(box.reshape((-1, 1, 2)))
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box = self._rescale_box(box, dst_size, ratio)
            boxes.append(box)
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(
        self, shrink_maps: np.ndarray, bitmap: np.ndarray, dst_size: Tuple[int, int], ratio: Tuple[float, float]
    ):
        """
        Args:
            shrink_maps: model prediction with shape (H, W),
            bitmap: single map with shape (H, W), whose values are binarized as {0, 1}
            dst_size: the size of the original image
            ratio: the ratio of the original image and the processed image that was passed to the model

        Returns:
            boxes: the detected boxes
            scores: the scores of the detected boxes
        """
        outs = cv2.findContours(bitmap.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        assert len(outs) == 2, f"Expected 2 outputs from cv2.findContours, got {len(outs)}"
        contours = outs[0]

        num_contours = min(len(contours), self.max_candidates)
        boxes, scores = [], []
        for index in range(num_contours):
            contour: np.ndarray = contours[index]  # (n_points, 1, 2)
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue

            # Calculates the average prediction score of the points in the box
            if self.score_mode == "fast":
                score = self.box_score_fast(shrink_maps, points.reshape(-1, 2))
            else:  # self.score_mode == "slow":
                score = self.box_score_slow(shrink_maps, contour)
            if self.box_thresh > score:
                continue

            # Unclip the box (i.e., expand the box)
            box = self.unclip(points)
            if len(box) > 1:  # Check if the unclip box has more than 1 polygon
                continue

            box = np.array(box).reshape(-1, 1, 2)  # (n_points, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue

            box = np.array(box)
            box = self._rescale_box(box, dst_size, ratio)

            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores

    def _rescale_box(self, box: np.ndarray, dst_size: Tuple[int, int], ratio: Tuple[float, float]) -> np.ndarray:
        # Rescale the box to the original image size, below was a mistake in PaddleOCR that caused images
        # incorect scaling for images that were resized while preserving the aspect ratio
        dst_height, dst_width = dst_size
        ratio_h, ratio_w = ratio
        box[:, 0] = np.clip(np.round(box[:, 0] / ratio_w), 0, dst_width)
        box[:, 1] = np.clip(np.round(box[:, 1] / ratio_h), 0, dst_height)
        return box

    def unclip(self, points: np.ndarray) -> List[List[List[int]]]:
        poly = Polygon(points)
        dist = poly.area * self.unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(points, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = offset.Execute(dist)  # (n_polygons, n_points, 2)
        return expanded

    def get_mini_boxes(self, contour: np.ndarray) -> Tuple[np.ndarray, int]:
        """Returns the bounding box and the minimum side length of the contour."""
        bbox = cv2.minAreaRect(contour)  # ((center_x, center_y), (w, h), angle)
        points = sorted(cv2.boxPoints(bbox), key=lambda x: (x[0], x[1]))
        box = np.array([points[0], points[2], points[3], points[1]])
        return box, round(min(bbox[1]))

    def box_score_fast(self, bitmap: np.ndarray, _box: np.ndarray) -> float:
        """
        box_score_fast: use bbox mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def box_score_slow(self, bitmap: np.ndarray, contour: np.ndarray) -> float:
        """
        box_score_slow: use polyon mean score as the mean score
        """
        h, w = bitmap.shape[:2]
        contour = contour.copy()
        contour = np.reshape(contour, (-1, 2))

        xmin = np.clip(np.min(contour[:, 0]), 0, w - 1)
        xmax = np.clip(np.max(contour[:, 0]), 0, w - 1)
        ymin = np.clip(np.min(contour[:, 1]), 0, h - 1)
        ymax = np.clip(np.max(contour[:, 1]), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)

        contour[:, 0] = contour[:, 0] - xmin
        contour[:, 1] = contour[:, 1] - ymin

        cv2.fillPoly(mask, contour.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def __call__(
        self, shrink_maps: np.ndarray, img_size: List[Tuple[int, int]], ratios: List[Tuple[float, float]]
    ) -> List[Dict[str, np.ndarray]]:
        """
        Args:
            shrink_maps (np.ndarray): The shrink maps with shape (bsz, h, w).
            img_size (List[Tuple[int, int]]): The original image size.
            ratios (List[Tuple[float, float]]): The ratio of the original image and the input image.
        Returns:
            boxes_batch (List[Dict[str, np.ndarray]]): The detected boxes.
        """
        assert shrink_maps.ndim == 3, f"Expected 3D input (bsz, h, w), got {shrink_maps.ndim}D input"
        segmentation = shrink_maps > self.thresh
        boxes_batch = []
        for i in range(shrink_maps.shape[0]):

            mask = segmentation[i]
            if self.dilation_kernel is not None:
                mask = cv2.dilate(mask.astype(np.uint8), self.dilation_kernel)

            if self.box_type == "poly":
                boxes, _ = self.polygons_from_bitmap(shrink_maps[i], mask, img_size[i], ratios[i])
            else:  # self.box_type == "quad":
                boxes, _ = self.boxes_from_bitmap(shrink_maps[i], mask, img_size[i], ratios[i])

            boxes_batch.append({"points": boxes})
        return boxes_batch
