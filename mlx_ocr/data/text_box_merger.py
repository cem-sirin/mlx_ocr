from dataclasses import dataclass
from typing import List, Tuple
from shapely import Polygon
from ..structures import TextBox


@dataclass
class TextBoxMergerConfig:
    max_x_dist: int = 10
    y_tol: int = 3
    verbose: bool = False


class TextBoxMerger:
    def __init__(self, config: TextBoxMergerConfig = None):
        self.config = config or TextBoxMergerConfig()
        self._validate_config()

        self.max_x_dist = self.config.max_x_dist
        self.y_tol = self.config.y_tol
        self.verbose = self.config.verbose

    def _validate_config(self) -> None:
        if self.config.max_x_dist < 0:
            raise ValueError("max_x_dist must be non-negative")
        if self.config.y_tol < 0:
            raise ValueError("y_tol must be non-negative")

    def _assess_same_line(self, poly1: Polygon, poly2: Polygon) -> Tuple[float, bool, bool]:
        x1, y1, x2, y2 = poly1.bounds
        x3, y3, x4, y4 = poly2.bounds

        x_dist = min(abs(x1 - x4), abs(x2 - x3))
        y_same = (abs(y1 - y3) <= self.y_tol) and (abs(y2 - y4) <= self.y_tol)

        poly1_is_left = x1 < x3

        return x_dist, y_same, poly1_is_left

    def merge_same_line_boxes(self, text_boxes: List[TextBox]) -> List[TextBox]:
        total_merged = 0

        unfinished = True
        while unfinished:
            found_one = False
            for i, tb1 in enumerate(text_boxes):
                for tb2 in text_boxes[i + 1 :]:
                    if tb1 == tb2:
                        continue
                    x_dist, y_same, poly1_is_left = self._assess_same_line(tb1.poly, tb2.poly)
                    if x_dist <= self.max_x_dist and y_same:
                        if poly1_is_left:
                            new_text = tb1.text + tb2.text
                        else:
                            new_text = tb2.text + tb1.text

                        new_poly = tb1.poly.union(tb2.poly).simplify(0.0)
                        new_poly = Polygon.from_bounds(*new_poly.bounds)

                        new_tb = TextBox(text=new_text, poly=new_poly)
                        found_one = True

                        # Remove the old ones and add the new one
                        text_boxes.remove(tb1)
                        text_boxes.remove(tb2)
                        text_boxes.append(new_tb)
                        total_merged += 1
                        break

                if found_one:
                    break
            if not found_one:
                unfinished = False

        if self.verbose:
            print(f"Merged {total_merged} boxes")

        return text_boxes
