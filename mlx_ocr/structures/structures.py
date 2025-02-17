from dataclasses import dataclass
from shapely import Polygon


@dataclass
class TextBox:
    text: str
    poly: Polygon


@dataclass
class TextBoxDB(TextBox):
    """TextBox with dilated and shrunk polygons to be used for Differentiable Binarization"""

    shrink_poly: Polygon = None
    padded_poly: Polygon = None
