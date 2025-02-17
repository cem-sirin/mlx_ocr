from dataclasses import dataclass
from typing import List, Tuple, Optional
from math import ceil, floor
from PIL import Image, ImageDraw, ImageFont

from pypdfium2 import PdfPage, PdfTextPage
from shapely import Polygon

from ..text_box_merger import TextBoxMerger, TextBoxMergerConfig
from ...structures import TextBox
from ...utils.visuals.visuals import draw_text_boxes


@dataclass
class PageHandlerConfig:
    page: PdfPage
    resize: Optional[Tuple[int, int]] = None
    keep_ratio: bool = True  # keep aspect ratio
    font_path: str = "fonts/roboto_condensed.ttf"
    font_size: int = 11
    verbose: bool = False
    text_box_merger_config: TextBoxMergerConfig = None


class PageHandler:
    def __init__(self, config: PageHandlerConfig):
        self.page = config.page
        self.keep_ratio = config.keep_ratio
        self.verbose = config.verbose

        self.text_box_merger = TextBoxMerger(config.text_box_merger_config)
        # Set page dimensions
        w, h = config.page.get_size()
        if config.resize is not None:
            self.width, self.height = config.resize
        else:
            self.width, self.height = ceil(w), ceil(h)

        # Calculate scale factors
        self.factor_w = self.width / w
        self.factor_h = self.height / h
        self.scale = min(self.factor_w, self.factor_h)
        if config.keep_ratio:
            self.factor_h = self.scale
            self.factor_w = self.scale

        if config.verbose:
            print(f"scale: {self.scale}, factor_w: {self.factor_w}, factor_h: {self.factor_h}")

        # Set font
        self.font = ImageFont.truetype(config.font_path, size=config.font_size)

    def _create_base_image(self) -> Image.Image:
        if self.keep_ratio:
            bitmap = self.page.render(scale=self.scale)
            image = bitmap.to_pil()

            # By default the padding is white (255, 255, 255)
            blank_image = self._create_blank_image()
            if self.verbose:
                print(f"image size: {image.size}")
                print(f"blank image size: {blank_image.size}")

            # Check if the image is indeed smaller than the blank image
            assert image.size[0] <= blank_image.size[0]
            assert image.size[1] <= blank_image.size[1]

            blank_image.paste(image, (0, 0))
            return blank_image
        else:
            bitmap = self.page.render(scale=max(self.factor_w, self.factor_h))
            image = bitmap.to_pil()
            return image.resize((self.width, self.height))

    def _create_blank_image(self) -> Image.Image:
        return Image.new("RGB", (self.width, self.height), (255, 255, 255))

    def _adjust_y_coordinates(self, rect: List[float]) -> List[float]:
        """Pydifium returns y axis from the bottom to the top, while we want it from top to bottom"""
        _, h = self.page.get_size()
        adjusted_rect = rect.copy()
        adjusted_rect[1] = h - rect[3]  # y0
        adjusted_rect[3] = h - rect[1]  # y1
        return adjusted_rect

    def _get_text_and_rect(self, textpage: PdfTextPage, index: int) -> Tuple[str, List[float]]:
        """Get the text and bounding box coordinates for a given text box.
        Returns:
            text (str): The text of the text box.
            rect (List[float]): The bounding box coordinates of the text box (x0, y0, x1, y1).
        """
        rect = list(textpage.get_rect(index))
        text = textpage.get_text_bounded(*rect)
        rect = self._adjust_y_coordinates(rect)
        return text, rect

    def _round_rect(self, rect: List[float]) -> List[float]:
        """Round the coordinates of a rectangle to the nearest integer.
        Args:
            rect (List[float]): The rectangle coordinates (x0, y0, x1, y1).
        Returns:
            List[float]: The rounded rectangle coordinates (x0, y0, x1, y1).
        """
        rect[0] = floor(rect[0] * self.factor_w)
        rect[1] = floor(rect[1] * self.factor_h)
        rect[2] = ceil(rect[2] * self.factor_w)
        rect[3] = ceil(rect[3] * self.factor_h)
        return rect

    def extract_text_boxes(self) -> List[TextBox]:
        """Procedure to extract text boxes from a page."""
        # Extract all the text-box pairs from text page
        textpage = self.page.get_textpage()
        text_boxes = []
        for index in range(textpage.count_rects()):
            # Get the text and bounding box coordinates
            text, rect = self._get_text_and_rect(textpage, index)
            # Round the coordinates of the rectangle
            rect = self._round_rect(rect)
            # Convert the rectangle to a shapely polygon
            poly = Polygon.from_bounds(*rect)
            # Add the text box to the list
            text_boxes.append(TextBox(text=text, poly=poly))

        # Post-process of merging boxes on the same line
        text_boxes = self.text_box_merger.merge_same_line_boxes(text_boxes)

        return text_boxes

    def visualize(self, text_boxes: Optional[List[TextBox]] = None) -> Image.Image:
        if text_boxes is None:
            text_boxes = self.extract_text_boxes()
        source_image = self._create_base_image()
        return draw_text_boxes(source_image, text_boxes, self.font)
