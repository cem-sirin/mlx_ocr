import matplotlib.pyplot as plt
import numpy as np
import io

from typing import List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont

from ...structures import TextBox, TextBoxDB


def create_blank_image(size: Tuple[int, int]) -> Image.Image:
    blank_image = Image.new("RGB", size, (255, 255, 255))
    return blank_image


def draw_text_boxes(source_image, text_boxes: List[TextBox], font=None) -> Image.Image:

    w, h = source_image.size
    text_image = create_blank_image((w, h))

    source_draw = ImageDraw.Draw(source_image)
    text_draw = ImageDraw.Draw(text_image)

    for tb in text_boxes:
        # Draw rectangle on source image
        source_draw.polygon(tb.poly.exterior.coords, outline="red", width=1)

        # Write text on blank image
        xy = [int(x) for x in tb.poly.exterior.coords[0]]
        xy = tb.poly.bounds[:2]
        text_draw.text(xy, tb.text, fill="black", font=font)

    canvas = create_blank_image((2 * w, h))
    canvas.paste(source_image, (0, 0))
    canvas.paste(text_image, (w, 0))
    return canvas


def draw_db_text_boxes(source_image, text_boxes: List[TextBoxDB], font=None) -> Image.Image:
    w, h = source_image.size
    text_image = create_blank_image((w, h))

    source_draw = ImageDraw.Draw(source_image)
    text_draw = ImageDraw.Draw(text_image)

    for tb in text_boxes:
        # Draw rectangle on source image
        for i in range(len(tb.poly.exterior.coords)):
            j = (i + 1) % len(tb.poly.exterior.coords)
            x, y = tb.poly.exterior.coords[i]
            x2, y2 = tb.poly.exterior.coords[j]
            source_draw.line([x, y, x2, y2], fill=(255, 0, 0))
        for i in range(len(tb.padded_poly.exterior.coords)):
            j = (i + 1) % len(tb.padded_poly.exterior.coords)
            x, y = tb.padded_poly.exterior.coords[i]
            x2, y2 = tb.padded_poly.exterior.coords[j]
            source_draw.line([x, y, x2, y2], fill=(0, 255, 0))

        for i in range(len(tb.shrink_poly.exterior.coords)):
            j = (i + 1) % len(tb.shrink_poly.exterior.coords)
            x, y = tb.shrink_poly.exterior.coords[i]
            x2, y2 = tb.shrink_poly.exterior.coords[j]
            source_draw.line([x, y, x2, y2], fill=(0, 0, 255))

        # Write text on blank image
        xy = [int(x) for x in tb.poly.exterior.coords[0]]
        xy = tb.poly.bounds[:2]
        text_draw.text(xy, tb.text, fill="black", font=font)

    canvas = create_blank_image((2 * w, h))
    canvas.paste(source_image, (0, 0))
    canvas.paste(text_image, (w, 0))

    return canvas


def draw_db_labels(labels: dict) -> Image.Image:
    assert "thresh_maps" in labels and "thresh_mask" in labels and "shrink_maps" in labels and "shrink_mask" in labels
    cat = np.concatenate
    im = cat(
        [
            cat([labels["thresh_maps"], labels["thresh_mask"]], axis=1),
            cat([labels["shrink_maps"], labels["shrink_mask"]], axis=1),
        ],
        axis=0,
    )
    plt.imshow(im, vmin=0, vmax=1)
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    return Image.open(img_buf)
