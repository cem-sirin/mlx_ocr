import os.path as osp
from typing import Dict, List, Optional
from time import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data.ops import (
    decode_image,
    normalize_image,
    resize_image_test,
    to_mlx_array,
    make_divisible,
    resize_width_right_pad,
)

from .utils.logger import setup_logger
from .utils.ppocr.model_loader import load_det_pp_ocrv3, load_rec_pp_ocrv3
from .postprocess.db_postprocess import DBPostProcess
from .postprocess.rec_tokenizer import CTCLabelDecode


logger = setup_logger(__name__)
PROJECT_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
FONT_PATH = osp.join(PROJECT_ROOT, "mlx_ocr", "misc", "fonts", "roboto_condensed.ttf")


class TextDetector:
    def __init__(self, lang, select_model: str = None):
        self.model = load_det_pp_ocrv3(lang, select_model)
        self.model.eval()
        self.postprocess = DBPostProcess(min_size=4)

    def __repr__(self):
        return f"TextDetector({self.model.name})"

    def detect(self, image: Image.Image) -> Dict[str, np.ndarray]:
        img, scale = resize_image_test(image, (640, 896))
        img = decode_image(img, mode="BGR")
        img = normalize_image(img)
        img = to_mlx_array(img)

        infer_time = time()
        output = self.model(img)
        infer_time = time() - infer_time
        logger.info(f"det time: {infer_time:.4f}s")

        assert "shrink_maps" in output
        shrink_maps = output["shrink_maps"][:, :, :, 0]
        shrink_maps = np.array(shrink_maps)
        img_size = [(image.size[1], image.size[0])]
        ratios = [(scale, scale)]

        detections = self.postprocess(shrink_maps, img_size, ratios)
        return detections[0]

    def box_images(self, image: Image.Image, detection: Dict[str, np.ndarray] = None) -> List[np.ndarray]:
        """Detects and crops images"""
        if detection is None:
            detection = self.detect(image)
        assert "points" in detection, "detections should have 'points' key"

        cropped_images = []
        contours = detection["points"]

        # Sort contours by w/h ratio
        def w_h_ratio(contour):
            x, y, w, h = cv2.boundingRect(contour)
            return w / h

        # Sorting boxes by w/h ratio increases speed for batched images
        # contours = sorted(contours, key=w_h_ratio)

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_image = np.array(image.copy())[y : y + h, x : x + w]
            cropped_images.append(cropped_image)
        return cropped_images

    def draw_boxes(self, image: Image.Image, detection: Optional[Dict[str, np.ndarray]] = None) -> np.ndarray:
        if detection is None:
            detection = self.detect(image)[0]

        assert "points" in detection, "detection should have 'points' key"
        contours = detection["points"]
        contoured_image = cv2.drawContours(np.array(image.copy()), contours, -1, (0, 255, 0), 1)
        return contoured_image


class TextRecognizer:
    def __init__(self, lang, select_model: str = None):
        self.model = load_rec_pp_ocrv3(lang, select_model)
        self.model.eval()
        self.decoder = CTCLabelDecode(character_dict_path=self.model.vocab_path)

    def __repr__(self):
        return f"TextRecognizer({self.model.name})"

    def recognize(self, images: List[np.ndarray]) -> List[str]:
        B = 1
        H = 48
        N = len(images)
        results, infer_time = [], time()
        for i in range(0, N, B):
            batch_images = images[i : i + B].copy()
            max_wh_ratio = max([image.shape[1] / image.shape[0] for image in batch_images])
            max_width = make_divisible(48 * max_wh_ratio, 8)
            batch_images = [resize_width_right_pad(image, (max_width, H)) for image in batch_images]
            batch_images = np.stack(batch_images)
            # img = Image.fromarray(np.vstack(batch_images)) # For debugging
            batch_images = to_mlx_array(batch_images) / 255.0
            batch_images = batch_images[:, :, :, ::-1]
            batch_images = 2 * (batch_images - 0.5)

            out = self.model(batch_images)
            results.extend(self.decoder(out))

        infer_time = time() - infer_time
        total_chars = sum([len(text) for text, _ in results])
        time_per_char = infer_time / total_chars
        logger.info(f"rec time: {infer_time:.4f}s, time_per_char: {time_per_char:.4f}s")
        return results


class MLXOCR:
    def __init__(self, det_lang, rec_lang, det_select_model=None, rec_select_model=None):
        self.text_detector = TextDetector(det_lang, det_select_model)
        self.text_recognizer = TextRecognizer(rec_lang, rec_select_model)

    def __repr__(self):
        return f"MLXOCR(\n\ttext_detector={self.text_detector},\n\ttext_recognizer={self.text_recognizer}\n)"

    def detect(self, image: Image.Image) -> List[Dict[str, np.ndarray]]:
        return self.text_detector.detect(image)

    def recognize(self, images: List[np.ndarray]) -> List[str]:
        return self.text_recognizer.recognize(images)

    def __call__(self, image: Image.Image) -> List[str]:
        detection = self.text_detector.detect(image)
        cropped_images = self.text_detector.box_images(image, detection)

        texts = self.text_recognizer.recognize(cropped_images)

        assert len(detection["points"]) == len(texts)
        text_boxes = []
        for box, (text, rec_score) in zip(detection["points"], texts):
            text_boxes.append({"box": box, "text": text, "rec_score": rec_score})
        return text_boxes

    def visualize_ocr(self, image: Image.Image, text_boxes=None, font_size: int = 16) -> Image.Image:
        # text_boxes = self(image)
        if text_boxes is None:
            text_boxes = self(image)

        font = ImageFont.truetype(FONT_PATH, font_size)

        image_draw = ImageDraw.Draw(image)
        blank_image = Image.fromarray(np.zeros_like(np.array(image), dtype=np.uint8) + 255)
        blank_draw = ImageDraw.Draw(blank_image)

        for tb in text_boxes:
            box = tb["box"]
            text = tb["text"]
            x, y, w, h = cv2.boundingRect(box)
            image_draw.rectangle([x, y, x + w, y + h], outline="green", width=1)
            blank_draw.text((x, y), text, fill="black", font=font)

        # Put two images together
        canvas = np.hstack([np.array(image), np.array(blank_image)])
        return Image.fromarray(canvas)
