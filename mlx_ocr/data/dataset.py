from dataclasses import dataclass
from typing import List
from shapely import Polygon
import numpy as np

from .labels.make_thresh_maps import MakeThreshMaps, MakeThreshMapsConfig
from .labels.make_shrink_maps import MakeShrinkMaps, MakeShrinkMapsConfig
from .labels.make_db_labels import MakeDBLabels
from ..structures import TextBox, TextBoxDB


@dataclass
class DatasetConfig:
    make_border_map_config: MakeThreshMapsConfig = None
    make_segment_map_config: MakeShrinkMapsConfig = None
    return_text_boxes: bool = False


class Dataset:
    def __init__(self, data, augs, config: DatasetConfig = None):
        self.config = config or DatasetConfig()
        self.data = data
        self.augs = augs
        self.make_border_map = MakeThreshMaps(self.config.make_border_map_config)
        self.make_segment_map = MakeShrinkMaps(self.config.make_segment_map_config)
        self.make_db_labels = MakeDBLabels()

        self.return_text_boxes = self.config.return_text_boxes
        if augs:
            augs.add_targets(
                {
                    "shrink_maps": "mask",
                    "shrink_mask": "mask",
                    "thresh_maps": "mask",
                    "thresh_mask": "mask",
                }
            )

    def __getitem__(self, index):
        text_boxes = self.data[index]["text_boxes"]
        image = self.data[index]["image"].convert("RGB").copy()
        image = np.array(image)

        out = {"image": image}
        img_size = (image.shape[0], image.shape[1])

        out.update(self.make_db_labels(text_boxes, img_size))
        if self.augs:
            out.update(
                self.augs(
                    image=out["image"],
                    shrink_maps=out["shrink_maps"],
                    shrink_mask=out["shrink_mask"],
                    thresh_maps=out["thresh_maps"],
                    thresh_mask=out["thresh_mask"],
                )
            )

        return out

    def getitem2(self, index):
        text_boxes = self.data[index]["text_boxes"]
        image = self.data[index]["image"].convert("RGB").copy()
        image = np.array(image)

        out = {"image": image}
        img_size = (image.shape[0], image.shape[1])

        thresh_labels, padded_polys = self.make_border_map(text_boxes, img_size)
        shrink_labels, shrink_polys = self.make_segment_map(text_boxes, img_size)

        out.update(thresh_labels)
        out.update(shrink_labels)

        out["orig_image"] = image
        if self.return_text_boxes:
            out["text_box_dbs"] = self._create_text_boxdbs(text_boxes, padded_polys, shrink_polys)

        return out

    def _create_text_boxdbs(
        self, text_boxes: List[TextBox], padded_polys: List[Polygon], shrink_polys: List[Polygon]
    ) -> List[TextBoxDB]:
        text_boxes2 = []
        for tb, pp, sp in zip(text_boxes, padded_polys, shrink_polys):
            text_boxes2.append(
                TextBoxDB(
                    text=tb.text,
                    poly=tb.poly,
                    shrink_poly=sp,
                    padded_poly=pp,
                )
            )
        return text_boxes2

    def __len__(self):
        return len(self.data)
