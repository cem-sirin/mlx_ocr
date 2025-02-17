from dataclasses import dataclass
from typing import Literal

import mlx.nn as nn

BackboneType = Literal["mobilenetv3fpn", "mobilenetv1"]


@dataclass
class BackboneConfig:
    backbone_type: BackboneType


def build_backbone(config: BackboneConfig) -> nn.Module:

    if config.backbone_type == "mobilenetv1":
        from .rec_mobilenetv1 import MobileNetV1

        return MobileNetV1(config)
    elif config.backbone_type == "mobilenetv3fpn":
        from .det_mobilenetv3fpn import MobileNetV3FPN

        return MobileNetV3FPN(config)
    else:
        raise ValueError(f"Unsupported backbone_type: {config.backbone_type}")
