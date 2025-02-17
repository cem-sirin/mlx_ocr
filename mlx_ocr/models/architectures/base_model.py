from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from ..backbones import BackboneConfig, build_backbone
from ..necks import NeckConfig, build_neck
from ..heads import HeadConfig, build_head


@dataclass
class BaseModelConfig:
    backbone_config: BackboneConfig
    neck_config: NeckConfig
    head_config: HeadConfig


class BaseModel(nn.Module):
    def __init__(self, config: BaseModelConfig):
        super(BaseModel, self).__init__()

        self.config = config

        # Build backbone
        self.backbone = build_backbone(config.backbone_config)
        in_channels = self.backbone.out_channels

        # Build neck
        if config.neck_config is not None:
            config.neck_config.in_channels = in_channels
            self.neck = build_neck(config.neck_config)
            in_channels = self.neck.out_channels
        else:
            self.neck = nn.Identity()
            self.neck.out_channels = in_channels

        # Build head
        config.head_config.in_channels = in_channels
        self.head = build_head(config.head_config)

    def __call__(self, x: mx.array, head_inputs: dict = {}):
        return self.head(self.neck(self.backbone(x)), **head_inputs)
