"""
MobileNetV3FPN backbone implementation in MLX.
Notes: The name has been changed to MobileNetV3FPN to avoid conflict with the existing MobileNetV3 backbone.
Reference: https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/det_mobilenet_v3.py
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

import mlx.core as mx
import mlx.nn as nn

from . import BackboneConfig, BackboneType
from ..modules.modules import SEModule, ConvBNLayer, make_divisible

SUPPORTED_SCALE = [0.35, 0.5, 0.75, 1.0, 1.25]
LARGE_CFG = [
    # k, exp, c,  se,     nl,  s,
    [3, 16, 16, False, "relu", 1],
    [3, 64, 24, False, "relu", 2],
    [3, 72, 24, False, "relu", 1],
    [5, 72, 40, True, "relu", 2],
    [5, 120, 40, True, "relu", 1],
    [5, 120, 40, True, "relu", 1],
    [3, 240, 80, False, "hardswish", 2],
    [3, 200, 80, False, "hardswish", 1],
    [3, 184, 80, False, "hardswish", 1],
    [3, 184, 80, False, "hardswish", 1],
    [3, 480, 112, True, "hardswish", 1],
    [3, 672, 112, True, "hardswish", 1],
    [5, 672, 160, True, "hardswish", 2],
    [5, 960, 160, True, "hardswish", 1],
    [5, 960, 160, True, "hardswish", 1],
]
SMALL_CFG = [
    # k, exp, c,  se,     nl,  s,
    [3, 16, 16, True, "relu", 2],
    [3, 72, 24, False, "relu", 2],
    [3, 88, 24, False, "relu", 1],
    [5, 96, 40, True, "hardswish", 2],
    [5, 240, 40, True, "hardswish", 1],
    [5, 240, 40, True, "hardswish", 1],
    [5, 120, 48, True, "hardswish", 1],
    [5, 144, 48, True, "hardswish", 1],
    [5, 288, 96, True, "hardswish", 2],
    [5, 576, 96, True, "hardswish", 1],
    [5, 576, 96, True, "hardswish", 1],
]


@dataclass
class MobileNetV3FPNConfig(BackboneConfig):
    """Configuration for MobileNetV3FPN backbone. The default values reproduce the student
    model of PP-OCRv3.
    Args:
        model_name (str): The model name. It must be "large" or "small".
        in_channels (int): The number of input channels.
        scale (float): The scale factor of the model. It must be in [0.35, 0.5, 0.75, 1.0, 1.25].
        disable_se (bool): Whether to disable the SE module.
        backbone_type (BackboneType): The backbone type.
    """

    model_name: Literal["large", "small"] = "large"
    in_channels: int = 3
    scale: float = 0.5
    disable_se: bool = True
    backbone_type: BackboneType = "mobilenetv3fpn"


class MobileNetV3FPN(nn.Module):
    def __init__(self, config: MobileNetV3FPNConfig):
        super(MobileNetV3FPN, self).__init__()
        assert config.model_name in ["large", "small"], "model_name must be in ['large', 'small']"
        assert config.scale in SUPPORTED_SCALE, f"supported scale are {SUPPORTED_SCALE}, got scale={config.scale}"

        self.config = config

        self.disable_se = config.disable_se
        cfg = LARGE_CFG if config.model_name == "large" else SMALL_CFG
        cls_ch_squeeze = 960 if config.model_name == "large" else 576

        scale = config.scale

        inplanes = 16
        self.conv = ConvBNLayer(
            in_channels=config.in_channels,
            out_channels=make_divisible(inplanes * scale),
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            act="hardswish",
        )

        self.stages, self.out_channels = [], []
        block_list, i = [], 0
        inplanes = make_divisible(inplanes * scale)
        for k, exp, c, se, nl, s in cfg:
            se = se and not self.disable_se
            start_idx = 2 if config.model_name == "large" else 0

            if s == 2 and i > start_idx:
                self.out_channels.append(inplanes)
                self.stages.append(nn.Sequential(*block_list))
                block_list = []

            block_list.append(
                ResidualUnit(
                    in_channels=inplanes,
                    mid_channels=make_divisible(scale * exp),
                    out_channels=make_divisible(scale * c),
                    kernel_size=k,
                    stride=s,
                    act=nl,
                )
            )
            inplanes = make_divisible(scale * c)
            i += 1

        block_list.append(
            ConvBNLayer(
                inplanes,
                make_divisible(scale * cls_ch_squeeze),
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                act="hardswish",
            )
        )
        self.stages.append(nn.Sequential(*block_list))
        self.out_channels.append(make_divisible(scale * cls_ch_squeeze))

    def __call__(self, x: mx.array) -> List[mx.array]:
        x = self.conv(x)
        out_list = []
        for stage in self.stages:
            x = stage(x)
            out_list.append(x)
        return out_list


class ResidualUnit(nn.Module):
    """Residual unit for MobileNetV3."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool = False,
        act: str = None,
    ):
        super(ResidualUnit, self).__init__()
        assert act in [None, "relu", "hardswish"]
        self.if_shortcut = stride == 1 and in_channels == out_channels
        self.expand_conv = ConvBNLayer(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, act=act)

        padding = int((kernel_size - 1) // 2)
        self.bottleneck_conv = ConvBNLayer(
            mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, act=act
        )
        self.mid_se = SEModule(mid_channels) if use_se else nn.Identity()
        self.linear_conv = ConvBNLayer(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, act=None)

    def __call__(self, inputs: mx.array) -> mx.array:
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x += inputs
        return x
