"""
MobileNetV1 backbone implementation in MLX.
Reference: https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/backbones/rec_mv1_enhance.py
"""

from dataclasses import dataclass
from typing import Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from . import BackboneConfig, BackboneType
from ..modules.modules import ConvBNLayer, SEModule


@dataclass
class MobileNetV1Config(BackboneConfig):
    in_channels: int = 3
    scale: float = 0.5
    last_conv_stride: Union[int, Tuple[int, int]] = (1, 2)
    last_pool_type: str = "avg"
    last_pool_kernel_size: Union[int, Tuple[int, int]] = (2, 2)
    backbone_type: BackboneType = "mobilenetv1"


class MobileNetV1(nn.Module):
    def __init__(self, config: MobileNetV1Config):
        super(MobileNetV1, self).__init__()
        self.scale = config.scale
        self.block_list = []

        self.conv1 = ConvBNLayer(config.in_channels, int(32 * config.scale), 3, stride=2, padding=1, act="hardswish")

        # Block Configurations
        cfg = [
            # n_c, n_f1, n_f2, n_g, stride, dw_size, padding, use_se
            (32, 32, 64, 32, 1, 3, 1, False),  # conv2_1
            (64, 64, 128, 64, 1, 3, 1, False),  # conv2_2
            (128, 128, 128, 128, 1, 3, 1, False),  # conv3_1
            (128, 128, 256, 128, (2, 1), 3, 1, False),  # conv3_2
            (256, 256, 256, 256, 1, 3, 1, False),  # conv4_1
            (256, 256, 512, 256, (2, 1), 3, 1, False),  # conv4_2
            (512, 512, 512, 512, 1, 5, 2, False),  # conv5_1
            (512, 512, 512, 512, 1, 5, 2, False),  # conv5_2
            (512, 512, 512, 512, 1, 5, 2, False),  # conv5_3
            (512, 512, 512, 512, 1, 5, 2, False),  # conv5_4
            (512, 512, 512, 512, 1, 5, 2, False),  # conv5_5
            (512, 512, 1024, 512, (2, 1), 5, 2, True),  # conv5_6
            (1024, 1024, 1024, 1024, config.last_conv_stride, 5, 2, True),  # conv6
        ]

        self.block_list = []
        for n_c, n_f1, n_f2, n_g, stride, dw_size, padding, use_se in cfg:
            self.block_list.append(
                DepthwiseSeparable(
                    num_channels=int(n_c * config.scale),
                    num_filters1=n_f1,
                    num_filters2=n_f2,
                    num_groups=n_g,
                    stride=stride,
                    dw_size=dw_size,
                    padding=padding,
                    use_se=use_se,
                    scale=config.scale,
                )
            )

        self.block_list = nn.Sequential(*self.block_list)
        if config.last_pool_type == "avg":
            self.pool = nn.AvgPool2d(
                kernel_size=config.last_pool_kernel_size,
                stride=config.last_pool_kernel_size,
                padding=0,
            )
        else:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.out_channels = int(1024 * config.scale)

    def __call__(self, inputs: mx.array) -> mx.array:
        return self.pool(self.block_list(self.conv1(inputs)))


class DepthwiseSeparable(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_filters1: int,
        num_filters2: int,
        num_groups: int,
        stride: int,
        scale: float,
        dw_size: int = 3,
        padding: int = 1,
        use_se: bool = False,
    ):
        super(DepthwiseSeparable, self).__init__()
        self.use_se = use_se
        self.depthwise_conv = ConvBNLayer(
            in_channels=num_channels,
            out_channels=int(num_filters1 * scale),
            kernel_size=dw_size,
            stride=stride,
            padding=padding,
            groups=int(num_groups * scale),
            act="hardswish",
        )
        self.se = SEModule(int(num_filters1 * scale), slope=1 / 6) if use_se else nn.Identity()
        self.pointwise_conv = ConvBNLayer(
            in_channels=int(num_filters1 * scale),
            out_channels=int(num_filters2 * scale),
            kernel_size=1,
            stride=1,
            padding=0,
            act="hardswish",
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        y = self.depthwise_conv(inputs)
        y = self.se(y)
        y = self.pointwise_conv(y)
        return y
