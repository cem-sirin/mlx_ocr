"""
Differentiable Binarization (DB) Head for text detection.
Notes: In future, I will remove the if statements for training mode, since MLX will not
calculate thresh and binary maps if they are not called.

ArXiv: https://arxiv.org/abs/1911.08947
References:
- https://github.com/MhLiao/DB
- https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/heads/det_db_head.py
"""

from typing import Dict, List
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from . import HeadConfig, HeadType


@dataclass
class DBHeadConfig(HeadConfig):
    in_channels: int = None
    kernel_list: List[int] = field(default_factory=lambda: [3, 2, 2])
    k: int = 50
    fix_nan: bool = True
    head_type: HeadType = "DBHead"


class DBHead(nn.Module):
    """Differentiable Binarization (DB) Head for text detection:
    see https://arxiv.org/abs/1911.08947
    """

    def __init__(self, config: DBHeadConfig):
        super(DBHead, self).__init__()
        self.k = config.k
        self.binarize = Head(config.in_channels, config.kernel_list, fix_nan=config.fix_nan)
        self.thresh = Head(config.in_channels, config.kernel_list, fix_nan=config.fix_nan)

    def step_function(self, x: mx.array, y: mx.array) -> mx.array:
        return 1 / (1 + mx.exp(-self.k * (x - y)))

    def __call__(self, x: mx.array, targets=None) -> Dict[str, mx.array]:
        shrink_maps = self.binarize(x)
        if not self.training:
            return {"shrink_maps": shrink_maps}

        threshold_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, threshold_maps)
        return {"shrink_maps": shrink_maps, "thresh_maps": threshold_maps, "binary_maps": binary_maps}


class Head(nn.Module):
    """Some unnamed head module :)"""

    def __init__(self, in_channels, kernel_list=[3, 2, 2], fix_nan=True):
        super(Head, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, in_channels // 4, kernel_list[0], padding=int(kernel_list[0] // 2), bias=False
        )
        self.conv_bn1 = nn.BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, kernel_list[1], stride=2)
        self.conv_bn2 = nn.BatchNorm(in_channels // 4)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.ConvTranspose2d(in_channels // 4, 1, kernel_list[2], stride=2)
        self.fix_nan = fix_nan

    def __call__(self, x: mx.array) -> mx.array:

        x = self.relu1(self.conv_bn1(self.conv1(x)))
        if self.fix_nan and self.training:
            x = mx.where(mx.isnan(x), mx.zeros_like(x), x)

        x = self.relu2(self.conv_bn2(self.conv2(x)))
        if self.fix_nan and self.training:
            x = mx.where(mx.isnan(x), mx.zeros_like(x), x)

        x = self.conv3(x)
        x = mx.sigmoid(x)
        return x
