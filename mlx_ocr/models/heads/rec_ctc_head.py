"""
Connectionist Temporal Classification (CTC) head for OCR models.
Reference: https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppocr/modeling/heads/rec_ctc_head.py
"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from . import HeadConfig, HeadType


@dataclass
class CTCHeadConfig(HeadConfig):
    in_channels: int = 512
    out_channels: int = 512
    mid_channels: int = None
    head_type: HeadType = "CTCHead"

    def __post_init__(self):
        assert self.mid_channels is None or self.mid_channels > 0


class CTCHead(nn.Module):
    def __init__(self, config: CTCHeadConfig):
        super(CTCHead, self).__init__()
        if config.mid_channels is None:
            self.fc = nn.Linear(config.in_channels, config.out_channels)
        else:
            self.fc1 = nn.Linear(config.in_channels, config.mid_channels)
            self.fc2 = nn.Linear(config.mid_channels, config.out_channels)

        self.out_channels = config.out_channels
        self.mid_channels = config.mid_channels

    def __call__(self, x: mx.array) -> mx.array:

        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            predicts = self.fc2(self.fc1(x))

        if not self.training:
            predicts = mx.softmax(predicts, axis=-1)
        return predicts
