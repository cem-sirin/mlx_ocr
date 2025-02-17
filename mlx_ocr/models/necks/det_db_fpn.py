from dataclasses import dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn

from . import NeckConfig, NeckType


@dataclass
class DBFPNConfig(NeckConfig):
    in_channels: List[int]
    out_channels: int = 256
    neck_type: NeckType = "DBFPN"


class DBFPN(nn.Module):
    """Neck module from Differentiable Binarization"""

    def __init__(self, config: DBFPNConfig):
        super(DBFPN, self).__init__()
        self.config = config
        in_channels = config.in_channels
        out_channels = config.out_channels
        self.out_channels = out_channels

        self.in2_conv = nn.Conv2d(in_channels[0], out_channels, kernel_size=1, bias=False)
        self.in3_conv = nn.Conv2d(in_channels[1], out_channels, kernel_size=1, bias=False)
        self.in4_conv = nn.Conv2d(in_channels[2], out_channels, kernel_size=1, bias=False)
        self.in5_conv = nn.Conv2d(in_channels[3], out_channels, kernel_size=1, bias=False)

        self.p5_conv = nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p4_conv = nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p3_conv = nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=1, bias=False)
        self.p2_conv = nn.Conv2d(out_channels, out_channels // 4, kernel_size=3, padding=1, bias=False)

    def __call__(self, x: List[mx.array]) -> mx.array:
        c2, c3, c4, c5 = x

        # Project all features to the same dimension
        in5 = self.in5_conv(c5)
        in4 = self.in4_conv(c4)
        in3 = self.in3_conv(c3)
        in2 = self.in2_conv(c2)

        upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        upsample4 = nn.Upsample(scale_factor=4, mode="nearest")
        upsample8 = nn.Upsample(scale_factor=8, mode="nearest")

        # Start building the pyramid, i.e., from the smallest resolution to the largest
        out4 = in4 + upsample2(in5)  # 1/16
        out3 = in3 + upsample2(out4)  # 1/8
        out2 = in2 + upsample2(out3)  # 1/4

        # Reduce the features by a factor of 4, so that we can concatenate them together
        # to the same output dimension
        p5 = self.p5_conv(in5)
        p4 = self.p4_conv(out4)
        p3 = self.p3_conv(out3)
        p2 = self.p2_conv(out2)

        # Upsample all to the same resolution
        p5 = upsample8(p5)
        p4 = upsample4(p4)
        p3 = upsample2(p3)

        # Now these must have all the same shape, uncomment below to check
        # assert p5.shape == p4.shape == p3.shape == p2.shape

        fuse = mx.concat([p5, p4, p3, p2], axis=-1)  # Concat along channel axis
        return fuse
