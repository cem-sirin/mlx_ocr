from dataclasses import dataclass
from typing import List, Literal

import mlx.core as mx
import mlx.nn as nn

from . import NeckConfig, NeckType
from ..modules.modules import DSConv, IntraCLBlock


@dataclass
class LKPANConfig(NeckConfig):
    in_channels: List[int]
    out_channels: int
    mode: Literal["lite", "large"] = "large"
    intracl: bool = True
    neck_type: NeckType = "LKPAN"


class LKPAN(nn.Module):
    """Large Kernel PAN (Pixel Aggregation Network)"""

    def __init__(self, config: LKPANConfig):
        super(LKPAN, self).__init__()
        self.config = config
        in_channels = config.in_channels
        out_channels = config.out_channels
        mode = config.mode
        self.out_channels = out_channels
        self.intracl = config.intracl

        self.ins_conv = []
        self.inp_conv = []

        # pan head
        self.pan_head_conv = []
        self.pan_lat_conv = []

        p_layer = DSConv if mode == "lite" else nn.Conv2d

        for i in range(len(in_channels)):
            self.ins_conv += [nn.Conv2d(in_channels[i], out_channels, kernel_size=1, bias=False)]
            self.inp_conv += [p_layer(out_channels, out_channels // 4, kernel_size=9, padding=4, bias=False)]
            self.pan_lat_conv.append(p_layer(out_channels // 4, out_channels // 4, 9, padding=4, bias=False))

        for i in range(len(in_channels) - 1):
            self.pan_head_conv += [nn.Conv2d(out_channels // 4, out_channels // 4, 3, stride=2, padding=1, bias=False)]

        if self.intracl:
            self.incl1 = IntraCLBlock(out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(out_channels // 4, reduce_factor=2)

    def __call__(self, x: List[mx.array]) -> mx.array:
        c2, c3, c4, c5 = x

        in5 = self.ins_conv[3](c5)
        in4 = self.ins_conv[2](c4)
        in3 = self.ins_conv[1](c3)
        in2 = self.ins_conv[0](c2)

        upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        upsample4 = nn.Upsample(scale_factor=4, mode="nearest")
        upsample8 = nn.Upsample(scale_factor=8, mode="nearest")

        out4 = in4 + upsample2(in5)  # 1/16
        out3 = in3 + upsample2(out4)  # 1/8
        out2 = in2 + upsample2(out3)  # 1/4

        f5 = self.inp_conv[3](in5)
        f4 = self.inp_conv[2](out4)
        f3 = self.inp_conv[1](out3)
        f2 = self.inp_conv[0](out2)

        pan3 = f3 + self.pan_head_conv[0](f2)
        pan4 = f4 + self.pan_head_conv[1](pan3)
        pan5 = f5 + self.pan_head_conv[2](pan4)

        p2 = self.pan_lat_conv[0](f2)
        p3 = self.pan_lat_conv[1](pan3)
        p4 = self.pan_lat_conv[2](pan4)
        p5 = self.pan_lat_conv[3](pan5)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = upsample8(p5)
        p4 = upsample4(p4)
        p3 = upsample2(p3)

        fuse = mx.concat([p5, p4, p3, p2], axis=-1)  # Concat along channel axis
        return fuse
