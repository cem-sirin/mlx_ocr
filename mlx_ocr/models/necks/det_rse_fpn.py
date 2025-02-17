from dataclasses import dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn

from . import NeckConfig, NeckType
from ..modules.modules import SEModule, IntraCLBlock


@dataclass
class RSEFPNConfig(NeckConfig):
    in_channels: List[int] = None
    out_channels: int = 96
    shortcut: bool = True
    intracl: bool = False
    neck_type: NeckType = "RSEFPN"


class RSEFPN(nn.Module):
    """Residual Squeeze-and-Excitation FPN (Feature Pyramid Network)"""

    def __init__(self, config: RSEFPNConfig):
        super(RSEFPN, self).__init__()

        in_channels = config.in_channels
        out_channels = config.out_channels
        self.out_channels = out_channels

        self.ins_conv = []
        self.inp_conv = []

        self.intracl = config.intracl
        if self.intracl:
            self.incl1 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl2 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl3 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)
            self.incl4 = IntraCLBlock(self.out_channels // 4, reduce_factor=2)

        for i in range(len(in_channels)):
            self.ins_conv.append(RSELayer(in_channels[i], out_channels, kernel_size=1, shortcut=config.shortcut))
            self.inp_conv.append(RSELayer(out_channels, out_channels // 4, kernel_size=3, shortcut=config.shortcut))

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

        p5 = self.inp_conv[3](in5)
        p4 = self.inp_conv[2](out4)
        p3 = self.inp_conv[1](out3)
        p2 = self.inp_conv[0](out2)

        if self.intracl is True:
            p5 = self.incl4(p5)
            p4 = self.incl3(p4)
            p3 = self.incl2(p3)
            p2 = self.incl1(p2)

        p5 = upsample8(p5)
        p4 = upsample4(p4)
        p3 = upsample2(p3)

        fuse = mx.concat([p5, p4, p3, p2], axis=-1)
        return fuse


class RSELayer(nn.Module):
    """Residual Squeeze-and-Excitation Layer"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, shortcut=True):
        super(RSELayer, self).__init__()
        self.out_channels = out_channels
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.se_block = SEModule(self.out_channels)
        self.shortcut = shortcut

    def __call__(self, ins: mx.array) -> mx.array:
        x = self.in_conv(ins)
        out = self.se_block(x)
        if self.shortcut:
            out = x + out
        return out
