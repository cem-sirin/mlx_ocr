"""Work in progress... Use at your own risk!"""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from . import HeadConfig, HeadType
from ..modules.modules import ConvBNLayer


@dataclass
class PFHeadConfig(HeadConfig):
    in_channels: int
    k: int = 50
    mode: str = "small"
    head_type: HeadType = "PFHead"


class PFHeadLocal(nn.Module):
    # def __init__(self, in_channels: int, k=50, mode="small"):
    def __init__(self, config: PFHeadConfig):
        super().__init__()
        self.config = config
        in_channels = config.in_channels

        self.up_conv = nn.Upsample(scale_factor=2, mode="nearest")  # , align_mode=1)
        if config.mode == "large":
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 4)
        elif config.mode == "small":
            self.cbn_layer = LocalModule(in_channels // 4, in_channels // 8)

    def __call__(self, x):
        shrink_maps, f = self.binarize(x, return_f=True)
        base_maps = shrink_maps
        cbn_maps = self.cbn_layer(self.up_conv(f), shrink_maps, None)
        cbn_maps = mx.sigmoid(cbn_maps)
        if not self.training:
            return {"maps": 0.5 * (base_maps + cbn_maps)}

        thresh_maps = self.thresh(x)
        binary_maps = self.step_function(shrink_maps, thresh_maps)

        # The dict below is quite confusing, I am just matching the original code :)
        return {
            "shrink_maps": cbn_maps,
            "thresh_maps": thresh_maps,
            "binary_maps": binary_maps,
            "distance_maps": cbn_maps,
            "cbn_maps": binary_maps,
        }


class LocalModule(nn.Module):
    def __init__(self, in_c, mid_c, use_distance=True):
        super(LocalModule, self).__init__()
        self.last_3 = ConvBNLayer(in_c + 1, mid_c, 3, 1, 1, act="relu")
        self.last_1 = nn.Conv2D(mid_c, 1, 1, 1, 0)

    def __call__(self, init_map, distance_map):
        outf = mx.concat([init_map, distance_map], axis=1)
        # last Conv
        out = self.last_1(self.last_3(outf))
        return out
