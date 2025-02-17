from dataclasses import dataclass
from typing import Literal

import mlx.nn as nn

NeckType = Literal["DBFPN", "RSEFPN", "LKPAN"]


@dataclass
class NeckConfig:
    neck_type: NeckType


def build_neck(config: NeckConfig) -> nn.Module:
    if config.neck_type == "DBFPN":
        from .det_db_fpn import DBFPN

        return DBFPN(config)
    elif config.neck_type == "RSEFPN":
        from .det_rse_fpn import RSEFPN

        return RSEFPN(config)
    elif config.neck_type == "LKPAN":
        from .det_lk_pan import LKPAN

        return LKPAN(config)
