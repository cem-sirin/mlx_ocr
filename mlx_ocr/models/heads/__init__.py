from dataclasses import dataclass
from typing import Literal

import mlx.nn as nn

HeadType = Literal["CTCHead", "DBHead", "MultiHead", "PFHead", "SARHead"]


@dataclass
class HeadConfig:
    head_type: HeadType


def build_head(config: HeadConfig) -> nn.Module:
    if config.head_type == "CTCHead":
        from .rec_ctc_head import CTCHead

        return CTCHead(config)
    if config.head_type == "DBHead":
        from .det_db_head import DBHead

        return DBHead(config)

    elif config.head_type == "PFHead":
        from .det_pf_head import PFHeadLocal

        return PFHeadLocal(config)

    elif config.head_type == "MultiHead":
        from .rec_multi_head import MultiHead

        return MultiHead(config)
    elif config.head_type == "SARHead":
        from .rec_sar_head import SARHead

        return SARHead(config)
    else:
        raise ValueError(f"Unknown head type: {config.head_type}")
