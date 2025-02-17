from dataclasses import dataclass, field
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

from .rec_sar_head import SARHeadConfig
from .rec_ctc_head import CTCHeadConfig
from . import build_head, HeadConfig, HeadType
from ..modules.sequences import SequenceEncoder

PPOCRv3_ENC_ARGS = {
    "dims": 64,
    "depth": 2,
    "hidden_dims": 120,
    "use_guide": True,
}


@dataclass
class MultiHeadConfig(HeadConfig):
    in_channels: int = None
    ctc_config: CTCHeadConfig = field(default_factory=lambda: CTCHeadConfig())
    sar_config: SARHeadConfig = field(default_factory=lambda: SARHeadConfig())
    ctc_encoder_type: str = "svtr"
    ctc_encoder_args: dict = field(default_factory=lambda: PPOCRv3_ENC_ARGS)
    head_type: HeadType = "MultiHead"


class MultiHead(nn.Module):
    def __init__(self, config: MultiHeadConfig):
        super().__init__()

        self.in_channels = config.in_channels
        self.ctc_encoder = SequenceEncoder(config.in_channels, config.ctc_encoder_type, **config.ctc_encoder_args)
        config.ctc_config.in_channels = self.ctc_encoder.out_channels  # update the in_channels of ctc head
        self.ctc_head = build_head(config.ctc_config)
        self.sar_head = build_head(config.sar_config)

    def __call__(self, x: mx.array, labels_sar: Optional[mx.array] = None, valid_ratios: Optional[List[float]] = None):
        """
        Args:
            x: the feature map from backbone, [N, C, H, W]
            labels_sar: the encoded labels for sar head, [N, L]. Required during training.
            valid_ratios: the ratio of the width of the actual image and the padded image, [N,]
        """
        if valid_ratios is not None:
            assert len(valid_ratios) == x.shape[0], f"valid_ratios: {len(valid_ratios)} != x: {x.shape[0]}"

        ctc_encoder = self.ctc_encoder(x)
        ctc_out = self.ctc_head(ctc_encoder)
        # eval mode
        if not self.training:
            return ctc_out

        sar_out = self.sar_head(x, labels_sar, valid_ratios)
        return {"ctc": ctc_out, "ctc_neck": ctc_encoder, "sar": sar_out}
