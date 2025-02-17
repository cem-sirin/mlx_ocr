from ..model_loader import load_weights
from .sanitize import load_and_process_weights
from .download import get_weight_path
from .constants import MODEL_URLS

PPOCRv3_REC_MAPPINGS = {
    "._conv.": ".conv.",
    ".block_list.": ".block_list.layers.",
    "._depthwise_conv.": ".depthwise_conv.",
    "._pointwise_conv.": ".pointwise_conv.",
    "._se.": ".se.",
    ".rnn_decoder.": ".rnn_decoder.layers.",
    ".rnn_encoder.": ".rnn_encoder.layers.",
}
PPOCRv3_DET_PATTERN_MAPPINGS = {r"stage(\d+)\.(\d+)": r"stages.\1.layers.\2"}


def load_det_pp_ocrv3(lang: str = "eng", select_model: str = "Student"):
    assert (
        lang in MODEL_URLS["OCR"]["PP-OCRv3"]["det"].keys()
    ), f"Lang supports only {MODEL_URLS['OCR']['PP-OCRv3']['det'].keys()}, got {lang}."

    from mlx_ocr.models.architectures.base_model import BaseModel, BaseModelConfig
    from mlx_ocr.models.backbones.det_mobilenetv3fpn import MobileNetV3FPNConfig
    from mlx_ocr.models.necks.det_rse_fpn import RSEFPNConfig
    from mlx_ocr.models.heads.det_db_head import DBHeadConfig

    model = BaseModel(
        BaseModelConfig(
            backbone_config=MobileNetV3FPNConfig(),
            neck_config=RSEFPNConfig(),
            head_config=DBHeadConfig(),
        )
    )
    weight_path = get_weight_path("det", lang)
    weights = load_and_process_weights(
        model, weight_path, pattern_mappings=PPOCRv3_DET_PATTERN_MAPPINGS, select_model=select_model
    )
    model = load_weights(model, weights)
    model.name = f"PP-OCRv3_{lang}_det[{select_model}]"
    return model


def load_rec_pp_ocrv3(lang: str = "lat", select_model: str = "Student"):
    from mlx_ocr.models.architectures.base_model import BaseModel, BaseModelConfig
    from mlx_ocr.models.backbones.rec_mobilenetv1 import MobileNetV1Config
    from mlx_ocr.models.heads.rec_multi_head import MultiHeadConfig
    from mlx_ocr.models.heads.rec_ctc_head import CTCHeadConfig
    from mlx_ocr.models.heads.rec_sar_head import SARHeadConfig

    weight_path, vocab_path = get_weight_path("rec", lang)

    with open(vocab_path, "r") as f:
        vocab = f.read().splitlines()

    vocab_len = len(vocab)
    config = BaseModelConfig(
        backbone_config=MobileNetV1Config(),
        neck_config=None,
        head_config=MultiHeadConfig(
            ctc_config=CTCHeadConfig(out_channels=vocab_len + 2),
            sar_config=SARHeadConfig(out_channels=vocab_len + 4),
        ),
    )

    model = BaseModel(config)
    weights = load_and_process_weights(
        model, weight_path, additional_key_mappings=PPOCRv3_REC_MAPPINGS, select_model=select_model
    )
    model = load_weights(model, weights)
    model.vocab_path = vocab_path
    model.name = f"PP-OCRv3_{lang}_rec[{select_model}]"
    return model
