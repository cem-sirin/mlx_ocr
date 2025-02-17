import os
import os.path as osp

# Get the absolute path to the project root directory
PROJECT_ROOT = osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
VOCABS_DIR = osp.join(PROJECT_ROOT, "mlx_ocr", "misc", "vocabs")
PPOCR_URL = "https://paddleocr.bj.bcebos.com"
MODELS_DIR = os.path.expanduser("~/.paddleocr/models/")
MODEL_URLS = {
    "OCR": {
        "PP-OCRv3": {
            "det": {
                "zho": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/chinese/ch_PP-OCRv3_det_distill_train.tar",
                },
                "eng": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/english/en_PP-OCRv3_det_distill_train.tar",
                },
                "ml": {"url": f"{PPOCR_URL}/PP-OCRv3/multilingual/Multilingual_PP-OCRv3_det_distill_train.tar"},
            },
            "rec": {
                "zho": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/chinese/ch_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/zho.txt",
                },
                "eng": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/english/en_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/eng.txt",
                },
                "kor": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/multilingual/korean_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/kor.txt",
                },
                "jpn": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/multilingual/japan_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/jpn.txt",
                },
                "zho_cht": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/multilingual/chinese_cht_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/zho_cht.txt",
                },
                "tam": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/multilingual/ta_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/tam.txt",
                },
                "tel": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/multilingual/te_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/tel.txt",
                },
                "kan": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/multilingual/ka_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/kan.txt",
                },
                "lat": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/multilingual/latin_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/lat.txt",
                },
                "ara": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/multilingual/arabic_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/ara.txt",
                },
                "cyr": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/multilingual/cyrillic_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/cyr.txt",
                },
                "devanagari": {
                    "url": f"{PPOCR_URL}/PP-OCRv3/multilingual/devanagari_PP-OCRv3_rec_train.tar",
                    "vocab_path": f"{VOCABS_DIR}/devanagari.txt",
                },
            },
            "cls": {
                "ch": {
                    "url": f"{PPOCR_URL}/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_train.tar",
                }
            },
        },
    },
}


def get_weight_url(task: str, lang: str):
    if task == "det":
        assert (
            lang in MODEL_URLS["OCR"]["PP-OCRv3"]["det"].keys()
        ), f"Lang supports only {MODEL_URLS['OCR']['PP-OCRv3']['det'].keys()}, got {lang}."

        return MODEL_URLS["OCR"]["PP-OCRv3"]["det"][lang]["url"]
    elif task == "rec":
        assert (
            lang in MODEL_URLS["OCR"]["PP-OCRv3"]["rec"].keys()
        ), f"Lang supports only {MODEL_URLS['OCR']['PP-OCRv3']['rec'].keys()}, got {lang}."

        d = MODEL_URLS["OCR"]["PP-OCRv3"]["rec"][lang]
        return d["url"], d["vocab_path"]
    else:
        raise ValueError(f"Task {task} not supported.")
