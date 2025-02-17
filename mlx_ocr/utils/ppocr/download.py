from typing import Union, Tuple
import os
import os.path as osp
import time
import shutil
import tarfile
import requests
from tqdm import tqdm
from .constants import MODELS_DIR, get_weight_url
from ..logger import setup_logger

logger = setup_logger(__name__)
DOWNLOAD_RETRY_LIMIT = 3


def download_with_progressbar(url, save_path):
    if save_path and os.path.exists(save_path):
        return
    else:
        _download(url, save_path)


def _download(url, save_path):
    """
    Download from url, save to path.

    url (str): download url
    save_path (str): download to given path
    """

    fname = osp.split(url)[-1]
    retry_cnt = 0

    while not osp.exists(save_path):
        if retry_cnt < DOWNLOAD_RETRY_LIMIT:
            retry_cnt += 1
        else:
            raise RuntimeError("Download from {} failed. " "Retry limit reached".format(url))

        try:
            req = requests.get(url, stream=True)
        except Exception as e:  # requests.exceptions.ConnectionError
            print("Downloading {} from {} failed {} times with exception {}".format(fname, url, retry_cnt + 1, str(e)))
            time.sleep(1)
            continue

        if req.status_code != 200:
            raise RuntimeError("Downloading from {} failed with code " "{}!".format(url, req.status_code))

        # For protecting download interrupted, download to
        # tmp_file firstly, move tmp_file to save_path
        # after download finished
        tmp_file = save_path + ".tmp"
        os.makedirs(osp.dirname(tmp_file), exist_ok=True)
        total_size = req.headers.get("content-length")
        with open(tmp_file, "wb") as f:
            if total_size:
                with tqdm(total=(int(total_size) + 1023) // 1024) as pbar:
                    for chunk in req.iter_content(chunk_size=1024):
                        f.write(chunk)
                        pbar.update(1)
            else:
                for chunk in req.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        shutil.move(tmp_file, save_path)

    return save_path


def get_weight_path(task: str, lang: str, option: str = None) -> Union[str, Tuple[str, str]]:
    url = get_weight_url(task, lang)

    if task == "rec":
        url, vocab_path = url

    weight_dir = osp.join(MODELS_DIR, "/".join(url.split("/")[-3:]))

    if osp.exists(weight_dir):
        weight_dir = weight_dir.replace(".tar", "")
    else:
        logger.info(f"Downloading model from {url} to {weight_dir}")
        download_with_progressbar(url, weight_dir)

        # Downloaded the tar file, now extract it
        with tarfile.open(weight_dir, "r") as tar:
            tar.extractall(osp.dirname(weight_dir))
        weight_dir = weight_dir.replace(".tar", "")

    options = [f for f in os.listdir(weight_dir) if f.endswith(".pdparams")]
    if len(options) > 1:
        print(option is None)
        if option is None:
            logger.info(
                f"Note: there are multiple options available: {options}. Defaulting to first. You can set the option using the `select_model` argument."
            )
            option = options[0]
        elif option not in options:
            logger.info(f"Option '{option}' not found. Available options: {options}, defaulting to first.")
            option = options[0]
    elif len(options) == 1:
        if option is None:
            option = options[0]
        elif option is not None and options[0] != option:
            logger.info(f"Option '{option}' not found. Using '{options[0]}' from {options}.")
            option = options[0]

    elif len(options) == 0:
        logger.error(
            f"No .pdparams files found in {weight_dir}. This is unexpected, check the download or create an issue."
        )
        raise FileNotFoundError(f"No .pdparams files found in {weight_dir}")

    weight_path = osp.join(weight_dir, option)

    if task == "rec":
        return weight_path, vocab_path
    else:
        return weight_path
