"""Download required models. Qwen3-4B via ModelScope; Kokoro via pip/model scope.

Usage: python scripts/download_models.py [--qwen] [--kokoro]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MS_DOWNLOAD = Path(r"H:\SHARE\arch-transfer\setup\ms_download.py")  # reuse proven downloader
QWEN_DIR = ROOT / "models" / "Qwen3-4B"


def download_qwen() -> None:
    repo = "Qwen/Qwen3-4B"
    files = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
             "vocab.json", "merges.txt", "model-00001-of-00003.safetensors",
             "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]
    QWEN_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        url = f"https://www.modelscope.cn/{repo}/resolve/master/{f}"
        print(f"downloading {f} ...")
        subprocess.run([sys.executable, "-I", str(MS_DOWNLOAD), url, str(QWEN_DIR / f)], check=False)


def install_kokoro() -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "kokoro>=0.9", "soundfile"], check=True)
    print("kokoro installed; weights download on first use (HF cache)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen", action="store_true")
    ap.add_argument("--kokoro", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all or a.qwen:
        download_qwen()
    if a.all or a.kokoro:
        install_kokoro()
