"""Whisper ASR wrapper for round-trip evaluation.

Uses the local HF cache copy of openai/whisper-large-v3-turbo.
"""
from __future__ import annotations

_MODEL_ID = "openai/whisper-large-v3-turbo"
_pipe = None


def get_pipe():
    global _pipe
    if _pipe is None:
        import torch
        from transformers import pipeline

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _pipe = pipeline(
            "automatic-speech-recognition",
            model=_MODEL_ID,
            chunk_length_s=30,
            device=device,
        )
    return _pipe


def transcribe(wav_path: str) -> str:
    result = get_pipe()(wav_path)
    return result["text"].strip()
