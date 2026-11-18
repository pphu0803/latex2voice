"""Kokoro-82M local TTS wrapper (English). Falls back gracefully if not installed."""
from __future__ import annotations


def synthesize(text: str, out_path: str, voice: str = "af_heart", speed: float = 1.0) -> str:
    import soundfile as sf  # noqa
    from kokoro import KPipeline

    # NOTE: device='cpu' — ROCm first-run hangs on this box; CPU takes seconds.
    global _pipe
    if _pipe is None:
        _pipe = KPipeline(lang_code="a", device="cpu")  # 'a' = American English
    audio_chunks = []
    for result in _pipe(text, voice=voice, speed=speed):
        audio_chunks.append(result.audio)
    import numpy as np
    audio = np.concatenate(audio_chunks)
    sf.write(out_path, audio, 24000)
    return out_path


_pipe = None
