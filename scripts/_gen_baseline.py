"""Generate 'stock TTS reads raw LaTeX' baseline audio for the demo comparison.

No TN: the raw LaTeX string goes straight into Kokoro, exactly what an
untuned TTS pipeline would do. These clips are the 'Before' half of the
Before/After cards.
"""
import sys

sys.path.insert(0, ".")
from src.latextts.tts_kokoro import synthesize  # noqa: E402
from pathlib import Path  # noqa: E402

OUT = Path("demo/audio")
OUT.mkdir(parents=True, exist_ok=True)

BASELINES = [
    ("baseline-limsup", r"\limsup_{n \to \infty} a_n = \inf_{n} \sup_{k \geq n} a_k"),
    ("baseline-pmod", r"a \equiv b \pmod{n}"),
]
for name, raw in BASELINES:
    wav = OUT / f"{name}.wav"
    synthesize(raw, str(wav))
    print(f"[{name}] raw -> {wav}", flush=True)
print("BASELINE-DONE")
