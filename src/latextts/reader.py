"""reader.py — text with LaTeX -> spoken English + one wav (prototype).

Segments plain text vs formulas with regex; formula segments go through the
TN model, plain segments pass through unchanged. The joined spoken text is
synthesized as a single Kokoro utterance (v1; per-segment concat with pauses
is the v2 upgrade).
"""
from __future__ import annotations

import hashlib
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.latextts.tn_data import SYSTEM_PROMPT, USER_TMPL  # noqa: E402

# $$...$$  |  $...$  |  \(...\)  |  \[...\]
_SEGMENT = re.compile(r"\$\$(.+?)\$\$|\$([^$]+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)

_model_cache = {}
AUDIO_DIR = ROOT / "data" / "audio" / "reader"


def _get_model():
    """TN model singleton; CPU device (ROCm hangs Kokoro; TN gen is CPU-safe)."""
    if "m" not in _model_cache:
        from transformers import AutoModelForCausalLM, AutoTokenizer, PeftModel  # noqa: PLC0415

        base_dir = Path(r"H:\SHARE\latex-tts\models\Qwen3-4B")
        lora_dir = ROOT / "models" / "tn-lora"
        tok = AutoTokenizer.from_pretrained(base_dir)
        model = AutoModelForCausalLM.from_pretrained(
            base_dir, dtype=torch.bfloat16, attn_implementation="eager"
        )
        if lora_dir.exists():
            model = PeftModel.from_pretrained(model, lora_dir)
        model.eval()
        _model_cache["m"], _model_cache["t"] = model, tok
    return _model_cache["m"], _model_cache["t"]


def tn_formula(latex: str, max_new_tokens: int = 150) -> str:
    model, tok = _get_model()
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TMPL.format(latex=latex)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


def _segments(text: str) -> list[tuple[str, str]]:
    """[(kind, payload)] with kind in {plain, formula}, formulas unwrapped."""
    out: list[tuple[str, str]] = []
    pos = 0
    for m in _SEGMENT.finditer(text):
        if m.start() > pos:
            out.append(("plain", text[pos:m.start()]))
        latex = next(g for g in m.groups() if g is not None)
        out.append(("formula", latex.strip()))
        pos = m.end()
    if pos < len(text):
        out.append(("plain", text[pos:]))
    return out


def convert_text(text: str, want_audio: bool = True) -> tuple[str, str]:
    """Returns (joined spoken text, audio path or '')."""
    parts = []
    for kind, payload in _segments(text):
        parts.append(payload if kind == "plain" else tn_formula(payload))
    spoken = re.sub(r"\s+", " ", " ".join(parts)).strip()

    audio_path = ""
    if want_audio and spoken:
        AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha1(spoken.encode()).hexdigest()[:16]
        audio_path = str(AUDIO_DIR / f"{key}.wav")
        if not Path(audio_path).exists():  # formula-hash caching
            from src.latextts.tts_kokoro import synthesize  # noqa: PLC0415

            synthesize(spoken, audio_path)
    return spoken, audio_path


if __name__ == "__main__":
    demo = ("The loss follows $L = \\frac{1}{n} \\sum_{i=1}^{n} \\ell(y_i, \\hat{y}_i)$ "
            "with learning rate 0.001, so $a \\equiv b \\pmod{n}$ holds.")
    print(convert_text(demo))
