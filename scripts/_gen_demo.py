"""Generate demo audio: famous formulas from 'Attention Is All You Need'.

TN -> Kokoro TTS -> wav files for the demo page. CPU route (stable).
"""
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402

from src.latextts.tn_data import SYSTEM_PROMPT, USER_TMPL  # noqa: E402
from src.latextts.tn_infer import get_model  # noqa: E402
from src.latextts.tts_kokoro import synthesize  # noqa: E402

from pathlib import Path  # noqa: E402

OUT = Path("demo/audio")
OUT.mkdir(parents=True, exist_ok=True)

model, tok = get_model("cpu")


def tn(latex: str) -> str:
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TMPL.format(latex=latex)},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=150, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


FORMULAS = [
    ("softmax", r"\operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}",
     "softmax(z)_i = e^{z_i} / Σ_{j=1..n} e^{z_j}"),
    ("relu", r"\hat{y} = \operatorname{ReLU}(W x + b)",
     "ŷ = ReLU(Wx + b)"),
    ("layernorm", r"\operatorname{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}",
     "LayerNorm(x) = γ · (x-μ)/√(σ²+ε)"),
    ("limsup-inf-sup", r"\limsup_{n \to \infty} a_n = \inf_{n} \sup_{k \geq n} a_k",
     "limsup a_n = inf_n sup_{k≥n} a_k"),
    ("kernel-trace-rank", r"\ker(A) = \{0\}, \quad \tr(A) = \sum_{i=1}^{n} \lambda_i, \quad \rank(A) = 3",
     "ker(A) = {0}, tr(A) = Σλ_i, rank(A) = 3"),
    ("diag-sign", r"\diag(v) \cdot \sign(v) \geq 0",
     "diag(v) · sign(v) ≥ 0"),
    ("pmod", r"a \equiv b \pmod{n}",
     "a ≡ b (mod n)"),
    ("prob-hom", r"\Pr(X > 0) \geq \hom(X, Y) - \lg n",
     "Pr(X>0) ≥ hom(X,Y) - lg n"),
]

import json  # noqa: E402

spoken_map = {}
for name, latex, _display in FORMULAS:
    spoken = tn(latex)
    wav = OUT / f"{name}.wav"
    synthesize(spoken, str(wav))
    spoken_map[name] = spoken
    print(f"[{name}]\n  latex : {latex}\n  spoken: {spoken}\n  audio : {wav}", flush=True)

(OUT / "spoken.json").write_text(json.dumps(spoken_map, ensure_ascii=False, indent=1), encoding="utf-8")

# regression spot checks
from src.latextts.eval_roundtrip import normalize_en  # noqa: E402
checks = {
    r"(a+b)^2 = a^2 + 2ab + b^2": "the quantity a plus b, squared equals a squared plus 2 a b plus b squared",
}
for latex, expect in checks.items():
    pred = tn(latex)
    print(f"[REGRESS] {latex!r} -> {pred!r} match={normalize_en(expect) == normalize_en(pred)}", flush=True)

print("DEMO-DONE")
