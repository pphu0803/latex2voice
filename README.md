# latex2voice

[![Live demo](https://img.shields.io/badge/demo-demo.latex2voice.dev-8c1d18)](https://demo.latex2voice.dev)
[![Hugging Face weights](https://img.shields.io/badge/%F0%9F%A4%97-weights-ffb200)](https://huggingface.co/petehu0803/latex2voice)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

**Make any TTS read math like a mathematician.** A LoRA-tuned text-normalization
front end converts LaTeX into fluent spoken English, then hands it to any
TTS engine — fully offline.

```text
"\limsup_{n \to \infty} a_n"
    → "the limit superior as n goes to infinity of a sub n"

"a \equiv b \pmod{n}"
    → "a is congruent to b modulo n"

"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"
    → "the 2 by 2 matrix with, first row 1, 2, and second row 3, 4"
```

Stock TTS reads `\limsup_{n \to \infty} a_n` as *"backslash lim sup underscore…"*
— or silence. That breaks paper audiobooks, screen-reader access to lecture
notes, and video voice-overs. latex2voice fixes it at the **text layer**, so
the TTS engine stays hot-swappable (Kokoro, Piper, commercial APIs — anything
that reads English).

## How it works

```
LaTeX ──▶ TN module (Qwen3-4B + LoRA) ──▶ spoken English ──▶ any TTS ──▶ audio
              ▲ fully synthetic training data
```

Why a TN front-end instead of fine-tuning TTS: math→words is a pure
text-to-text task. A small LLM with a LoRA solves it with no audio data,
no per-engine retraining, ~40 minutes of training on a consumer GPU.

## Results

Training: 6,272 fully synthetic pairs (difficulty levels L1–L4 + 400
plain-text guard samples). Zero human annotations.

| Check | Result |
|---|---|
| Symbol coverage audit (143 cases, incl. full amsmath `\DeclareMathOperator` set) | rules 143/143 · model 133/143 (rest = variant readings) |
| Held-out formulas L1–L3 (exact match) | 75/75 |
| Held-out formulas L4 (LLM-judged) | 25/25 |
| Guard: plain text passes through unchanged | 30/30 |
| End-to-end round-trip (TTS → Whisper → judge) | spot-checked, see demo |

## Try it

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "petehu0803/latex2voice")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

SYSTEM = ("You convert LaTeX math into fluent English spoken form (what a person "
          "reading the formula aloud would say). If the text contains no LaTeX math, "
          "output it unchanged. Output ONLY the resulting text.")
USER = "Convert this LaTeX to spoken English:\n{latex}"

prompt = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": USER.format(latex=r"\limsup_{n \to \infty} a_n")}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
out = model.generate(ids, max_new_tokens=120, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip())
# the limit superior as n goes to infinity of a sub n
```

> `enable_thinking=False` is required — Qwen3-4B is a thinking model and the
> adapter is trained for direct answers.

~2 GB RAM runtime, ~3–6 s per formula on CPU only.

## Repo layout

```
src/latextts/
  rules_en.py       # rule-based LaTeX→English library (training gold, 143/143 audited)
  formula_gen.py    # formula generators, difficulty levels L1–L4
  tn_data.py        # synthetic TN training data builder
  train/eval via scripts/
scripts/
  train_tn.py       # hand-written SFT loop (LoRA r=32, q/k/v/o)
  eval_tn.py        # held-out eval: exact match + LLM judge
  _coverage_audit.py# 143-case symbol coverage audit (rules layer)
  _model_audit.py   # model-layer audit with variant tolerance
  _gen_demo.py      # demo audio generation
  _gen_baseline.py  # stock-TTS baseline clips (the "Before" half)
data/               # generated formulas + training sets (jsonl)
demo/               # static demo page + audio (deployed at demo.latex2voice.dev)
```

## Known limitations

- English only; Chinese planned.
- Known gaps: `\vec{v}` occasionally hallucinated ("velocity"), `\nabla \cdot`
  not reliably read as "divergence", `\dots` sometimes output literally,
  `\odot` not learned.
- Single-letter formulas are homophone-ambiguous when spoken ("c" = "see") —
  a property of speech, not a bug.

## Links

- 🎧 Live demo: <https://demo.latex2voice.dev>
- 🤗 Weights: <https://huggingface.co/petehu0803/latex2voice>
