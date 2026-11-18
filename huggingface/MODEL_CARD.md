---
license: apache-2.0
base_model: Qwen/Qwen3-4B
tags:
- latex
- text-normalization
- tts-frontend
- math
- asr
language:
- en
pipeline_tag: text2text-generation
library_name: peft
---

# latex-tts-tn: LaTeX → Spoken English for TTS front-ends

A LoRA adapter for **Qwen3-4B** that converts LaTeX math into fluent English
spoken form — the text a person reading the formula aloud would say. It is a
drop-in **text-normalization front-end for TTS**: most TTS engines read plain
English fluently but choke on `x^2 + y^2`; this module rewrites the math into
words first.

```
"x^2 + y^2 = z^2"  →  "x squared plus y squared equals z squared"
"\int_0^\infty \frac{1}{x^2} dx"  →  "the integral from 0 to infinity of 1 over x squared, dx"
"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"  →  "the 2 by 2 matrix with, first row 1, 2, and second row 3, 4"
```

## Why a TN front-end instead of fine-tuning TTS

TTS engines already speak English well; what they lack is LaTeX literacy.
Converting math to words is a pure text-to-text task, so a small LLM with a
LoRA solves it — no audio data, no TTS retraining, fully offline.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B", torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "YOUR_USERNAME/latex-tts-tn")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

SYSTEM = ("You convert LaTeX math into fluent English spoken form (what a person "
          "reading the formula aloud would say). If the text contains no LaTeX math, "
          "output it unchanged. Output ONLY the resulting text.")
USER = "Convert this LaTeX to spoken English:\n{latex}"

latex = r"\int_0^\infty \frac{1}{x^2}\, dx"
prompt = tok.apply_chat_template(
    [{"role": "system", "content": SYSTEM},
     {"role": "user", "content": USER.format(latex=latex)}],
    tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
out = model.generate(ids, max_new_tokens=120, do_sample=False)
print(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip())
# the integral from 0 to infinity of 1 over x squared, dx
```

`enable_thinking=False` is required — the base model is a thinking model and
the adapter is trained for direct answers.

## Training

- Base: Qwen3-4B (Apache 2.0)
- LoRA r=32 on q/k/v/o projections (23.6M trainable params, 0.58%)
- Data: 6,272 fully synthetic pairs — L1 (single operations), L2 (sums,
  integrals, derivatives), L3 (matrices, piecewise functions, norms), L4
  (formulas embedded in English sentences), plus 400 identity samples of
  plain text (guard behavior: text without math passes through unchanged)
- Parenthesized expressions under an exponent are read with explicit
  grouping ("the quantity a plus b, squared") so `(a+b)^2` is never
  misread as `a + b^2`

## Evaluation

Three-layer protocol (all reproducible from the repo):

1. **Held-out formulas** (seed-separated from training): L1–L4 100/100
   accepted — L1–L3 by exact text match, L4 by LLM-judged semantic
   equivalence (local Qwen3-8B judge allows legitimate variant readings).
2. **Round-trip**: TTS (Kokoro-82M) → Whisper ASR → semantic match,
   37/40 end-to-end; the 3 failures are ASR homophone errors ("c over u"
   heard as "see over you"), not reading errors.
3. **Guard**: plain text without math passes through unchanged, 30/30.

## Known limitations

- English only; canonical readings are the generators' mainstream choices.
- Coverage follows the training distribution; unusual LaTeX (accents like
  `\hat`, deep exponent nesting) is not exhaustively verified.
- Single-letter formulas are homophone-ambiguous when spoken ("c" = "see");
  this is a property of speech, not a bug.

## Links

- Code, data generators and evaluation scripts: (repo URL)
