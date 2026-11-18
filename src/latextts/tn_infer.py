"""TN inference: load Qwen3-4B + trained LoRA, convert LaTeX -> spoken English."""
from __future__ import annotations

import os
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ.get("L2V_BASE", r"H:\SHARE\latex-tts\models\Qwen3-4B"))
LORA_DIR = Path(os.environ.get("L2V_LORA", ROOT / "models" / "tn-lora"))

SYSTEM_PROMPT = (
    "You convert LaTeX math into fluent English spoken form (what a person "
    "reading the formula aloud would say). Output ONLY the spoken text."
)
USER_TMPL = "Convert this LaTeX to spoken English:\n{latex}"

_model = None
_tok = None


def get_model(device: str = "cuda"):
    global _model, _tok
    if _model is None:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager"
        )
        _model = PeftModel.from_pretrained(_model, LORA_DIR)
        _model.eval()
    return _model, _tok


def tn(latex: str, device: str = "cuda", max_new_tokens: int = 120) -> str:
    model, tok = get_model(device)
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
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
