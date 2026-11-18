"""LLM judge: decide whether a TN output is a faithful reading of the LaTeX,
allowing legitimate variant readings. Uses local Qwen3-8B (chat) as judge.

Used only for rows where exact text match fails — cheap-first discipline.
"""
from __future__ import annotations

from pathlib import Path

JUDGE_MODEL_DIR = Path(r"H:\SHARE\arch-transfer\models\Qwen3-8B")

JUDGE_PROMPT = """You are judging whether a spoken reading of a math formula is faithful.

Original LaTeX:
{latex}

Spoken reading:
{spoken}

Accept the reading if it conveys the same mathematical content, even with
different wording (e.g. "goes to" vs "approaches", "base 2" vs "sub 2",
"minus" vs "negative"). Reject it if it changes, drops, or adds mathematical
content.

Answer with exactly one word: ACCEPT or REJECT."""

_model = None
_tok = None


def get_model(device: str = "cuda"):
    global _model, _tok
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(JUDGE_MODEL_DIR)
        _model = AutoModelForCausalLM.from_pretrained(
            JUDGE_MODEL_DIR, torch_dtype=torch.bfloat16, device_map=device, attn_implementation="eager"
        )
        _model.eval()
    return _model, _tok


def judge(latex: str, spoken: str, device: str = "cuda") -> bool:
    model, tok = get_model(device)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": JUDGE_PROMPT.format(latex=latex, spoken=spoken)}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    import torch

    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=8, do_sample=False)
    answer = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip().upper()
    return answer.startswith("ACCEPT")
