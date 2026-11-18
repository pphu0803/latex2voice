"""Round-trip evaluation: spoken text vs ASR transcript.

Layer 1: normalized text similarity (exact / cer / token match).
Layer 2 (optional): sympy semantic equivalence for parseable formulas.
"""
from __future__ import annotations

import re


_HOMOPHONES = {
    "sign": "sine",  # whisper hears 'sine' as 'sign'
}


def normalize_en(text: str) -> str:
    t = text.lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return " ".join(_HOMOPHONES.get(tok, tok) for tok in t.split())


def text_match(reference: str, transcript: str) -> dict:
    ref, hyp = normalize_en(reference), normalize_en(transcript)
    ref_tokens, hyp_tokens = ref.split(), hyp.split()
    from difflib import SequenceMatcher

    sm = SequenceMatcher(None, ref, hyp)
    return {
        "exact": ref == hyp,
        "similarity": round(sm.ratio(), 4),
        "ref_tokens": len(ref_tokens),
        "hyp_tokens": len(hyp_tokens),
    }


def sympy_equivalent(latex: str, transcript: str) -> bool | None:
    """Best-effort sympy check for simple L1 formulas. None = cannot parse."""
    try:
        from sympy.parsing.latex import parse_latex
    except Exception:
        return None
    try:
        # crude transcript->latex reverse map, only for trivial patterns
        rev = transcript.lower()
        rev = rev.replace(" squared", "**2").replace(" cubed", "**3")
        rev = rev.replace(" plus", "+").replace(" minus", "-")
        rev = rev.replace(" times", "*").replace(" equals", "=")
        exprs = parse_latex(rev)
        return bool(parse_latex(latex) == exprs)
    except Exception:
        return None
