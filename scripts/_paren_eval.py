"""Paren grouping held-out eval for v3."""
import random
import sys
import time

sys.path.insert(0, ".")
import torch  # noqa: E402

from src.latextts.eval_roundtrip import normalize_en  # noqa: E402
from src.latextts.rules_en import latex_to_spoken  # noqa: E402
from src.latextts.tn_data import SYSTEM_PROMPT, USER_TMPL  # noqa: E402
from src.latextts.tn_infer import get_model  # noqa: E402

t0 = time.time()
model, tok = get_model("cpu")
print(f"model loaded {time.time() - t0:.0f}s", flush=True)

rng = random.Random(99)
syms = ["a", "b", "x", "y", "u", "v"]
cases = []
while len(cases) < 10:
    a, b = rng.sample(syms, 2)
    op = rng.choice(["+", "-"])
    p = rng.choice(["2", "3", "{n}"])
    tmpl = rng.choice(
        [
            r"({a}{op}{b})^{p}",
            r"\left( {a} {op} {b} \right)^{p}",
            r"2({a}{op}{b})^{p}",
            r"\frac{{({a}{op}{b})^{{{p}}}}}{{4}}",
        ]
    )
    c = tmpl.format(a=a, b=b, op=op, p=p)
    if c not in cases:
        cases.append(c)

ok = 0
for i, latex in enumerate(cases):
    t1 = time.time()
    gold = latex_to_spoken(latex)
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
        out = model.generate(ids, max_new_tokens=120, do_sample=False)
    pred = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    m = normalize_en(gold) == normalize_en(pred)
    ok += m
    print(f"{i+1}/10 {time.time()-t1:.1f}s {'OK' if m else 'MISS'} {latex!r}", flush=True)
    if not m:
        print(f"    gold={gold!r}", flush=True)
        print(f"    pred={pred!r}", flush=True)
print(f"paren heldout: {ok}/10")
