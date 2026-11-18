"""Model-level audit: run the trained TN model over all audit cases."""
import re
import sys
from collections import defaultdict

sys.path.insert(0, ".")
import torch  # noqa: E402

from src.latextts.tn_data import SYSTEM_PROMPT, USER_TMPL  # noqa: E402
from src.latextts.tn_infer import get_model  # noqa: E402
import importlib  # noqa: E402

audit = importlib.import_module("scripts._coverage_audit") if False else None

# import CASES from the audit script without running its main block
import types  # noqa: E402

src = open("scripts/_coverage_audit.py", encoding="utf-8").read()
src = src.split("stats = defaultdict")[0]  # keep only CASES definition part
mod = types.ModuleType("cases")
mod.__dict__["__name__"] = "cases"
code = src.replace('sys.path.insert(0, ".")', "").replace(
    "from src.latextts.rules_en import latex_to_spoken as L", "L = None"
)
exec(compile(code, "cases", "exec"), mod.__dict__)
CASES = mod.CASES

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
        out = model.generate(ids, max_new_tokens=80, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


stats = defaultdict(lambda: [0, 0])
fails = []
variants = {
    r"\Gamma": "gamma", r"\Delta": "delta", r"\Theta": "theta",
    r"\Lambda": "lambda", r"\Xi": "xi", r"\Pi": "pi|symbol",
    r"\Sigma": "sigma|symbol", r"\Phi": "phi", r"\Psi": "psi",
    r"\Omega": "omega", r"\chi": "chi|kai",
    r"a \in S": "element of", r"a \notin S": "not.*element",
    r"x \equiv y": "congruent|identically",
    r"f \circ g": "circle|composed",
    r"\binom{n}{k}": "binomial|choose",
    r"n \bmod 3": "mod(ulo)?",
    r"\oint_C": "integral over",
    r"\csc x": "cosec", ("fn", r"\csc x")[1]: "cosec",
    r"\cosh x": "cosh",
    r"\vec{v}": "vector", r"\mathrm{kg}": "kilogram",
    r"\mathcal{L}": "calligraphic|mathcal",  # calligraphic accepted; mathcal leak = fail
    r"\nabla \cdot F": "divergence",
    r"\sinh x": ["hyperbolic sine|cosh"],
}
for i, (cat, latex, expect) in enumerate(CASES):
    pred = tn(latex)
    v = variants.get(latex, [])
    if isinstance(v, str):
        v = [v]
    accepts = [expect] + v
    ok = any(all(re.search(p, pred, re.I) for p in a.split("|")) for a in accepts)
    stats[cat][1] += 1
    if ok:
        stats[cat][0] += 1
    else:
        fails.append((cat, latex, expect, pred))
    if (i + 1) % 20 == 0:
        print(f"...{i+1}/{len(CASES)}", flush=True)

print(f"\n{'category':12} {'model-pass':>10}")
for cat, (okc, tot) in stats.items():
    print(f"{cat:12} {okc:>3}/{tot:<4}")
total_ok = sum(v[0] for v in stats.values())
total = sum(v[1] for v in stats.values())
print(f"\nMODEL TOTAL {total_ok}/{total}\n")
print("=== MODEL GAPS ===")
for cat, latex, expect, pred in fails:
    print(f"[{cat}] {latex!r}\n   want: {expect}\n   pred: {pred!r}")
