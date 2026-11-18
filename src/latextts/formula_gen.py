"""Formula generator with difficulty levels L1-L4. Emits jsonl."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.latextts.rules_en import GREEK_POOL, SYMBOL_POOL, latex_to_spoken

R = random.Random


# ---------- L1: single symbol / single operation ----------

def gen_l1(rng: R) -> str:
    kind = rng.choice(["power", "frac", "sqrt", "greek", "simple_eq", "trig",
                       "coef_term", "two_term", "coef_frac", "neg_power",
                       "paren_group", "paren_frac", "named_fn", "sub_paren",
                       "math_fonts"])
    x, y = rng.sample(SYMBOL_POOL, 2)
    coef = rng.choice(["", "2", "3", "4", "\\frac{1}{2}", "\\frac{3}{4}", "-"])
    if kind == "named_fn":
        fn = rng.choice([r"\mathrm{softmax}", r"\mathrm{sigmoid}", r"\mathrm{relu}",
                         r"\mathrm{Concat}", r"\mathrm{Attention}", r"\mathrm{LayerNorm}",
                         r"\operatorname{softmax}", r"\operatorname{sigmoid}",
                         r"\operatorname{LayerNorm}", r"\ker", r"\hom", r"\lg",
                         r"\coth", r"\tr", r"\rank", r"\diag"])
        inner = rng.choice([
            rf"\left(\frac{{QK^\top}}{{\sqrt{{d_k}}}}\right)",
            rf"({rng.choice(SYMBOL_POOL)}, {rng.choice(SYMBOL_POOL)})",
            rf"\left({rng.randint(0, 1)}, {rng.choice(SYMBOL_POOL)}\right)",
        ])
        return rf"{fn}{inner}"
    if kind == "sub_paren":
        base = rng.choice(["PE", "x", "W", "h"])
        return rf"{base}_{{({rng.choice(SYMBOL_POOL)}, {rng.randint(1, 9)}{rng.choice(['i', 'j', ''])})}}"
    if kind == "math_fonts":
        letter = rng.choice("LMHT")
        return rng.choice([rf"\mathcal{{{letter}}}", rf"\mathbb{{{rng.choice(['R', 'E'])}}}",
                           rf"\mathbf{{{rng.choice(SYMBOL_POOL)}}}"])
    x, y = rng.sample(SYMBOL_POOL, 2)
    coef = rng.choice(["", "2", "3", "4", "\\frac{1}{2}", "\\frac{3}{4}", "-"])
    if kind == "paren_group":
        op = rng.choice(["+", "-", " + ", " - "])
        a, b = rng.sample(SYMBOL_POOL, 2)
        lead = rng.choice(["", "2", "3", "4", "-", "\\frac{1}{2}"])
        return rf"{lead}({a}{op}{b})^{rng.choice(['2', '3', '{{n}}', '-1'])}"
    if kind == "paren_frac":
        a, b = rng.sample(SYMBOL_POOL, 2)
        num = rng.choice([f"{a}+{b}", f"{a}-{b}"])
        expo = rng.choice(["", "^2", "^3", "^{n}"])
        return rf"\frac{{({num}){expo}}}{{{rng.choice([str(rng.randint(2, 9)), y])}}}"
    x, y = rng.sample(SYMBOL_POOL, 2)
    coef = rng.choice(["", "2", "3", "4", "\\frac{1}{2}", "\\frac{3}{4}", "-"])
    if kind == "power":
        base = rng.choice(SYMBOL_POOL + GREEK_POOL)
        p = rng.choice(["2", "3", "-1", "{n}", "{2n+1}", "{k+1}", "{i}"])
        return f"{base}^{p}"
    if kind == "neg_power":
        return rf"{rng.choice(SYMBOL_POOL + GREEK_POOL)}^{{-{rng.randint(2, 5)}}}"
    if kind == "coef_term":
        return f"{coef}{rng.choice(SYMBOL_POOL + GREEK_POOL)}^{rng.choice(['2', '3', 'n'])}"
    if kind == "two_term":
        t1 = rng.choice([f"{rng.choice(SYMBOL_POOL)}^2", rf"\frac{{1}}{{{rng.choice(SYMBOL_POOL)}}}", rng.choice(GREEK_POOL)])
        op = rng.choice([" + ", " - ", " \\cdot ", " \\times ", " \\propto ", " \\pm "])
        t2 = rng.choice([rng.choice(SYMBOL_POOL), rng.choice(GREEK_POOL), rf"\sqrt{{{rng.choice(SYMBOL_POOL)}}}"])
        return f"{t1}{op}{t2}"
    if kind == "coef_frac":
        return rf"{coef}\frac{{{x}}}{{{rng.choice(SYMBOL_POOL)}}}"
    if kind == "frac":
        return rf"\frac{{{x}}}{{{y}}}"
    if kind == "sqrt":
        return rf"\sqrt{{{rng.choice(SYMBOL_POOL)}}}"
    if kind == "greek":
        return rng.choice(GREEK_POOL)
    if kind == "simple_eq":
        return f"{x} + {y} = {rng.randint(1, 20)}"
    return rng.choice([r"\sin x", r"\cos \theta", r"\ln x", r"\log_2 n", r"\sin(2\theta)", r"\tan x"])


# ---------- L2: combined structures ----------

def gen_l2(rng: R) -> str:
    kind = rng.choice(["sum", "int", "deriv", "chain", "exp_log", "lim_simple",
                       "double_frac", "trig_comp", "prod_chain", "bound_variants"])
    x, y = rng.sample(SYMBOL_POOL, 2)
    if kind == "sum":
        body = rng.choice([f"{rng.choice(SYMBOL_POOL)}^{2}", rf"\frac{{1}}{{{rng.choice(SYMBOL_POOL)}}}", "k",
                           f"{rng.randint(2, 5)}k", rf"\sqrt{{{rng.choice(SYMBOL_POOL)}}}"])
        return rf"\sum_{{k=1}}^{{n}} {body}"
    if kind == "int":
        inner = rng.choice([f"{rng.choice(SYMBOL_POOL)}^2", rf"\frac{{1}}{{{rng.choice(SYMBOL_POOL)}}}", rf"\ln {rng.choice(SYMBOL_POOL)}",
                            rf"\sin {rng.choice(SYMBOL_POOL)}", rf"{rng.choice(SYMBOL_POOL)}^{{3}}"])
        lo = rng.choice(["0", "-\\infty", "1"])
        hi = rng.choice(["\\infty", "\\pi", "{n}"])
        return rf"\int_{{{lo}}}^{{{hi}}} {inner}\, d{rng.choice(SYMBOL_POOL)}"
    if kind == "deriv":
        f = rng.choice(["f", "g"])
        return rf"{f}'(x) = {rng.randint(1, 9)}x + {rng.randint(1, 9)}"
    if kind == "chain":
        base = rng.choice(SYMBOL_POOL + GREEK_POOL)
        return rng.choice([
            rf"\frac{{1}}{{2}} {base}^{{2}} + \frac{{1}}{{{rng.choice(SYMBOL_POOL)}}}",
            rf"{rng.randint(2, 5)} {base}^{{2}} - {rng.randint(1, 9)}{rng.choice(SYMBOL_POOL)}",
            rf"\frac{{{rng.choice(SYMBOL_POOL)}^{{2}}}}{{{rng.randint(2, 5)}}} + {rng.choice(GREEK_POOL)}",
        ])
    if kind == "exp_log":
        return rng.choice([r"e^{i\pi} + 1 = 0", rf"\ln\left({rng.choice(SYMBOL_POOL)}^2 + 1\right)", rf"\exp\left(-{rng.choice(SYMBOL_POOL)}^{{2}}\right)",
                           rf"\log_{{{rng.randint(2, 5)}}} {rng.choice(SYMBOL_POOL)}", rf"e^{{-{rng.choice(SYMBOL_POOL)}}} {rng.choice(GREEK_POOL)}"])
    if kind == "double_frac":
        return rf"\frac{{\frac{{1}}{{{x}}}}}{{{y} + 1}}"
    if kind == "trig_comp":
        return rng.choice([rf"\sin\left({rng.choice(GREEK_POOL)} + {rng.choice(GREEK_POOL)}\right)",
                           rf"\cos(2{rng.choice(GREEK_POOL)})",
                           rf"\arctan\left(\frac{{1}}{{{rng.choice(SYMBOL_POOL)}}}\right)"])
    if kind == "prod_chain":
        return rf"{rng.choice(SYMBOL_POOL)}_{rng.randint(1, 5)} {rng.choice(GREEK_POOL)}^{rng.randint(2, 3)}"
    if kind == "bound_variants":
        return rf"\sum_{{i={rng.randint(0, 2)}}}^{{{rng.choice(['N', '2n', 'm'])}}} \frac{{1}}{{i + {rng.randint(1, 3)}}}"
    return rf"\lim_{{n \to \infty}} \frac{{1}}{{n}} = 0"


# ---------- L3: complex structures ----------

def gen_l3(rng: R) -> str:
    kind = rng.choice(["matrix", "cases", "norm", "limit_def", "partial"])
    if kind == "matrix":
        a, b, c, d = (rng.randint(1, 9) for _ in range(4))
        return rf"\begin{{pmatrix}} {a} & {b} \\ {c} & {d} \end{{pmatrix}}"
    if kind == "cases":
        k = rng.randint(1, 9)
        return rf"f(x) = \begin{{cases}} {k} & \text{{if }} x > 0 \\ 0 & \text{{otherwise}} \end{{cases}}"
    if kind == "norm":
        v = rng.choice(SYMBOL_POOL)
        return rf"\left\| {v} \right\|^2 = {v}^\top {v}"
    if kind == "limit_def":
        h = "h"
        return rf"f'(a) = \lim_{{{h} \to 0}} \frac{{f(a + {h}) - f(a)}}{{{h}}}"
    return rf"\frac{{\partial L}}{{\partial {rng.choice(SYMBOL_POOL)}}} = \sum_{{i}} \delta_i \, {rng.choice(SYMBOL_POOL)}_i"


# ---------- L4: formula embedded in English sentence ----------

_SENTENCES = [
    "where {f} denotes the {desc}, we obtain {eq}.",
    "It follows that {eq}, since {f} is the {desc}.",
    "By substituting {eq} into the equation, the {desc} vanishes.",
    "Note that {eq} holds whenever the {desc} is positive.",
    "This yields {eq} for the {desc}.",
]
_DESCS = ["learning rate", "variance", "loss function", "gradient", "eigenvalue", "regularizer", "kernel", "posterior"]


def gen_l4(rng: R) -> dict:
    core = rng.choice([gen_l1(rng), gen_l2(rng), gen_l3(rng)])
    tmpl = rng.choice(_SENTENCES)
    text = tmpl.format(eq=core, f=rng.choice(GREEK_POOL + SYMBOL_POOL), desc=rng.choice(_DESCS))
    return {"text": text, "core": core}


GENS = {1: gen_l1, 2: gen_l2, 3: gen_l3, 4: gen_l4}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", required=True, choices=["L1", "L2", "L3", "L4"])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rng = R(args.seed)
    lvl = int(args.level[1])
    out = Path(args.out) if args.out else Path(__file__).resolve().parents[2] / "data" / "formulas" / f"{args.level}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    rows = []
    while len(rows) < args.n:
        if lvl == 4:
            item = gen_l4(rng)
            key = item["text"]
        else:
            item = {"text": GENS[lvl](rng), "core": None}
            key = item["text"]
        if key in seen:
            continue
        seen.add(key)
        spoken = latex_to_spoken(item["core"] if item["core"] else item["text"])
        rows.append({"id": f"{args.level}-{len(rows):04d}", "level": args.level, "latex": key, "spoken_rules": spoken})
        if lvl == 4:
            rows[-1]["core"] = item["core"]

    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} formulas -> {out}")


if __name__ == "__main__":
    main()
