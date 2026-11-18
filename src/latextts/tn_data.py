"""Build TN training data v2: L1-L4 + identity guard samples.

Guard samples: plain English sentences with no LaTeX -> output unchanged verbatim.
These teach the model pass-through behavior; evaluated by the guard metric.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = (
    "You convert LaTeX math into fluent English spoken form (what a person "
    "reading the formula aloud would say). If the text contains no LaTeX math, "
    "output it unchanged. Output ONLY the resulting text."
)

USER_TMPL = "Convert this LaTeX to spoken English:\n{latex}"
USER_TMPL_PLAIN = "Convert this text to spoken English (output unchanged if no LaTeX math):\n{text}"


def _plain_sentences(n: int, rng: random.Random) -> list[str]:
    """Sentences guaranteed to contain no LaTeX math (guard set)."""
    subjects = ["The model", "This result", "Our method", "The experiment", "Figure three", "Table two",
                "The author", "Section four", "The proof", "The dataset", "The training run", "The baseline"]
    verbs = ["shows", "suggests", "improves", "requires", "avoids", "measures", "reports", "compares"]
    tails = ["a clear improvement.", "a small error.", "the expected behavior.", "a simple idea.",
             "an important detail.", "a strong result.", "two different cases.", "a well known fact."]
    out, seen = [], set()
    while len(out) < n:
        s = f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(tails)}"
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def build_rows(args, rng: random.Random) -> list[dict]:
    from src.latextts.formula_gen import GENS  # noqa: E402
    from src.latextts.rules_en import latex_to_spoken  # noqa: E402

    rows = []
    n_per = args.n_per_level
    for lvl in (1, 2, 3, 4):
        seen = set()
        made = 0
        attempts = 0
        while made < n_per:
            attempts += 1
            if attempts > 200 * n_per:
                raise RuntimeError(f"L{lvl}: generator space exhausted at {made}/{n_per}")
            if lvl == 4:
                core = rng.choice([GENS[1](rng), GENS[2](rng), GENS[3](rng)])
                core_spoken = latex_to_spoken(core)
                # embed the spoken core into a sentence: the "sentence with formula" is the input,
                # the spoken sentence is the target
                tmpl = rng.choice(
                    [
                        "where {core} holds, the gradient vanishes.",
                        "It follows that {core} for every layer.",
                        "We obtain {core} after substituting the learning rate.",
                        "This holds because {core} in all our runs.",
                        "Note that {core} whenever the loss decreases.",
                    ]
                )
                latex_input = tmpl.format(core=core)
                target = tmpl.format(core=core_spoken)
            else:
                latex_input = GENS[lvl](rng)
                target = latex_to_spoken(latex_input)
            if latex_input in seen:
                continue
            seen.add(latex_input)
            rows.append(
                {
                    "level": f"L{lvl}",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_TMPL.format(latex=latex_input)},
                        {"role": "assistant", "content": target},
                    ],
                }
            )
            made += 1
        print(f"L{lvl}: {n_per} pairs")

    # guard identity samples
    for s in _plain_sentences(args.n_guard, rng):
        rows.append(
            {
                "level": "guard",
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TMPL_PLAIN.format(text=s)},
                    {"role": "assistant", "content": s},
                ],
            }
        )
    print(f"guard: {args.n_guard} identity pairs")
    rng.shuffle(rows)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-level", type=int, default=1500)
    ap.add_argument("--n-guard", type=int, default=400)
    ap.add_argument("--val-ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data = build_rows(args, rng)
    n_val = max(1, int(len(data) * args.val_ratio))
    train_dir = ROOT / "data" / "tn_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in data[n_val:]), encoding="utf-8"
    )
    (train_dir / "val.jsonl").write_text(
        "\n".join(json.dumps(d, ensure_ascii=False) for d in data[:n_val]), encoding="utf-8"
    )
    print(f"train {len(data) - n_val}, val {n_val} -> {train_dir}")


if __name__ == "__main__":
    main()
