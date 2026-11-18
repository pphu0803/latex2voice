"""Full TN evaluation: held-out formulas L1-L4 + guard metric.

Cheap-first: exact text match first; only misses go to the LLM judge.
Guard: plain-text inputs must pass through unchanged (exact).

Usage:
  python -X utf8 scripts/eval_tn.py --n 25
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="held-out formulas per level")
    ap.add_argument("--guard-n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "logs" / "eval_tn_v2.json"))
    args = ap.parse_args()

    from src.latextts.eval_roundtrip import normalize_en
    from src.latextts.formula_gen import GENS  # noqa: E402
    from src.latextts.judge import judge  # noqa: E402
    from src.latextts.rules_en import latex_to_spoken  # noqa: E402
    from src.latextts.tn_data import SYSTEM_PROMPT, USER_TMPL, USER_TMPL_PLAIN  # noqa: E402
    from src.latextts.tn_infer import get_model  # noqa: E402

    model, tok = get_model(args.device)
    rng = random.Random(args.seed)

    def tn_raw(user_content: str) -> str:
        prompt = tok.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        import torch

        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=150, do_sample=False)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    results = {"levels": {}, "guard": None, "judged": []}

    # ---- formula levels ----
    for lvl in (1, 2, 3, 4):
        seen, rows = set(), []
        while len(rows) < args.n:
            item = GENS[lvl](rng)
            latex = item["text"] if isinstance(item, dict) else item
            if latex in seen:
                continue
            seen.add(latex)
            rows.append(latex)
        exact = judge_pass = judge_fail = 0
        misses = []
        t0 = time.time()
        for latex in rows:
            gold = latex_to_spoken(latex)
            pred = tn_raw(USER_TMPL.format(latex=latex))
            if normalize_en(gold) == normalize_en(pred):
                exact += 1
                continue
            ok = judge(latex=latex, spoken=pred, device=args.device)
            if ok:
                judge_pass += 1
            else:
                judge_fail += 1
                misses.append({"latex": latex, "gold": gold, "pred": pred})
        dt = time.time() - t0
        accepted = exact + judge_pass
        results["levels"][f"L{lvl}"] = {
            "n": args.n, "exact": exact, "judge_pass": judge_pass,
            "judge_fail": judge_fail, "accepted": accepted,
            "accept_rate": round(accepted / args.n, 4), "seconds": round(dt, 1),
            "misses": misses,
        }
        print(
            f"L{lvl}: accepted {accepted}/{args.n} (exact {exact}, judge {judge_pass}, reject {judge_fail})  {dt:.0f}s",
            flush=True,
        )
        for m in misses:
            print(f"  REJECT {m['latex']!r}\n    pred={m['pred']!r}", flush=True)

    # ---- guard: plain text must pass through unchanged ----
    from src.latextts.tn_data import _plain_sentences  # noqa: E402

    guard_rows = _plain_sentences(args.guard_n, rng)
    guard_exact = 0
    guard_fails = []
    for s in guard_rows:
        pred = tn_raw(USER_TMPL_PLAIN.format(text=s))
        if pred == s:
            guard_exact += 1
        else:
            guard_fails.append({"input": s, "pred": pred})
    results["guard"] = {
        "n": args.guard_n, "exact": guard_exact,
        "rate": round(guard_exact / args.guard_n, 4), "fails": guard_fails,
    }
    print(f"guard: {guard_exact}/{args.guard_n} unchanged", flush=True)
    for f in guard_fails[:5]:
        print(f"  GUARD-FAIL in={f['input']!r} pred={f['pred']!r}", flush=True)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"detail -> {args.out}")


if __name__ == "__main__":
    main()
