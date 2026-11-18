"""End-to-end batch eval: TN -> Kokoro TTS -> Whisper ASR -> judged match.

Samples across L1-L4. TTS is CPU (Kokoro); ASR is whisper-large-v3-turbo.

Usage:
  python -X utf8 scripts/eval_e2e.py --n 10
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
    ap.add_argument("--n", type=int, default=10, help="samples per level")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(ROOT / "logs" / "eval_e2e.json"))
    args = ap.parse_args()

    from src.latextts.asr_whisper import transcribe  # noqa: E402
    from src.latextts.eval_roundtrip import normalize_en  # noqa: E402
    from src.latextts.formula_gen import GENS  # noqa: E402
    from src.latextts.judge import judge  # noqa: E402
    from src.latextts.rules_en import latex_to_spoken  # noqa: E402
    from src.latextts.tn_infer import get_model  # noqa: E402
    from src.latextts.tts_kokoro import synthesize  # noqa: E402

    model, tok = get_model(args.device)
    rng = random.Random(args.seed)
    audio_dir = ROOT / "data" / "audio" / "e2e_v2"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # warm up ASR once
    warm = audio_dir / "_warm.wav"
    if not warm.exists():
        synthesize("testing one two three", str(warm))
    transcribe(str(warm))

    results = {}
    for lvl in (1, 2, 3, 4):
        seen, rows = set(), []
        while len(rows) < args.n:
            item = GENS[lvl](rng)
            latex = item["text"] if isinstance(item, dict) else item
            if latex in seen:
                continue
            seen.add(latex)
            rows.append(latex)
        ok = fail = 0
        fails = []
        t0 = time.time()
        for i, latex in enumerate(rows):
            spoken = tn_predict(model, tok, latex)
            wav = str(audio_dir / f"L{lvl}_{i:02d}.wav")
            synthesize(spoken, wav)
            transcript = transcribe(wav)
            if normalize_en(spoken) == normalize_en(transcript):
                ok += 1  # ASR round trip clean
                continue
            # ASR heard something different; judge if the transcript is still faithful
            if judge(latex=latex, spoken=transcript, device=args.device):
                ok += 1
            else:
                fail += 1
                fails.append({"latex": latex, "spoken": spoken, "transcript": transcript})
        dt = time.time() - t0
        results[f"L{lvl}"] = {"n": args.n, "ok": ok, "fail": fail,
                              "rate": round(ok / args.n, 4), "seconds": round(dt, 1), "fails": fails}
        print(f"L{lvl} e2e: {ok}/{args.n} pass  ({dt:.0f}s, {dt / args.n:.1f}s/sample)", flush=True)
        for f in fails:
            print(f"  FAIL {f['latex']!r}\n    spoken={f['spoken']!r}\n    ASR   ={f['transcript']!r}", flush=True)

    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"detail -> {args.out}")


def tn_predict(model, tok, latex: str) -> str:
    import torch

    from src.latextts.tn_data import SYSTEM_PROMPT, USER_TMPL

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
        out = model.generate(ids, max_new_tokens=150, do_sample=False)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()


if __name__ == "__main__":
    main()
