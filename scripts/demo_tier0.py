"""Tier-0 demo: rules TN -> TTS -> ASR -> round-trip match.

Usage: python scripts/demo_tier0.py --level L1 --n 5
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.latextts.eval_roundtrip import text_match  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L1")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--skip-tts", action="store_true", help="only test TN text vs ASR-free match (no audio)")
    args = ap.parse_args()

    formula_file = ROOT / "data" / "formulas" / f"{args.level}.jsonl"
    if not formula_file.exists():
        subprocess.run(
            [sys.executable, "-m", "src.latextts.formula_gen", "--level", args.level, "--n", str(max(args.n, 50))],
            cwd=ROOT, check=True,
        )
    rows = [json.loads(l) for l in formula_file.open(encoding="utf-8")][: args.n]

    from src.latextts.asr_whisper import transcribe  # noqa: E402
    from src.latextts.tts_kokoro import synthesize  # noqa: E402

    audio_dir = ROOT / "data" / "audio" / f"demo_{args.level}"
    audio_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for row in rows:
        spoken = row["spoken_rules"]
        if args.skip_tts:
            transcript = spoken  # degenerate: rules vs itself
        else:
            wav = str(audio_dir / f"{row['id']}.wav")
            synthesize(spoken, wav)
            transcript = transcribe(wav)
        m = text_match(spoken, transcript)
        results.append({"id": row["id"], "latex": row["latex"], "spoken": spoken, "transcript": transcript, **m})
        status = "OK " if m["exact"] else "MISS"
        print(f"[{status}] {row['latex']!r}")
        print(f"       spoken    : {spoken}")
        print(f"       transcript: {transcript}   (sim={m['similarity']})")

    exact = sum(r["exact"] for r in results)
    sim = sum(r["similarity"] for r in results) / len(results)
    print(f"\nexact {exact}/{len(results)}   mean similarity {sim:.3f}")

    out = ROOT / "logs" / f"demo_tier0_{args.level}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"detail -> {out}")


if __name__ == "__main__":
    main()
