"""LoRA fine-tune Qwen3-4B as TN module (LaTeX -> spoken English).

Usage (venv python):
  python -X utf8 scripts/train_tn.py --max-steps 300
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(r"H:\SHARE\latex-tts\models\Qwen3-4B")
OUT_DIR = ROOT / "models" / "tn-lora"


def load_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def build_example(tok, messages: list[dict]) -> dict:
    """Tokenize full chat; supervise only the assistant answer tokens."""
    full = tok(
        tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        ),
        add_special_tokens=False,
    ).input_ids
    # prompt = everything up to and including the assistant header
    prompt = tok(
        tok.apply_chat_template(
            messages[:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
        ),
        add_special_tokens=False,
    ).input_ids
    n_prompt = len(prompt)
    input_ids = list(full)
    labels = [-100] * n_prompt + input_ids[n_prompt:]
    return {"input_ids": input_ids, "labels": labels}


def pad_batch(examples: list[dict], pad_id: int) -> tuple:
    maxlen = max(len(e["input_ids"]) for e in examples)
    input_ids, labels, attn = [], [], []
    for e in examples:
        n = len(e["input_ids"])
        pad = maxlen - n
        input_ids.append(e["input_ids"] + [pad_id] * pad)
        labels.append(e["labels"] + [-100] * pad)
        attn.append([1] * n + [0] * pad)
    return input_ids, labels, attn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "tn_train" / "train.jsonl"))
    ap.add_argument("--val", default=str(ROOT / "data" / "tn_train" / "val.jsonl"))
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    import sys

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with (ROOT / "logs" / "train_tn_stage.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log(f"stage=device={device} loading tokenizer")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    log("stage=loading model (8GB)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
        attn_implementation="eager",
    )
    log("stage=model loaded")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    log("stage=peft wrapped")

    train_rows = [build_example(tok, r["messages"]) for r in load_rows(args.data)]
    val_rows_raw = load_rows(args.val)
    log(f"stage=data ready: train {len(train_rows)}, val {len(val_rows_raw)}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    steps_per_epoch = math.ceil(len(train_rows) / args.batch / args.grad_accum)
    total_steps = args.max_steps or int(steps_per_epoch * args.epochs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / "train_tn.jsonl"
    rng = random.Random(args.seed)
    model.train()
    step, t0, running = 0, time.time(), 0.0
    done = False
    for _epoch in range(math.ceil(args.epochs) if not args.max_steps else 10**9):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        micro = [train_rows[i] for i in order]
        for bstart in range(0, len(micro), args.batch):
            batch = micro[bstart : bstart + args.batch]
            input_ids, labels, attn = pad_batch(batch, pad_id)
            input_ids = torch.tensor(input_ids, device=model.device)
            labels = torch.tensor(labels, device=model.device)
            attn = torch.tensor(attn, device=model.device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            if step == 0 and bstart == 0:
                log(f"stage=first forward done, loss={out.loss.item():.4f}")
            (out.loss / args.grad_accum).backward()
            if step == 0 and bstart == 0:
                log("stage=first backward done")
            running += out.loss.item()
            if (bstart // args.batch + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0:
                    lr = sched.get_last_lr()[0]
                    print(
                        f"step {step}/{total_steps}  loss {running / (10 * args.grad_accum):.4f}  lr {lr:.2e}  "
                        f"{(time.time() - t0) / step:.1f}s/step",
                        flush=True,
                    )
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, "loss": running, "lr": lr}) + "\n")
                    running = 0.0
                if step >= total_steps:
                    done = True
                    break
        if done:
            break

    model.save_pretrained(OUT_DIR)
    print(f"saved LoRA -> {OUT_DIR}")

    # quick val: greedy decode 5 val prompts
    model.eval()
    model.config.use_cache = True
    for r in val_rows_raw[:5]:
        prompt_ids = tok(
            tok.apply_chat_template(
                r["messages"][:-1], tokenize=False, add_generation_prompt=True, enable_thinking=False
            ),
            add_special_tokens=False,
        ).input_ids
        ids = torch.tensor([prompt_ids], device=model.device)
        gen = model.generate(ids, max_new_tokens=80, do_sample=False)
        text = tok.decode(gen[0][len(prompt_ids):], skip_special_tokens=True)
        gold = r["messages"][-1]["content"]
        mark = "OK " if text.strip().lower() == gold.strip().lower() else "MISS"
        print(f"[{mark}] gold={gold!r} pred={text.strip()!r}")


if __name__ == "__main__":
    main()
