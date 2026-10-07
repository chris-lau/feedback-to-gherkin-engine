#!/usr/bin/env python3
"""Deterministic benchmark: base Llama-3.2-3B-Instruct vs the QLoRA fine-tune.

Runs every held-out sample from data/eval.json through both engines with the
identical system prompt (greedy decoding), then scores:

  - gherkin_compliance : strict Scenario/Given/When/Then schema (shared code
                         with data/lint_dataset.py, so train-time and eval-time
                         scoring cannot drift)
  - preamble           : conversational chatter / non-schema content present
  - output_tokens      : token economy (pure signal vs verbose rambling)
  - gen_seconds        : wall-clock generation time

Writes eval/results.json and prints a comparison table.

Modes:
  (default)      full run — requires an NVIDIA GPU (Colab T4) + the adapter
  --lint-only    validate the eval.json ground-truth targets, no models (CPU)
  --mock         canned outputs to exercise scoring/writing without a GPU

Usage (after training, in Colab or on a GPU box):
    python eval/run_eval.py --adapter mrchrislau/feedback-to-gherkin-lora \
        --out eval/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "data"))
from lint_dataset import is_gherkin_compliant, has_preamble  # noqa: E402

SYSTEM_PROMPT = (
    "You are a requirements formatting engine. Convert the raw user feedback "
    "into exactly three sections in this order: 'PROBLEM STATEMENT:', "
    "'USER STORY:', 'ACCEPTANCE CRITERIA:'. The user story must follow the "
    "pattern 'As a ..., I want ..., So that ...'. Acceptance criteria must use "
    "strict Gherkin (Scenario:, Given, When, Then, And). Output only the three "
    "sections. No greetings, no explanations, no markdown."
)

MOCK_BASE = (
    "Sure, I'd be happy to help you turn this feedback into a user story! "
    "Here's what I came up with based on your description:\n\n"
    "**Problem:** Sessions keep expiring which is frustrating for users.\n\n"
    "- Users need to log in again\n- It happens constantly\n- Maybe add a timeout setting?\n\n"
    "Hope this helps — let me know if you'd like me to adjust anything or add more scenarios!"
)
MOCK_TUNED = (
    "PROBLEM STATEMENT:\n"
    "A workspace tenancy administrator reports that active SSO sessions expire mid-workday, forcing "
    "constant re-authentication. The interruptions drive helpdesk volume and erode trust.\n\n"
    "USER STORY:\n"
    "As a workspace tenancy administrator, I want a configurable idle-session timeout, "
    "So that users stay signed in during active work.\n\n"
    "ACCEPTANCE CRITERIA:\n"
    "Scenario: SSO session expires during active use\n"
    "Given an authenticated user active within the idle window\n"
    "When the idle timeout elapses\n"
    "Then the session refreshes transparently\n"
    "And no unsaved work is lost"
)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_output(text: str, n_tokens: int, gen_seconds: float) -> dict:
    return {
        "gherkin_compliance": is_gherkin_compliant(text),
        "has_preamble": has_preamble(text),
        "output_tokens": n_tokens,
        "output_chars": len(text),
        "gen_seconds": round(gen_seconds, 3),
        "text": text,
    }


def aggregate(runs: list[dict]) -> dict:
    n = len(runs)
    return {
        "n": n,
        "gherkin_compliance": round(sum(r["gherkin_compliance"] for r in runs) / n, 4),
        "preamble_rate": round(sum(r["has_preamble"] for r in runs) / n, 4),
        "avg_output_tokens": round(sum(r["output_tokens"] for r in runs) / n, 1),
        "avg_output_chars": round(sum(r["output_chars"] for r in runs) / n, 1),
        "avg_gen_seconds": round(sum(r["gen_seconds"] for r in runs) / n, 3),
    }


# ---------------------------------------------------------------------------
# Full mode: generation on GPU
# ---------------------------------------------------------------------------


def run_full(samples, adapter: str, base_model: str, max_new_tokens: int) -> tuple[list, list]:
    from unsloth import FastLanguageModel  # heavy import, GPU-only

    print(f"loading {base_model} + adapter {adapter} (4-bit)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter,  # base + merged LoRA
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)

    def generate(feedback: str) -> tuple[str, int, float]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": feedback},
        ]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        t0 = time.time()
        out_ids = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0
        new_tokens = out_ids[0][inputs.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return text, len(new_tokens), elapsed

    base_runs, tuned_runs = [], []
    for i, s in enumerate(samples):
        text, ntok, secs = generate(s["input"])
        with model.disable_adapter():
            base_text, base_ntok, base_secs = generate(s["input"])
        base_runs.append(score_output(base_text, base_ntok, base_secs))
        tuned_runs.append(score_output(text, ntok, secs))
        print(f"  [{i + 1}/{len(samples)}] {s['id']}: base ok={base_runs[-1]['gherkin_compliance']} "
              f"tuned ok={tuned_runs[-1]['gherkin_compliance']}")
    return base_runs, tuned_runs


def run_mock(samples) -> tuple[list, list]:
    base_runs, tuned_runs = [], []
    for s in samples:
        base_runs.append(score_output(MOCK_BASE, 148, 6.2))
        tuned_runs.append(score_output(MOCK_TUNED, 112, 1.4))
    return base_runs, tuned_runs


# ---------------------------------------------------------------------------


def print_table(summary: dict) -> None:
    metrics = ["gherkin_compliance", "preamble_rate", "avg_output_tokens", "avg_gen_seconds"]
    header = f"{'metric':<22}{'base':>12}{'fine-tuned':>14}{'target':>10}"
    targets = {"gherkin_compliance": ">= 0.95", "preamble_rate": "0.00", "avg_output_tokens": "-50%"}
    print("\n" + header)
    print("-" * len(header))
    for m in metrics:
        tgt = targets.get(m, "")
        print(f"{m:<22}{summary['base'][m]:>12}{summary['finetuned'][m]:>14}{tgt:>10}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-file", type=Path, default=REPO_ROOT / "data" / "eval.json")
    ap.add_argument("--adapter", default="mrchrislau/feedback-to-gherkin-lora",
                    help="HF repo of the LoRA adapter (or local path)")
    ap.add_argument("--base-model", default="unsloth/Llama-3.2-3B-Instruct")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results.json")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N samples")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--lint-only", action="store_true", help="validate ground truth, no generation")
    ap.add_argument("--mock", action="store_true", help="canned outputs, no GPU needed (plumbing test)")
    args = ap.parse_args()

    samples = json.loads(args.eval_file.read_text(encoding="utf-8"))
    if args.limit:
        samples = samples[: args.limit]
    print(f"eval set: {len(samples)} samples from {args.eval_file}")

    if args.lint_only:
        bad = [s["id"] for s in samples
               if not is_gherkin_compliant(s["output"]) or has_preamble(s["output"])]
        if bad:
            sys.exit(f"ground truth failed lint: {bad}")
        print(f"ground truth: all {len(samples)} targets schema-compliant, zero preamble")
        return

    if args.mock:
        base_runs, tuned_runs = run_mock(samples)
    else:
        base_runs, tuned_runs = run_full(samples, args.adapter, args.base_model, args.max_new_tokens)

    summary = {"base": aggregate(base_runs), "finetuned": aggregate(tuned_runs)}
    results = {
        "config": {
            "eval_file": str(args.eval_file),
            "n_samples": len(samples),
            "base_model": args.base_model,
            "adapter": None if args.mock else args.adapter,
            "max_new_tokens": args.max_new_tokens,
            "decoding": "greedy",
            "mock": args.mock,
        },
        "summary": summary,
        "samples": [
            {"id": s["id"], "domain": s["domain"], "base": b, "finetuned": t}
            for s, b, t in zip(samples, base_runs, tuned_runs)
        ],
    }
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print_table(summary)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
