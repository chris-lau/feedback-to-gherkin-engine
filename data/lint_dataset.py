#!/usr/bin/env python3
"""Schema linter + train/eval splitter for the feedback -> Gherkin dataset.

Validates every record produced by generate_synthetic.py against the canonical
output schema, drops (or fails on, with --strict) invalid records, and writes a
domain-stratified 80/20 split.

Checks on `output`:
  1. Starts with exactly 'PROBLEM STATEMENT:' (no preamble of any kind).
  2. All three sections present, in order, non-empty.
  3. PROBLEM STATEMENT is 1-2 sentences.
  4. USER STORY matches: As a ..., I want ..., So that ....
  5. ACCEPTANCE CRITERIA is strict Gherkin: a Scenario line followed only by
     Given/When/Then/And/But lines, with Given before When before Then.
  6. No conversational chatter (banned phrases) anywhere in the output.
  7. No markdown artifacts (fences, bullets, bold headers).

The check helpers (is_gherkin_compliant, has_preamble, ...) are imported by
eval/run_eval.py so training-time and eval-time scoring stay identical.

Usage:
    python data/lint_dataset.py --in data/raw/generated.jsonl \
        --out-train data/train.json --out-eval data/eval.json [--strict]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

SECTIONS = ("PROBLEM STATEMENT", "USER STORY", "ACCEPTANCE CRITERIA")

USER_STORY_RE = re.compile(r"^As a .+?, I want .+?, So that .+?\.$")

GHERKIN_LINE_RE = re.compile(r"^(Scenario: |Given |When |Then |And |But )")
KEYWORD_ORDER = ("Given", "When", "Then")

# Conversational chatter that must never appear in a compliant output.
BANNED_PATTERNS = [
    r"\bsure\b",
    r"\bcertainly\b",
    r"\bof course\b",
    r"\bhere (is|are)\b",
    r"\bhere's\b",
    r"\bhope this helps\b",
    r"\bhappy to help\b",
    r"\blet me know\b",
    r"\bas an ai\b",
    r"\bi cannot\b",
    r"\bi'd be happy\b",
    r"\babsolutely\b",
    r"\bgreat question\b",
    r"\bthanks for\b",
]
BANNED_RE = re.compile("|".join(BANNED_PATTERNS), re.IGNORECASE)

MARKDOWN_RE = re.compile(r"(```|\*\*|^#{1,6} |^- |\* )", re.MULTILINE)

MAX_INPUT_CHARS = 4000
MIN_INPUT_CHARS = 80
MAX_OUTPUT_CHARS = 2000


# ---------------------------------------------------------------------------
# Individual checks (shared with eval/run_eval.py)
# ---------------------------------------------------------------------------


def split_sections(text: str) -> dict[str, str] | None:
    """Return {section: body} if the three sections exist in order, else None."""
    positions = []
    for name in SECTIONS:
        m = re.search(rf"^{name}:\s*$", text, re.MULTILINE)
        if not m:
            return None
        positions.append((name, m.start(), m.end()))
    if [p[1] for p in positions] != sorted(p[1] for p in positions):
        return None
    if not text.lstrip().startswith("PROBLEM STATEMENT:"):
        return None
    bodies = {}
    for i, (name, _s, header_end) in enumerate(positions):
        body_end = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        bodies[name] = text[header_end:body_end].strip()
    return bodies


def is_gherkin_compliant(text: str) -> bool:
    """All structural checks on a candidate output, returns True/False."""
    sections = split_sections(text)
    if sections is None:
        return False
    if any(not sections[s] for s in SECTIONS):
        return False

    problem, story, ac = sections[SECTIONS[0]], sections[SECTIONS[1]], sections[SECTIONS[2]]

    # Problem statement: 1-2 sentences.
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", problem.strip()) if s]
    if not 1 <= len(sentences) <= 2:
        return False

    # User story: exact single-line pattern.
    if not USER_STORY_RE.match(story):
        return False

    # Acceptance criteria: strict Gherkin block.
    lines = [l for l in ac.splitlines() if l.strip()]
    if not lines or not lines[0].startswith("Scenario: "):
        return False
    keyword_lines = lines[1:]
    if not keyword_lines:
        return False
    for line in keyword_lines:
        if not GHERKIN_LINE_RE.match(line) or line.startswith("Scenario"):
            return False
    first_idx = {}
    for kw in KEYWORD_ORDER:
        idxs = [i for i, l in enumerate(keyword_lines) if l.startswith(kw + " ")]
        if not idxs:
            return False
        first_idx[kw] = idxs[0]
    if not (first_idx["Given"] < first_idx["When"] < first_idx["Then"]):
        return False

    return True


def has_preamble(text: str) -> bool:
    """Conversational chatter detection: banned phrases, or content before the
    first section header, or markdown artifacts."""
    if BANNED_RE.search(text):
        return True
    stripped = text.lstrip()
    if not stripped.startswith("PROBLEM STATEMENT:"):
        return True
    if MARKDOWN_RE.search(text):
        return True
    return False


def validate_record(rec: dict) -> list[str]:
    """Return a list of failure reasons (empty list == valid)."""
    errors = []
    inp, out = rec.get("input", ""), rec.get("output", "")
    if not inp or len(inp) < MIN_INPUT_CHARS:
        errors.append(f"input_too_short (<{MIN_INPUT_CHARS} chars)")
    if len(inp) > MAX_INPUT_CHARS:
        errors.append("input_too_long")
    if not out:
        errors.append("output_empty")
    if len(out) > MAX_OUTPUT_CHARS:
        errors.append("output_too_long")
    if errors:
        return errors
    if has_preamble(out):
        errors.append("preamble_or_chatter")
    if not is_gherkin_compliant(out):
        errors.append("schema_violation")
    return errors


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


def stratified_split(samples: list[dict], train_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_domain: dict[str, list[dict]] = {}
    for s in samples:
        by_domain.setdefault(s["domain"], []).append(s)
    train, eval_ = [], []
    for domain, group in sorted(by_domain.items()):
        rng.shuffle(group)
        cut = round(len(group) * train_frac)
        train.extend(group[:cut])
        eval_.extend(group[cut:])
    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="infile", type=Path, required=True, help="generated JSONL to validate")
    ap.add_argument("--out-train", type=Path, default=Path(__file__).parent / "train.json")
    ap.add_argument("--out-eval", type=Path, default=Path(__file__).parent / "eval.json")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any record fails")
    args = ap.parse_args()

    records = []
    seen_inputs = set()
    dupes = 0
    for i, line in enumerate(args.infile.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["input"] in seen_inputs:
            dupes += 1
            continue
        seen_inputs.add(rec["input"])
        records.append(rec)

    valid, failure_counts = [], Counter()
    for rec in records:
        errors = validate_record(rec)
        if errors:
            failure_counts.update(errors)
        else:
            valid.append(rec)

    print(f"read {len(records)} records ({dupes} exact-duplicate inputs dropped)")
    print(f"valid: {len(valid)}   invalid: {len(records) - len(valid)}")
    if failure_counts:
        print("failure breakdown:")
        for reason, n in failure_counts.most_common():
            print(f"  {reason:24s} {n}")
    print("domain distribution (valid):", dict(Counter(s["domain"] for s in valid)))

    if len(valid) < 10:
        sys.exit("too few valid records to split; aborting")

    train, eval_ = stratified_split(valid, args.train_frac, args.seed)
    args.out_train.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_eval.write_text(json.dumps(eval_, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(train)} -> {args.out_train}")
    print(f"wrote {len(eval_)} -> {args.out_eval}")

    if args.strict and failure_counts:
        sys.exit(1)


if __name__ == "__main__":
    main()
