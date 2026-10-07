#!/usr/bin/env python3
"""Prepare the QLoRA adapter for Cloudflare Workers AI BYO-LoRA upload.

Downloads adapter_config.json + adapter_model.safetensors from the Hub into
workers-ai/lora/ and injects "model_type": "llama" into the config — required
by Workers AI before it will accept the adapter.

No ML libraries needed; runs on any Python 3.9+.

Usage:  python3 workers-ai/prep_adapter.py
"""

import json
import sys
import urllib.request
from pathlib import Path

HF_REPO = "https://huggingface.co/mrchrislau/feedback-to-gherkin-lora/resolve/main/"
OUT = Path(__file__).resolve().parent / "lora"
WEIGHTS = "adapter_model.safetensors"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    dest = OUT / WEIGHTS
    if dest.exists():
        print(f"skip download, {WEIGHTS} already present")
    else:
        print(f"downloading {WEIGHTS} (~97 MB) ...")
        urllib.request.urlretrieve(HF_REPO + WEIGHTS, dest)

    cfg = json.loads(urllib.request.urlopen(HF_REPO + "adapter_config.json").read())
    # Workers AI requirements: base architecture tag + an allowed base_model_name_or_path.
    # Unsloth trains against its own bnb-4bit repo, which Cloudflare rejects; the adapter
    # is shape-compatible with the official meta-llama base, so repoint it there.
    cfg["model_type"] = "llama"
    cfg["base_model_name_or_path"] = "meta-llama/Llama-3.2-3B-Instruct"
    (OUT / "adapter_config.json").write_text(json.dumps(cfg, indent=2))

    size_mb = dest.stat().st_size / 1e6
    if size_mb > 300:
        sys.exit(f"adapter is {size_mb:.0f} MB — over Workers AI's 300 MB limit")
    print(f"ready -> {OUT}")
    print(f"  adapter_model.safetensors: {size_mb:.1f} MB, r={cfg['r']}, model_type={cfg['model_type']}")
    print("next: npx wrangler ai finetune create @cf/meta/llama-3.2-3b-instruct-lora feedback-to-gherkin ./workers-ai/lora")


if __name__ == "__main__":
    main()
