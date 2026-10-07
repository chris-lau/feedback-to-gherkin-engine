"""Hugging Face Space backend: dual-endpoint streaming inference.

Serves ONE copy of Llama-3.2-3B-Instruct with the QLoRA adapter attached, and
exposes two named API endpoints:

  /predict_base      generation with the adapter disabled  (`model.disable_adapter()`)
  /predict_finetuned generation with the adapter active

Both stream token-by-token via TextIteratorStreamer. The portfolio site
(demo/index.html) calls these through @gradio/client, which the Gradio server
permits cross-origin (its API routes send permissive CORS headers).

Space setup (CPU basic):
  - SDK: Gradio. Secrets/env: ADAPTER_REPO (default below).
  - fp32 keeps 3B weights + KV cache inside 16 GB; expect a few tokens/sec —
    streaming keeps the UX acceptable. For production latency, upgrade to
  ZeroGPU and set PRECISION=bfloat16.

Run locally:  python backend/app.py   (expects an adapter dir or HF repo)
"""

from __future__ import annotations

import contextlib
import functools
import threading

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- Configuration ----------------------------------------------------------

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_REPO = "mrchrislau/feedback-to-gherkin-lora"   # or set the ADAPTER_REPO env var
PRECISION = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
    __import__("os").environ.get("PRECISION", "float32")
]
MAX_NEW_TOKENS = 512

SYSTEM_PROMPT = (
    "You are a requirements formatting engine. Convert the raw user feedback "
    "into exactly three sections in this order: 'PROBLEM STATEMENT:', "
    "'USER STORY:', 'ACCEPTANCE CRITERIA:'. The user story must follow the "
    "pattern 'As a ..., I want ..., So that ...'. Acceptance criteria must use "
    "strict Gherkin (Scenario:, Given, When, Then, And). Output only the three "
    "sections. No greetings, no explanations, no markdown."
)

PRESETS = {
    "SAML SSO Session Timeout": (
        "Our team relies on the Okta integration all day — I'm the IT operations lead at a "
        "mid-size fintech. Basically, sessions keep expiring mid-workday and everyone has to "
        "re-authenticate constantly. Helpdesk tickets about this spiked to ~40 last week alone."
    ),
    "Billing CSV Parser Failure": (
        "I manage the customer billing workspace for our org (finance operations manager at a "
        "design agency). Basically, exporting more than about 5,000 invoices to CSV always fails "
        "with a 504. Our auditors asked why the reconciliations don't tie out."
    ),
    "Slow Export Crash": (
        "I'm the marketing operations manager here and I look after the insights platform. "
        "Basically, any date range over 90 days fails the workbook export. CSV export of the "
        "same range works, so we convert manually. Quarterly reporting now takes days."
    ),
}

# --- Model ------------------------------------------------------------------

print(f"loading {BASE_MODEL} ({PRECISION}) + adapter {ADAPTER_REPO} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=PRECISION, device_map="cpu")
model = PeftModel.from_pretrained(model, ADAPTER_REPO)
model.eval()
print("model ready")


# --- Inference --------------------------------------------------------------


def _generate(feedback: str, use_adapter: bool, streamer: TextIteratorStreamer) -> None:
    ctx = model.disable_adapter() if not use_adapter else contextlib.nullcontext()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": feedback},
    ]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    with ctx, torch.inference_mode():
        model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )


def stream_generate(feedback: str, use_adapter: bool):
    feedback = (feedback or "").strip()
    if not feedback:
        yield "Paste some raw user feedback on the left first."
        return
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    worker = threading.Thread(target=functools.partial(_generate, feedback, use_adapter, streamer))
    worker.start()
    acc = ""
    for delta in streamer:
        acc += delta
        yield acc
    worker.join()


def predict_base(feedback: str):
    """API: /predict_base — raw Llama-3.2-3B-Instruct with the adapter off."""
    yield from stream_generate(feedback, use_adapter=False)


def predict_finetuned(feedback: str):
    """API: /predict_finetuned — QLoRA-specialized formatting engine."""
    yield from stream_generate(feedback, use_adapter=True)


# --- UI (the Space itself doubles as a demo) --------------------------------

with gr.Blocks(title="Feedback → Gherkin Engine") as demo:
    gr.Markdown(
        "## Feedback → Gherkin Engine\n"
        "Dual-endpoint demo: the same 3B model with and without the QLoRA adapter. "
        "`/predict_base` vs `/predict_finetuned` are callable via `@gradio/client`."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(
                label="Raw user feedback",
                placeholder="Paste a support ticket, call transcript, or escalation...",
                lines=10,
            )
            preset = gr.Dropdown(choices=list(PRESETS), label="Presets", value=None)
            preset.input(lambda name: PRESETS.get(name, ""), [preset], [inp])
            btn_base = gr.Button("Run base model", variant="secondary")
            btn_tuned = gr.Button("Run fine-tuned model", variant="primary")
        with gr.Column():
            out_base = gr.Textbox(label="Base (verbose / chatty)", lines=18)
            out_tuned = gr.Textbox(label="Fine-tuned (strict schema)", lines=18)
    btn_base.click(predict_base, [inp], [out_base], api_name="predict_base")
    btn_tuned.click(predict_finetuned, [inp], [out_tuned], api_name="predict_finetuned")

demo.queue(default_concurrency_limit=1)
if __name__ == "__main__":
    demo.launch(ssr_mode=False)
