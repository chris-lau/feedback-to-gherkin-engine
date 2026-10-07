"""Dual-endpoint streaming inference backend — one file, three deployment modes.

Serves ONE copy of Llama-3.2-3B-Instruct with the QLoRA adapter attached, and exposes
two named API endpoints:

  /predict_base      generation with the adapter disabled  (`model.disable_adapter()`)
  /predict_finetuned generation with the adapter active

Both stream token-by-token via TextIteratorStreamer. The portfolio site
(demo/index.html) calls these through @gradio/client, which accepts either a
`owner/space` id or any Gradio server URL (e.g. a *.gradio.live tunnel).

Deployment modes (auto-detected, no code changes between them):

  1. HF Space on ZeroGPU (free for personal accounts, 2 Spaces max):
     create the Space with Gradio SDK + ZeroGPU hardware and push this folder.
     The `spaces` package present in that image is picked up automatically;
     generation is wrapped in @spaces.GPU and served on GPU slices.
  2. Colab ephemeral link: with a GPU runtime,
         SHARE=1 python backend/app.py
     launches with share=True and prints a temporary *.gradio.live URL
     (lives only as long as the notebook runtime; Colab's terms disallow
     persistent serving — for live one-offs, not a portfolio link).
  3. Anywhere else: plain `python backend/app.py` runs on CPU (fp32) or GPU
     (PRECISION=bfloat16 recommended); point CONFIG.spaceId at its public URL.
"""

from __future__ import annotations

import contextlib
import functools
import os
import threading

# `spaces` must be imported before anything initializes CUDA (ZeroGPU requirement).
try:
    import spaces  # present only on HF ZeroGPU images / if pip-installed
except ImportError:
    spaces = None

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# --- Configuration ----------------------------------------------------------

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct"
ADAPTER_REPO = os.environ.get("ADAPTER_REPO", "mrchrislau/feedback-to-gherkin-lora")
MAX_NEW_TOKENS = 512

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_DTYPE_OVERRIDE = os.environ.get("PRECISION")  # float32 | bfloat16 | float16 | auto
if _DTYPE_OVERRIDE in {"float32", "bfloat16", "float16"}:
    DTYPE = getattr(torch, _DTYPE_OVERRIDE)
elif DEVICE == "cuda":
    # T4 (Turing) has no fast bf16; ZeroGPU GPUs do.
    DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DTYPE = torch.float32

SYSTEM_PROMPT = (
    "You are a requirements formatting engine. Convert the raw user feedback "
    "into exactly three sections in this order: 'PROBLEM STATEMENT:', "
    "'USER STORY:', 'ACCEPTANCE CRITERIA:'. The user story must follow the "
    "pattern 'As a ..., I want ..., So that ...'. Acceptance criteria must use "
    "strict Gherkin (Scenario:, Given, When, Then, And). Output only the three "
    "sections. No greetings, no explanations, no markdown."
)

PRESETS = {
    "SAML SSO Clock-Skew Lockout": (
        "I'm the IT operations lead at a mid-size fintech here and I look after the corporate login flow. "
        "Basically, SAML logins fail with 'Assertion not yet valid' for a chunk of our users. "
        "Started right after the DST switch. support is walking users through workaround logins which wastes hours."
    ),
    "Billing CSV Parser Failure": (
        "Second ticket I'm filing about this: I manage the customer billing workspace for our org (finance "
        "operations manager at a design agency). Basically, exporting more than about 5,000 invoices to CSV "
        "always fails with a 504. our auditors asked why the reconciliations don't tie out."
    ),
    "Slow Export Crash": (
        "I'm the marketing operations manager here and I look after the insights platform. Basically, any date "
        "range over 90 days fails the workbook export. CSV export of the same range works, so we convert "
        "manually. quarterly reporting now takes days of manual conversion. We've worked around it for now "
        "but it's not sustainable."
    ),
}

# --- Model ------------------------------------------------------------------

print(f"loading {BASE_MODEL} on {DEVICE} ({DTYPE}) + adapter {ADAPTER_REPO} ...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=DTYPE, device_map=DEVICE)
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
    ).to(model.device)
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


def _stream_generate(feedback: str, use_adapter: bool):
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


if spaces is not None:  # ZeroGPU: hold a GPU slice for the whole streamed generation
    stream_generate = spaces.GPU(duration=120)(_stream_generate)
else:
    stream_generate = _stream_generate


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
            out_base = gr.Textbox(label="Base (schema-invalid without the adapter)", lines=18)
            out_tuned = gr.Textbox(label="Fine-tuned (strict Gherkin schema)", lines=18)
    btn_base.click(predict_base, [inp], [out_base], api_name="predict_base")
    btn_tuned.click(predict_finetuned, [inp], [out_tuned], api_name="predict_finetuned")

demo.queue(default_concurrency_limit=1)
if __name__ == "__main__":
    demo.launch(ssr_mode=False, share=os.environ.get("SHARE") == "1")
