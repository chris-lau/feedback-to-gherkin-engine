// Cloudflare Worker: /predict_base + /predict_finetuned for the feedback→Gherkin engine.
// Runs on Workers AI (BYO-LoRA). Deploy: npx wrangler deploy (see wrangler.toml).
//
// /predict_base       → @cf/meta/llama-3.2-3b-instruct            (adapter off)
// /predict_finetuned  → same model + the `feedback-to-gherkin`     (adapter on)
//                       fine-tune, via the `-lora` variant
// Both accept POST { "feedback": "..." } and respond with a plain-text chunked
// stream, which demo/index.html renders as it arrives.

const BASE_MODEL = "@cf/meta/llama-3.2-3b-instruct";
const FINETUNE_ID = "feedback-to-gherkin-v2";
const SYSTEM_PROMPT =
  "You are a requirements formatting engine. Convert the raw user feedback " +
  "into exactly three sections in this order: 'PROBLEM STATEMENT:', " +
  "'USER STORY:', 'ACCEPTANCE CRITERIA:'. The user story must follow the " +
  "pattern 'As a ..., I want ..., So that ...'. Acceptance criteria must use " +
  "strict Gherkin (Scenario:, Given, When, Then, And). Output only the three " +
  "sections. No greetings, no explanations, no markdown.";

const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "POST, OPTIONS",
  "access-control-allow-headers": "content-type",
};
const cors = (body, init = {}) =>
  new Response(body, { ...init, headers: { ...CORS, ...(init.headers || {}) } });

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") return cors(null, { status: 204 });
    const url = new URL(request.url);
    if (request.method !== "POST") return cors("not found", { status: 404 });
    if (url.pathname === "/predict_base") return infer(request, env, false);
    if (url.pathname === "/predict_finetuned") return infer(request, env, true);
    return cors("not found", { status: 404 });
  },
};

async function infer(request, env, finetuned) {
  let feedback = "";
  try {
    ({ feedback = "" } = await request.json());
  } catch {}
  feedback = (feedback || "").trim();
  if (!feedback) return cors("Paste some raw user feedback first.", { status: 400 });

  const payload = {
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user", content: feedback },
    ],
    max_tokens: 512,
    temperature: 0,
    stream: true,
  };
  if (finetuned) payload.lora = FINETUNE_ID;

  // Same model id for both endpoints: passing `lora` switches the adapter on.
  const result = await env.AI.run(BASE_MODEL, payload);

  // Workers AI stream:true yields SSE lines; re-emit as a plain text stream.
  const text = new TransformStream();
  const writer = text.writable.getWriter();
  const enc = new TextEncoder();
  (async () => {
    const dec = new TextDecoder();
    let buf = "";
    try {
      for await (const chunk of result) {
        buf += dec.decode(chunk, { stream: true });
        let idx;
        while ((idx = buf.indexOf("\n")) >= 0) {
          const line = buf.slice(0, idx).trim();
          buf = buf.slice(idx + 1);
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") return writer.close();
          try {
            const j = JSON.parse(data);
            if (j.response) await writer.write(enc.encode(j.response));
          } catch {}
        }
      }
      await writer.close();
    } catch (e) {
      await writer.abort(e);
    }
  })();

  return cors(text.readable, {
    headers: { "content-type": "text/plain; charset=utf-8" },
  });
}
