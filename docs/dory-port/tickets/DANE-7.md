<!-- tracker: DANE-7 (issue id 740), project DANE, column Todo -->
**Blocked by:** nothing  |  **Blocks:** nothing

## Parent

DANE-1 — Spec: Dory1.0-4B on the M4 Max Neural Engine at 24K context

## What to build

Serve the converted model through the spec's D10 server, built on the owner's existing `anemll-server`:
- a models list reporting the context length, the loop range and the default loop count;
- chat completions with SSE streaming, with reasoning returned as `reasoning_content`;
- Dory's native tool calls passed through;
- the loop count per request through an extra body field or a model alias (`dory-ane-1loop`, `dory-ane-2loops`, `dory-ane-4loops`);
- NVIDIA's sampling defaults (temperature 1.0, top_p 0.95, top_k 0, min_p 0) with per-request overrides, and greedy decoding at temperature 0;
- bound to localhost only.

Then add it to Unsloth Studio's Connected tab and chat with it there. This is the user-facing seam: an OpenAI client and Studio, nothing lower. Studio's own code is not modified.

## Acceptance criteria

- [ ] The OpenAI Python client streams a reply with the reasoning and the answer separated.
- [ ] A request with tools returns a well-formed tool call.
- [ ] Both the model alias and the body field set the loop count, visibly in the server's log or response metadata, and a loop change resets the cache.
- [ ] The server listens on localhost only.
- [ ] Studio lists the model under Connected, and a chat with it completes.
- [ ] Studio's GPU Dory (llama.cpp side build) still loads and chats exactly as before.