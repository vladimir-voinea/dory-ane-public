# Spec — Dory1.0-4B on the M4 Max Neural Engine at 24K context

**Status:** ready for an agent. **Source:** the owner's request on 2026-10-23, after Dory was brought up on the GPU in Unsloth Studio and its Neural Engine feasibility was probed on this Mac. Numbers marked *measured* come from those probes; numbers marked *estimate* are derived from them.

## Problem Statement

The owner runs NVIDIA Nemotron-Labs **Dory1.0-4B**, a looped (depth-recurrent) 4B reasoning model, on a MacBook with an **M4 Max (36 GB)**. Today it runs on the GPU through a llama.cpp build carrying the unmerged Dory PR (ggml-org/llama.cpp#30306), wired into Unsloth Studio. The owner wants the same model on the **Apple Neural Engine** instead — lower power, the GPU left free for other models — without giving up **long context (24K tokens)** or the ability to **choose how many loops** the model runs.

Nothing does this today:

1. **ANEMLL's long-context runtime (anemll-forge)** — 8K–192K context, INT8 KV, attention tiling — supports only M5/M6; its README states that pre-M5 chips "fail before loading".
2. **Classic ANEMLL (0.3.5, Core ML)** runs on M4 but supports only LLaMA, Qwen 3, Qwen 2.5, Gemma 3 and DeepSeek-distill, and tops out at 4K context (Gemma 3, recommended 512–2048).
3. **No Core ML or PyTorch implementation of Dory exists.** NVIDIA's BF16 checkpoint declares `model_type: dory` with no remote code, and transformers 5.5 has no Dory class. The only runnable implementation is llama.cpp + PR #30306.

## Solution

A **Dory port for classic ANEMLL**, built and run entirely on this Mac, that:

- converts NVIDIA's BF16 checkpoint into Neural Engine Core ML chunks with LUT quantization;
- runs decode and prefill on the Neural Engine with a **24,576-token context**;
- lets the **loop count (1–4) be chosen at run time**, with no reconversion;
- serves the model through a **local OpenAI-compatible endpoint** that Unsloth Studio adds as a Connected provider;
- is **scored for quality, speed and energy** against the GPU llama.cpp build.

Expectation on this Mac (*estimate*): ~10 tok/s decode at 24K with 2 loops, ~16 with 1 loop, ~6 with 4 loops. The GPU does 22.7 tok/s at 24K with 2 loops (*measured*). The owner chose to build it knowing it will be slower than the GPU, in exchange for Neural Engine power draw.

## User Stories

### Converting

1. As the owner, I want to convert Dory's BF16 checkpoint to Neural Engine models with one command, so that I never hand-assemble chunks.
2. As the owner, I want to choose the LUT bit width separately for the embeddings, the transformer blocks and the LM head, so that I can trade quality for speed.
3. As the owner, I want the converter to stop with a clear message when any Core ML file would exceed the macOS ANE size limit (~2 GB), so that a conversion never silently produces an unloadable model.
4. As the owner, I want one meta file describing the converted model — context length, chunk layout, LUT widths, KV mode, supported loop range, tokenizer and chat template — so that the runtime and the server load it without extra flags.
5. As the owner, I want conversion to verify that every operation of every chunk is placed on the Neural Engine, so that nothing silently falls back to the CPU or GPU.
6. As the owner, I want the first-load ANE compilation cached, so that later starts are fast.
7. As the owner, I want conversion to start from NVIDIA's official BF16 safetensors, so that the result inherits NVIDIA's weights rather than a community quant.

### Running and chatting

8. As the owner, I want to chat with Dory on the Neural Engine from the command line with streamed tokens, so that I can try it without any other software.
9. As the owner, I want to set the loop count (1–4) when I start a session, so that I can trade depth for speed.
10. As the owner, I want changing the loop count mid-session to reset the conversation's cache with a clear message, so that I never get answers built on a cache computed at another loop count.
11. As the owner, I want 24,576 tokens of context, so that a long document plus a conversation fits.
12. As the owner, I want long prompts to prefill in batches with visible progress, so that a 24K prompt never looks hung.
13. As the owner, I want a clear error — not a crash and not silent truncation — when a prompt exceeds the context, so that I know to shorten it.
14. As the owner, I want reasoning on by default and switchable off, as Dory's chat template supports, so that I control the cost of thinking.
15. As the owner, I want reasoning streamed separately from the answer, so that clients show thinking as thinking.
16. As the owner, I want Dory's tool calls emitted in its native format, so that tool-using clients work.
17. As the owner, I want NVIDIA's sampling defaults (temperature 1.0, top_p 0.95, top_k 0, min_p 0) with per-request overrides, so that quality matches NVIDIA's recommendation out of the box.
18. As the owner, I want a deterministic greedy mode, so that runs are reproducible for testing.
19. As the owner, I want generation to stop on Dory's end-of-turn tokens, so that replies end where the template says they end.

### Serving and Studio

20. As the owner, I want an OpenAI-compatible endpoint (a models list and streaming chat completions), so that any OpenAI client can use the Neural Engine model.
21. As the owner, I want to choose the loop count per request — through a request field or a model alias such as `dory-ane-2loops` — so that clients without custom fields can still pick it.
22. As the owner, I want to add the endpoint to Unsloth Studio as a Connected provider and chat with it there, so that Dory-on-ANE sits beside my other models.
23. As the owner, I want the models list to report the context length, the loop range and the default loop count, so that clients can size prompts.
24. As the owner, I want the server bound to localhost only, so that the model is not exposed on my network.
25. As the owner, I want Studio's GPU llama.cpp path for Dory to keep working unchanged, so that I can switch back at any time.

### Measuring

26. As the owner, I want a reference harness that scores any implementation against llama.cpp-Dory running the BF16 weights at the same loop count, so that every stage is judged by the same yardstick.
27. As the owner, I want KLD and top-1 agreement reported per loop count (1, 2, 4) and per LUT configuration, so that I pick a configuration knowingly.
28. As the owner, I want a long-context recall check — codes planted at depths up to 24K — so that I know the far end of the context actually works.
29. As the owner, I want decode and prefill tok/s at 2K, 8K, 16K and 24K for loops 1, 2 and 4, plus resident memory, so that I can compare with the GPU.
30. As the owner, I want energy per generated token on the Neural Engine and on the GPU, so that I can judge whether the trade is worth it.
31. As the owner, I want an FP16 safety report — no inf or NaN in any chunk over the evaluation set — so that numerical failures are caught before they reach a chat.
32. As the owner, I want the port to follow ANEMLL's own conventions (model class, converter, chat runtime, meta file, tests), so that it reads like the rest of ANEMLL and could be upstreamed.

## Implementation Decisions

**D1 — Platform: classic ANEMLL, not anemll-forge.** Core ML through coremltools 8.x with stateful models (macOS 15+ deployment target), on a branch cut from ANEMLL 0.3.5 (upstream `main`, February 2026). anemll-forge requires M5/M6 and is not an option on this Mac. The local ANEMLL checkout dates from July 2025 and must not be the base.

**D2 — Source weights and reference.** Weights: NVIDIA's BF16 safetensors (`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16`, one 7.95 GB file). Tensor names: `backbone.embeddings`, per block `backbone.layers.N.mixer.{q,k,v,o,g}_proj`, `mixer.sqk`, `mixer.s_gate`, `attn_alpha` (attention blocks) or `mixer.{gate,up,down}_proj`, `mixer.suv_gate`, `mixer.suv_up`, `mlp_alpha` (MLP blocks), and an untied `lm_head.weight` with `lm_head.logit_scale`. Behavioural reference: **llama.cpp + PR #30306** running the BF16 GGUF. Where this spec and llama.cpp disagree, llama.cpp wins: the PR's model graph and its HF→GGUF converter are the authoritative description of the math and of the weight mapping.

**D3 — The architecture the port implements.**
- **Layout.** 72 blocks = 16 input + 40 recurrent + 16 output. Blocks alternate attention and MLP; attention blocks are sliding-window or full in a 3:1 pattern (`sliding, mlp, sliding, mlp, sliding, mlp, full, mlp` repeated), so there are **9 full-attention blocks (2 input, 5 recurrent, 2 output)**, 27 sliding and 36 MLP.
- **Sizes.** hidden 2560; MLP 7168; 16 query heads, 4 KV heads, head_dim 256; vocab 131,072 (Tekken / "pixtral" BPE); BOS 1, EOS 11.
- **Attention.** RoPE in NEOX (rotate-half) style over all 256 dims, θ = 10,000 for sliding blocks and 10,000,000 for full blocks. Sliding window 1024: a query sees the last 1024 positions including itself. Then q and k are **L2-normalized per head over head_dim**; q is scaled per (head, dim) by `(sqk · √256 / init_std)²`; softmax scale `1/√256`. The attention output (all 4096 channels) is multiplied by the gate `sigmoid((W_g·h) ⊙ (s_gate · √gate_init / init_std)²)`, L2-normalized, then projected by `o_proj`.
- **MLP.** `g = (W_gate·h) ⊙ suv_gate·√2560`, `u = (W_up·h) ⊙ suv_up·√2560`, output `down_proj(L2(silu(g) ⊙ u))`.
- **Residual (nGPT style).** Every block updates `h ← L2(a + |α|·(L2(f) − a))` with `a = L2(h)`, `f` the block output and `|α| = |α_param · α_init / init_std|`.
- **Loops.** `h_in` is the hidden state entering the recurrent block. Each loop starts with `h ← L2(h + h_in)` and ends with an extra `L2`.
- **Logits.** `(W_lm·h) ⊙ logit_scale / init_std`.
- **Constants** (from the GGUF): init_std 0.0177, α_init 0.05, gate_init 50.596; L2 uses ε = 1e-6.

**D4 — Loops and the KV mode: `last_loop`, loop count chosen at run time.** Implement llama.cpp's `last_loop` semantics: one KV cache per *physical* recurrent attention block; every loop pass writes the current rows into it and attends over older rows' final-loop KV plus the current rows' current-loop KV. Cache shapes are then independent of the loop count, so the loop count is simply how many times the host calls the recurrent chunk, for decode and prefill alike, selectable 1–4 at run time with no reconversion. Changing it invalidates the cache. On the GPU, `last_loop` at 4 loops answered test prompts correctly at the same speed as `per_loop` and used 2.7× less full-attention KV (*measured*, 64K context: 2,304 vs 6,144 MiB). Default loop count: 2 (NVIDIA's HF config default; the community GGUF defaults to 4).

`last_loop` results depend on batch boundaries: tokens in the current batch see each other's current-loop KV, while older tokens contribute only their final-loop KV. Within a single batch it equals `per_loop`, and `per_loop` equals a full recompute of the sequence — the model as trained. The port's decode (one token per step) and prefill (64-token batches) therefore differ slightly from a full recompute, and the gates below allow for it.

**D5 — ANE-safe formulation.**
- L2 normalization through ANEMLL's layer-norm-based RMSNorm, divided by √dim (Gemma 3 prior art), never a raw FP16 sum of squares, which overflows over 7,168 channels.
- Linear layers as 1×1 convolutions in ANEMLL's (N, C, H, W) layout.
- Static per-output-channel scales folded into weights at conversion, exactly, in FP32: `suv_gate·√2560` and `suv_up·√2560` into `gate_proj` and `up_proj` rows; the squared `s_gate` factor into `g_proj` rows; `logit_scale / init_std` into `lm_head` rows. The `sqk` factor stays an elementwise multiply, because it follows an L2 norm; `α` stays elementwise.
- Two RoPE tables (θ = 1e4 and 1e7), as Gemma 3 does with its two bases.
- Cache writes are single-row (decode) or slice (prefill) updates of Core ML state, and grouped-query attention reshapes queries per KV head. Never a mask-multiply rewrite of the cache, and never a materialised repeat of K/V: on this Mac the naive form took 37 ms per 24K step against 3.1 ms (*measured*).

**D6 — Chunk layout follows Dory's structure.** Five component models, each multifunction with shared weights — a single-token *infer* function and a 64-token *prefill* function, plus *rotate* variants where the sliding cache needs them:

| Component | Contents | Called per token |
|---|---|---|
| embeddings | token embedding | once |
| input block | blocks 0–15 | once |
| recurrent block | blocks 16–55 | N times (N = loop count) |
| output block | blocks 56–71 | once |
| LM head | lm_head, split into ≥ 2 parts (one ANE axis ≤ 65,536; vocab 131,072), optional in-model argmax | once |

If the recurrent block exceeds ~2 GB at the chosen LUT width, split it into two chunks, both called once per loop. ANEMLL's even FFN split (`--chunk N`) is not used.

**D7 — KV caches.** Full-attention caches are Core ML states of 24,576 rows (9 caches × 96 MiB in FP16 = 864 MiB). Sliding caches hold the 1,024-token window with ANEMLL's fill/rotate scheme (Gemma 3 prior art at window 512). On this Mac a single 24,576-row full-attention step compiled entirely to the ANE (3.1 ms; 16,384 rows: 2.1 ms; *measured*), although ANEMLL's guide caps an H/W axis at 16,384. Placement must be re-verified inside the real chunks. If a 24,576-row state falls off the ANE there, split each full-attention cache into a 16,384-row and an 8,192-row tile and merge the two attention partials with a log-sum-exp combine.

**D8 — Quantization.** ANEMLL LUT palettization with per-channel groups. Start with LUT6 blocks, LUT6 LM head, embeddings unquantized or LUT8; try LUT4 blocks if the speed gate is missed and the quality gate still holds. At 4-bit on this Mac: 0.46 ms per MLP block and 0.30 ms per attention-projection block, about 60 GB/s effective (*measured*).

**D9 — Runtime.** A Python runtime in ANEMLL's style (coremltools predictions with state) runs embeddings → input → recurrent × N → output → LM head for each decode token and for each 64-token prefill batch. Tokenizer: the BF16 repo's `tokenizer.json`. Prompting: the repo's Jinja chat template (reasoning on by default, tool calls in its native format).

**D10 — Server.** OpenAI-compatible: a models list and chat completions with SSE streaming, reasoning returned as `reasoning_content`, the loop count per request through an extra body field or a model alias, NVIDIA's sampling defaults, bound to localhost, no auth. Built on the owner's existing `anemll-server`, an OpenAI-compatible server for ANEMLL models. It is added to Unsloth Studio as a Connected provider (Studio accepts custom provider base URLs); Studio's own code is not modified.

**D11 — Performance budget (*estimate*).** Per token at 24K with N loops: (16 + 20N) MLP passes × 0.46 ms + (16 + 20N) projection passes × 0.30 ms + (4 + 5N) full-attention passes × ~2.9 ms + (12 + 15N) sliding passes (small) + LM head (~3 ms) + the overhead of 4 + N chunk calls. That is N=1 ≈ 63 ms (~16 tok/s), N=2 ≈ 97 ms (~10 tok/s), N=4 ≈ 163 ms (~6 tok/s). GPU baseline (*measured*, llama.cpp Q8_0, 2 loops, `last_loop`, 24,130-token prompt): 22.7 tok/s decode, 345 tok/s prefill, taken while ~14 GB of swap was in use.

## Testing Decisions

- **One seam: next-token log-probabilities.** Given a token sequence and a loop count, every implementation returns per-position log-probabilities: the PyTorch port in FP32, the PyTorch port in its ANE-safe FP16 form, and the converted Core ML pipeline. All three are scored at this one seam against the same reference, llama.cpp + PR #30306 running the BF16 GGUF, in one of two modes:
  - **exact** — `per_loop`, independent of batching and equal to a full recompute — for the full-precision PyTorch port;
  - **decode** — `last_loop` with the candidate's own batching (one token per step for decode, 64-token batches for prefill) — for the cached ANE-safe port and the Core ML pipeline.
  llama-server returns no log-probabilities for prompt tokens (it rejects `echo`), so the reference comes from llama-perplexity's saved-logits file (`--kl-divergence-base`) or from teacher-forced one-token requests with prompt caching and `n_probs`. Tests check external behaviour — log-probabilities, generated text, tok/s — and never internal tensors; comparing intermediate hidden states is a debugging aid, not a test.
- **The user-facing seam** is the OpenAI-compatible endpoint: an OpenAI client and Studio's Connected tab, nothing lower.
- **Gates.**
  - PyTorch FP32 against the exact reference, on ~4K tokens of held-out text at loops 1, 2 and 4: top-1 agreement ≥ 99%, mean KLD ≤ 0.005.
  - Core ML pipeline against the decode reference at 2 loops: mean KLD ≤ 0.05, top-1 ≥ 93%. For scale, the Q4_K_M GGUF scores 0.030 / 96.0% and Q8_0 0.003 / 98.9% (exact mode).
  - Long context: six codes planted at depths from 0 to 24K in one 24K prompt, all recalled, at 2 loops.
  - Placement: every op of every chunk on the ANE, by Core ML's compute plan and ANEMLL's ANE profiler.
  - FP16: no inf or NaN in any chunk output over the evaluation set; ANEMLL's FP16 preflight passes.
  - Speed: decode ≥ 8 tok/s at 24K with 2 loops. Below 6 tok/s, stop and reassess before tuning further.
  - Server: an OpenAI client streams a reply with reasoning and answer separated, and a chat through Studio's Connected tab works.
- **A good test here** runs the real thing — real weights, real Core ML on the real ANE — on fixed inputs, and compares against the reference or a threshold. No mocked Core ML.
- **Prior art:** ANEMLL's per-architecture conversion-and-inference tests (Gemma 3, Qwen 3, LLaMA), its lm-evaluation-harness integration for converted models, its FP16 preflight and its ANE profiler; the cloudnathan5 Dory GGUF methodology (KLD against BF16 at loops 1, 2 and 4).

## Out of Scope

- anemll-forge, M5/M6, iOS/iPadOS, and a Swift runtime.
- The `per_loop` KV mode, context beyond 24,576 tokens, and INT8 KV caches (a later follow-up).
- Speculative decoding, vision or audio input, fine-tuning.
- Any change to Unsloth Studio's code; Studio is only a client through its Connected tab.
- Running the GPU and ANE copies at the same time; upstreaming to ANEMLL (possible later, not required).

## Further Notes

- **This Mac:** M4 Max, 36 GB unified memory, macOS 27.2.
- **Where things are:** ANEMLL clone in `~/dev/Anemll` (origin is upstream; branch from `origin/main`), with venv `env-anemll` (coremltools 8.3, torch 2.5). Probes and their results: `~/dev/dory-ane/probes`. llama.cpp side build with Dory: `~/.unsloth/llama.cpp-dory` (llama-server built; build llama-perplexity if needed). GGUFs: `cloudnathan5/Dory1.0-4B-GGUF` (Q8_0 already in the Hugging Face cache; the BF16 GGUF is in the same repo). Server prior art: `~/dev/anemll-server`.
- **Risks:** FP16 range in the q scaling and the logit scale (fold or rescale); ANE placement of 24,576-row states inside full chunks (D7 fallback); LUT quality on normalized-transformer weights; Python call overhead at 4 + N calls per token; first-load ANE compile time; benchmarks distorted by memory pressure — unload Studio's models while measuring.
- **Energy:** `powermetrics` needs sudo, so the owner runs that step.