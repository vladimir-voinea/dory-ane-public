<!-- tracker: DANE-2 (issue id 735), project DANE, column Todo -->
**Blocked by:** nothing  |  **Blocks:** nothing

## Parent

DANE-1 — Spec: Dory1.0-4B on the M4 Max Neural Engine at 24K context

## What to build

The yardstick every later ticket is judged by (spec: Testing Decisions). One command takes a candidate's per-position log-probabilities for a fixed evaluation set at a given loop count, and reports mean KLD, top-1 agreement and the worst positions against llama.cpp + PR #30306 running Dory's BF16 GGUF. It works in the spec's two reference modes:

- **exact** (`per_loop`, independent of batching, equal to a full recompute);
- **decode** (`last_loop` with the candidate's batching: one token per step, or 64-token batches).

It also sets up the workspace the other tickets build in:
- an ANEMLL work branch cut from upstream `main` (0.3.5);
- NVIDIA's BF16 safetensors and the BF16 GGUF on disk;
- a fixed held-out evaluation text of about 4K tokens, tokenized with Dory's tokenizer.

llama-server rejects `echo`, so it returns no log-probabilities for prompt tokens. Take the reference from llama-perplexity's saved-logits file (`--kl-divergence-base`, which may need building in the side build), or from teacher-forced one-token requests with prompt caching and `n_probs` (which match decode mode naturally). The scorer is proven by scoring the Q8_0 GGUF as a candidate.

## Acceptance criteria

- [ ] An ANEMLL work branch exists, cut from upstream `main` (0.3.5), not from the July 2025 checkout.
- [ ] NVIDIA's BF16 safetensors and the BF16 GGUF are on disk and match the Hub's checksums.
- [ ] Reference log-probabilities exist for the evaluation set at loops 1, 2 and 4 in exact mode, and at 2 loops in decode mode for both one-token steps and 64-token batches.
- [ ] The scorer accepts candidate log-probabilities in one documented format and prints mean KLD, top-1 agreement and the worst positions.
- [ ] Scoring the Q8_0 GGUF in exact mode at 2 loops lands near the published figures (about 0.003 KLD and 99% top-1).
- [ ] The BF16 decode-mode references are scored against the BF16 exact-mode reference and the gap is recorded: it is the floor any `last_loop` candidate can reach.