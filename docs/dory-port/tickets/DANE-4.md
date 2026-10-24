<!-- tracker: DANE-4 (issue id 737), project DANE, column Todo -->
**Blocked by:** nothing  |  **Blocks:** nothing

## Parent

DANE-1 — Spec: Dory1.0-4B on the M4 Max Neural Engine at 24K context

## What to build

The form the Neural Engine needs, still in PyTorch and in FP16 (spec D4, D5 and D7 at small context):
- L2 normalization through ANEMLL's layer-norm-based RMSNorm divided by √dim;
- 1×1 convolutions in ANEMLL's layout;
- the static scales folded into the weights;
- two RoPE tables;
- `last_loop` KV caches held as state-like buffers — one per physical recurrent attention block, plus sliding caches of the 1,024-token window — written with single-row (decode) or 64-row slice (prefill) updates;
- grouped-query attention by reshaping queries per KV head.

It has a cached decode path and a 64-token prefill path, with the loop count chosen at run time. This is the second slice through the seam, scored in **decode** mode, and the exact shape the converter in the next ticket will trace.

## Acceptance criteria

- [ ] Folding the scales into the weights reproduces the unfolded full-precision outputs to within FP32 rounding.
- [ ] In FP16, cached decode at 2 loops scores mean KLD ≤ 0.01 and top-1 ≥ 98% against the decode-mode reference for one-token steps; 64-token prefill scores the same against the 64-token reference.
- [ ] Loops 1 and 4 pass the same decode-mode check.
- [ ] No inf or NaN anywhere over the evaluation set, and ANEMLL's FP16 preflight passes.
- [ ] Changing the loop count resets the caches, and a test shows it.
- [ ] Neither the cache writes nor grouped-query attention rewrite or repeat the whole cache (spec D5).