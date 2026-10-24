# The Core ML state ceiling on the Apple Neural Engine

*A measurement we could not find published anywhere, made because it blocked the port.*

## Summary

On macOS 27.2 with coremltools 8.3.0, a **stateful** Core ML `mlprogram` will load and
execute only if each individual Core ML state is at most ~24 MiB. A single state of
32 MiB or more fails with `error code: -14` — "Failed to build the model execution
plan" — at load time. The limit is **per state**, not per model and not per total bytes:

| declaration | total bytes | result |
|---|---|---|
| 1 state × 12.0 MiB | 12.0 MiB | loads, predicts |
| 1 state × 16.0 MiB | 16.0 MiB | loads, predicts |
| 1 state × 18.0 MiB | 18.0 MiB | loads, predicts (3/3 attempts) |
| 1 state × 24.0 MiB | 24.0 MiB | loads, predicts |
| 1 state × 32.0 MiB | 32.0 MiB | fails |
| 1 state × 36.0 MiB | 36.0 MiB | fails (3/3 attempts) |
| 2 states × 18.0 MiB | 36.0 MiB | loads, predicts |
| 4 states × 9.0 MiB | 36.0 MiB | loads, predicts |
| 18 states × 1.0 MiB | 18.0 MiB | loads, predicts |
| 144 MiB as an input tensor + output tensor | 144 MiB | loads, predicts |

The last row matters: **the same 144 MiB the ANE refuses as a state is fine as a
graph input.** The ceiling is on *residency*, not on the engine's ability to compute
over that much data.

## How to reproduce

Each row is an independent tiny model whose only variable is the state shape. The
computation is identical in every one: read one row at a dynamic index, lerp it with a
new value behind a host-supplied gate, write it back — the exact operation a KV-cache
write performs.

```python
import coremltools as ct, torch
from coremltools.converters.mil.mil import types

class Blk(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.register_buffer("cache", torch.zeros(18, 4, rows, 256, dtype=torch.float16),
                             persistent=False)
    def forward(self, hidden, index, gate):
        row = self.cache[:, :, index[0], :]              # dynamic read
        new = hidden.to(torch.float16).view(1, 4, 1, 64).expand(18, 4, 1, 256)
        self.cache[:, :, index[0]:index[0] + 1, :] = row + (new - row) * gate
        return (self.cache[:, :, index[0], :] * 0).view(1, 1, 256)

model = ct.convert(
    torch.jit.trace(Blk(1024).eval(), (...), strict=False),
    states=[ct.StateType(wrapped_type=types.fp16, name="cache")],
    compute_units=ct.ComputeUnit.CPU_AND_NE,
    minimum_deployment_target=ct.target.macOS15,
    convert_to="mlprogram")
```

Then load and predict `rows = 512, 1024, 2048, 4096` (18, 36, 72, 144 MiB) and observe
the transition between 24 MiB and 32 MiB. Splitting the same bytes into N states and
observing that it passes is the control that proves the limit is per state.

## What it is not

We ruled these out with explicit controls, and they are worth recording because each
one was a plausible explanation at some point:

- **Not the ANE refusing the operations.** A gated dynamic `slice_update` on a state,
  dynamic state indexing by an int32 input, a softmax over a 5376-wide mask, 36 declared
  states — all load and predict individually. No attention op is unsupported.
- **Not model complexity or depth.** The failing package differs from the passing one
  only in state size.
- **Not memory pressure.** The ladder is deterministic: 18 MiB passes 3/3 and 36 MiB
  fails 3/3 within the same minute, and a 144 MiB *input tensor* works while a 36 MiB
  *state* fails.
- **Not a corrupt artifact.** The same package that fails on `CPU_AND_NE` executes
  correctly on `CPU_ONLY`.

## Two adjacent traps, both worth knowing

1. **A sandboxed shell cannot load a stateful model at all.** Core ML writes its
   compiled-model cache under `~/Library/Caches/com.apple.CoreML` when loading; if that
   path is not writable, the load fails with the same generic `-14`. If your agent or CI
   runs in a workspace-only sandbox, you will see "the plan builder is broken" and it is
   really "permission denied".
2. **`ct.models.MLModel(path)` constructs lazily.** It does not build an execution plan,
   so "the model loads" proves the file is valid, not that it can execute. Always
   construct → `make_state()` → `predict()`. Likewise a stateful model must be given its
   state: `predict(feeds, state=model.make_state())`, or Core ML reports the cache as an
   unmet input feature.

## Consequence for a 24K-context model

Dory's KV cache at 24,576 tokens is 864 MiB (nine full-attention layers × 4 heads ×
24,576 × 256 × 2 B × 2 slots) plus 135 MiB sliding. Aggregated, or even split only by
head, it is far over the ceiling. Split per **(layer, head)** it is 36 states of exactly
24.0 MiB — inside the band — and every byte stays resident on the ANE, which avoids the
~2 GiB of host↔ANE traffic per token that offloading the cache as a tensor would cost.
