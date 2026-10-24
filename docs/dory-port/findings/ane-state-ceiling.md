# The Core ML state ceiling on the Apple Neural Engine

*Two ceilings, not one — and the second one inverts the obvious conclusion.*

## Summary

On macOS 27.2 with coremltools 8.3.0, a **stateful** Core ML `mlprogram` has two
independent limits, both measured here:

**1. A single state may not exceed ~24 MiB.** 32 MiB and 36 MiB fail to build an
execution plan (`error code: -14`) at load; 24 MiB and below load and predict.

**2. The number of states is capped between 56 and 63.** This is the
non-obvious one. 63 states of 1.0 MiB (63 MiB total) **fail**, while 36 states of
24.0 MiB (864 MiB total) **load and predict**. Total resident bytes are therefore *not*
the binding constraint, at least up to 864 MiB — the count is.

| declaration | states | total | result |
|---|---|---|---|
| 1 state × 12.0 MiB | 1 | 12.0 MiB | loads, predicts |
| 1 state × 16.0 MiB | 1 | 16.0 MiB | loads, predicts |
| 1 state × 18.0 MiB | 1 | 18.0 MiB | loads, predicts (3/3) |
| 1 state × 24.0 MiB | 1 | 24.0 MiB | loads, predicts |
| 1 state × 32.0 MiB | 1 | 32.0 MiB | **fails** |
| 1 state × 36.0 MiB | 1 | 36.0 MiB | **fails** (3/3) |
| 2 states × 18.0 MiB | 2 | 36.0 MiB | loads, predicts |
| 4 states × 9.0 MiB | 4 | 36.0 MiB | loads, predicts |
| 18 states × 1.0 MiB | 18 | 18.0 MiB | loads, predicts |
| 8 states × 24.0 MiB | 8 | 192.0 MiB | loads, predicts |
| 16 states × 24.0 MiB | 16 | 384.0 MiB | loads, predicts |
| **36 states × 24.0 MiB** | **36** | **864.0 MiB** | **loads, predicts** |
| **63 states × 1.0 MiB** | **63** | **63.0 MiB** | **fails** |
| 63 states × 5–24 MiB (a real 24K cache shape) | 63 | 999.0 MiB | fails |
| 144 MiB as a graph **input/output** (no state) | 0 | 144 MiB | loads, predicts |

The last row still matters: a cache too large or too numerous to declare as state can
be passed as a tensor instead, at the cost of streaming it in and out on every call.

## Why "more, smaller states" is the wrong lesson

The first measurement — 36 MiB in one state fails, 36 MiB in two states passes — invites
the conclusion that splitting is always safe. It is not. Splitting a cache into more
carriers is only free up to the count ceiling; past it the program stops loading no
matter how small the individual states are. The correct design target is therefore
**as few carriers as the per-state size limit allows**, not as many as possible:

```
per-state budget: 24 MiB           -> a 135 MiB sliding cache needs >= 6 carriers
state-count budget: 56 ok, 63 not  -> a 864 MiB full cache as 36 per-(layer,head) states fits,
                                      42 carriers carrying 999 MiB loads and predicts
```

The two limits were found in **opposite directions**: four times more bytes in one third
of the carriers is fine, and one third of the bytes in twice the carriers is not. A
reader who takes away only "keep states small" would draw exactly the wrong conclusion.

## How to reproduce

Each row is an independent tiny model whose only variable is the state declaration. The
computation is identical in every one: read one row at a dynamic index, lerp it with a
new value behind a host-supplied gate, write it back — exactly what a KV-cache write
does.

```python
import coremltools as ct, torch
from coremltools.converters.mil.mil import types

class Blk(torch.nn.Module):
    def __init__(self, rows):
        super().__init__()
        self.register_buffer("cache", torch.zeros(18, 4, rows, 256, dtype=torch.float16),
                             persistent=False)
    def forward(self, hidden, index, gate):
        row = self.cache[:, :, index[0], :]                      # dynamic read
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

Vary `rows` for the per-state limit; declare N independent state buffers to find the
count limit. Both are cheap: the packages carry no weights, so each is well under 1 MiB.

## What it is not

Ruled out with explicit controls, because each was a plausible explanation at some point:

- **Not the ANE refusing the operations.** A gated dynamic `slice_update` on a state,
  dynamic state indexing by an int32 input, a softmax over a 5376-wide mask, and dozens
  of declared states all load and predict individually.
- **Not memory pressure.** The results are deterministic and reproducible: 18 MiB passes
  3/3 and 36 MiB fails 3/3 within the same minute, and a 144 MiB *input tensor* works
  while a 36 MiB *state* fails.
- **Not a corrupt artifact.** The same package that fails on `CPU_AND_NE` executes
  correctly on `CPU_ONLY`.

## Two adjacent traps, both worth knowing

1. **A sandboxed shell cannot load a stateful model at all.** Core ML writes its
   compiled-model cache under `~/Library/Caches/com.apple.CoreML` when loading; if that
   path is not writable, the load fails with the same generic `-14`. If your agent or CI
   runs in a workspace-only sandbox you will see "the plan builder is broken" and it is
   really "permission denied".
2. **`ct.models.MLModel(path)` constructs lazily.** It does not build an execution plan,
   so "the model loads" proves the file is valid, not that it can execute. Always
   construct → `make_state()` → `predict()`. And a stateful model must be handed its
   state: `predict(feeds, state=model.make_state())`.

## Consequence for a 24K-context model

Dory's KV cache at 24,576 tokens is 864 MiB (nine full-attention layers × 4 heads ×
24,576 × 256 × 2 B × 2 slots) plus 135 MiB sliding. Aggregated it is a single 864 MiB
state and fails; per *head only* it is 216 MiB states and still fails; per
**(layer, head)** it is 36 states of exactly 24.0 MiB and **loads**. The sliding cache
then needs at most 6 carriers of 22.5 MiB to stay under the count ceiling — 42 states and
999 MiB in total, all resident on the ANE, with no host↔ANE traffic per token.
