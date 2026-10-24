# Per-(layer, head) cache refactor — plan and code sites

State of play at handover. The layout machinery is done and measured; the chunk
that has to consume it is not started.

## Done and verified

* `dory_chunks.state_layout(config)` — single source of truth, returns
  `(name, shape)` pairs. Two axes plus the existing tile axis:
  * `heads_per_state` splits the full cache per group of key heads
    (must divide n_kv_heads; raises otherwise).
  * `slots_per_sliding_carrier` (default 3) groups the windowed cache by
    layer-slot (must divide the slot count; the message names the per-state cap).
  * tiles + head split raises rather than guessing.
* `state_attrs` / `state_names` derive from it; `state_summary` prints
  "45 states, 24.0 MiB largest, 999.0 MiB total, ceiling 24 OK / 32 FAIL".
* Default (no tiles, no split) is byte-identical to the old behaviour, asserted
  element by element against `dory_caches.cache_shapes()`.
* Measured ceilings behind the design: per state 24 MiB loads / 32 MiB fails;
  count 56 loads / 63 fails; 45 real states carrying 999.0 MiB load on
  CPU_AND_NE (probe v18).
* `test_cache_layout_numerics.py` — the equality test, written and skipping with
  its reason: aggregated layout is the reference, relative tolerance 1e-3 on
  logits (justified in the docstring: the arithmetic is unchanged, so the true
  difference is zero), argmax equality asserted, run at 4,096 and 24,576.

## Not started: the chunk must read split buffers

`DoryANEChunk` currently hands each attention layer ONE aggregated tensor of
shape `(2 * num_slots, num_kv_heads, rows, head_dim)`, K at `slot`, V at
`slot + num_slots`. The split layout declares one buffer per (layer, head) of
shape `(2, heads, rows, head_dim)`. Two ways to bridge them:

1. **Concatenate at read time** — cheap to write, useless for the goal: the
   assembled tensor is the same 864 MiB intermediate the ceiling rejects.
2. **One head at a time** (the real change) — attention is per-head independent,
   so each head can read its own buffer and compute its own scores. This keeps
   every intermediate at one head's size, which is the whole point.

Option 2 is the refactor. Sites, in the order they must change:

* `dory_ane_model.DoryANEChunk.__init__` (buffer allocation, ~line 596-610):
  register one buffer per (slot, head) instead of `kv_cache_full` /
  `kv_cache_sliding`; keep the aggregated pair when `heads_per_state` is None.
* `DoryANEChunk.forward` (~line 671): thread the per-layer buffer list down
  instead of the two tensors.
* `DoryANEFullAttention.forward` (~line 373): loop over head groups, each using
  its own K/V buffer; the q/k/v projections and the softmax stay untouched.
* `DoryANEFullAttention` cache write (~line 470-486, `write_cache_rows` or the
  gated lerp): write the layer's own head buffer, same arithmetic.
* `DoryANESlidingAttention.forward` (~line 546) and its write: same two changes,
  grouped by slot rather than by (layer, head).
* `dory_converter` state declarations (~line 379-391 and ~1027-1034) and the
  states line of the report (~1381): already parameterised through
  `state_attrs`; they need only to pass the knobs from the CLI/config.
* `dory_chat.DoryRuntime`: no change expected — `reset_state()` calls
  `make_state()` on the model, so it inherits the declaration. Verify, do not
  assume.

## Acceptance, unchanged

1. default state list byte-identical at 4K — **done**;
2. numerics equality at 4,096 and 24,576 on a fixed seed — test written, blocked
   on the chunk change;
3. one real 24K component converted with the new layout loads on CPU_AND_NE and
   predicts;
4. the full ten-package 24K set, then DANE-6's recall gate and the beyond-16K
   KLD.

## Honest estimate

The attention loop-over-heads is the substantial part: the current graph is
batched over heads, so a naive per-head loop changes op counts, matmul shapes
and the traced signature. Plan half a day with the equality test as the guard
rail, and land it behind the default-off knob so 4K never moves.

## The trap, recorded before anyone starts

The head loop must preserve **write-then-read order** inside `forward`: write the current
rows, then read the tiles/ring back, then score. The sliding ring's `cache_pos` and the
tiled `write_index` both depend on the write having already happened. A loop that reads
before writing per head still passes a single-token test and fails on the second token.

## The half-answer that must be refused

Bridging the layouts by concatenating the per-head buffers at read time is cheap to write
and worse than useless: the assembled tensor is the same 864 MiB intermediate that the
per-state ceiling rejects, so it passes the equality test and then fails to load. The
equality test does not measure residency, and that gap is exactly where a refactor like
this goes green in the wrong way.

## What a clean stop looks like

A partly converted attention graph passes shallow checks and fails a recall gate later,
so the correct behaviour when the window is not available is to stop with the plan and a
skipping-with-a-reason test in hand — which is what happened here, rather than a
half-applied attention handed over at the end of a long session.
