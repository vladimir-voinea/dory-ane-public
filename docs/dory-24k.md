# Dory at 24,576 tokens — caches, the tile fallback, prefill and the gates

Spec D7, D5 and the DANE-6 ticket.  Everything here is the *same* code as the
tracer bullet with a larger `context_length`; nothing about 24K is a separate
path except the tile fallback, which is a configuration of the same module.

## What the 24,576-token configuration costs

| item | rows | shape | bytes |
|---|---|---|---|
| full-attention caches | 24,576 | `(18, 4, 24576, 256)` fp16 | 864 MiB (9 x 96 MiB) |
| sliding caches | 1,280 | `(54, 4, 1280, 256)` fp16 | 135 MiB |
| total state | | | **999 MiB** |
| mask input | | `(1, 1, q_len, 25664)` fp16 | 51 KiB per decode step |

`context_length` is the number of rows of a full-attention cache and the number of
positions the caches can address; the sliding cache is
`PAD(min(ctx, window + batch), 256) = 1,280` rows, the same sizing llama.cpp uses
(llama-kv-cache-iswa.cpp:69-105), so a 64-token batch never straddles the ring
seam (`1,280 % 64 == 0`, and prefill batches start on multiples of 64).

## The tile fallback (spec D7)

A 24,576-row state is above ANEMLL's documented 16,384-axis guidance, so the spec
says: if it does not stay on the ANE inside the real chunks, split each full
cache into a 16,384-row and an 8,192-row tile and merge the two attention
partials with a log-sum-exp combine.

That is `full_cache_tiles = (16384, 8192)`:

* two Core ML states per full cache instead of one — the same total bytes, but
  every attention op is 16,384 or 8,192 keys wide instead of 24,576;
* the mask needs no new input: the full part of the concatenated mask is split by
  static column slices (`split_mask`), and a query younger than a tile finds all
  of that tile's columns `-inf`, which the combine turns into zero weight;
* the merge is `dc.merge_attention_partials`: one maximum over both tiles, one
  `exp` per tile, `sum(o_t) / sum(l_t)` — exact, and the same arithmetic a stable
  softmax uses.

The one thing a traced graph cannot do is decide *which* tile a batch's rows
belong to, because that depends on the position.  So a tiled conversion declares
one write index and one write gate per tile
(`write_index_1, write_gate_1, write_index_2, write_gate_2`), computed by the
host from the batch start, and the graph writes `current + (new - current) * gate`
into each tile at that index — a gated-off tile rewrites its own rows unchanged,
which is exact in FP16.  A batch that straddles the tile boundary cannot be
expressed that way, so `tile_write_inputs` refuses it: 64-token prefill batches
are tile-aligned (16,384 and 8,192 are multiples of 64) and a short tail goes
through the one-token function, so the runtime never asks.

In eager PyTorch (tests, the recall harness's torch backend, exact-mode scoring)
no gates are used: `_write_full` splits a straddling batch itself, so the tiled
model is exact for any batch, including one covering the whole sequence.  A test
runs the tiled and the untiled model on the same weights and tokens and compares
logits to float32 rounding, prefill and decode, across the boundary.

**Which path was converted is recorded** in the meta file (this part of the
change is held back until DANE-5's converter is merged, see PATCHES.md):
`parameters.full_cache_tiles`, `parameters.full_cache_tiled`, `kv.tiled`,
`kv.write_frame`, and one `kv.states` entry per tile.

## The sliding cache across the wrap

The shipped cache is a ring: row `pos % rows`, no copy, and an age-based mask
(`sliding_ring_mask`) so a query sees exactly the 1,024 positions in its window.
`dc.sliding_phase` names the two phases of the ANEMLL prior art (`fill` while the
window fills, `rotate` afterwards) and `dc.rotate_cache_rows` implements the
physical rotate (shift-left + append with `torch.narrow` and positive constants,
gemma3_model.py:1024-1065).  A test drives both implementations 4 x rows steps
past the wrap and pins them to the same rows; another pins the mask at the wrap
to exactly 1,024 visible slots and to the newest 1,024 positions.

The ring is the shipped path because a rotate costs a copy of all 27 sliding
caches per token and, on the ANE, a whole extra traced function per component
(the Gemma 3 `infer_rotate` / `prefill_rotate` split); the ring needs neither.

## Batched prefill and the over-context error

`DoryRuntime.prefill` feeds full 64-token batches through the `prefill` function
and the tail one token at a time through `infer` — the traced graphs have a fixed
sequence length, so there is no third path.  It calls a `progress(done, total,
elapsed)` callback every few batches; the CLI prints one line with the percentage
and tok/s, so a 24K prompt (384 predicts per loop pass) never looks hung.

`ContextOverflowError` is raised *before* any work when the prompt plus the room
generation needs does not fit the context, and from `decode` when the context is
full.  The message names both numbers and the way out, and `dory_chat.main`
prints it as one line and exits 2 — never a traceback, never silent truncation.

## Gate commands (heavy: run in an exclusive window)

```bash
# 1. the D7 decision: one 24,576-row cache or the 16,384 + 8,192 tiles.
#    Latency evidence (CPU_ONLY vs CPU_AND_NE) plus the raw compute plan; the
#    probe retries in fresh processes because this host's stateful path is flaky.
python scripts/dory_24k_placement_probe.py --model <bf16> --out <out>/probe-24k.json
python scripts/dory_24k_placement_probe.py --model <bf16> \
    --full-tile-rows 16384,8192 --out <out>/probe-24k-tiled.json

# 1b. what the conversion would do, and what it would cost (no ANE, no writes)
python -m anemll.ane_converter.dory_converter --model <bf16> --output <out> \
    --context-length 24576 --full-tile-rows 16384,8192 --dry-run

# 2. the six-code recall gate at 2 loops
python scripts/dory_recall.py --backend coreml --model-dir <out> --context 24576 \
    --loops 2 --out <out>/recall-24k-loop2.json

# 3. decode and prefill tok/s at 24K, with the resident memory
python -m anemll.dory_chat bench --model-dir <out> --context 24576 --loops 2 \
    --tokens 64
```

## Evidence

| gate | threshold | measured |
|---|---|---|
| the 24,576-row cache (or the tiles) keeps the ANE: latency evidence, CPU_ONLY vs CPU_AND_NE on one chunk | CPU_AND_NE materially faster (ratio >= 1.5), raw plan attached | *pending the exclusive window* |
| six codes at depths 0..24K, 2 loops | all six | *pending* |
| decode-mode KLD on windows beyond 16K, 2 loops | mean KLD <= 0.05, top-1 >= 93 % | *pending* |
| decode at 24K, 2 loops | >= 8 tok/s (below 6: stop and report) | *pending* |
| prefill tok/s at 24K | recorded | *pending* |
| resident memory at 24K | recorded | *pending* |

## Known risks

* The mask input is one tensor of width `context_length + sliding_rows` = 25,664,
  above the 16,384-axis guidance.  If the placement gate blames it, the next step
  is per-tile mask inputs (the mask is already split per tile internally, so the
  change is the chunk's input list, not the math).
* A tiled conversion doubles the number of state buffers every component declares
  (three instead of two); the union-of-states rule still holds because all five
  components read the same `state_attrs(config)`.
* 999 MiB of state plus the chunk weights must fit next to whatever else the
  machine is doing; the benchmark reports the process's peak RSS.


## What "placement" means on this host (2026-10-24 17:05)

An op-by-op placement number **cannot be measured here** and is therefore no
longer a gate:

* `MLComputePlan` attributes **every operation to `other`** on macOS 27.2 with
  coremltools 8.3.  DANE-5's stateless control, a printout of the tool's own
  output, reads `ane 0, gpu 0, cpu 0, other 8` -- a plan that assigns nothing.
* It cannot build a plan for a **stateful** model at all, and every Dory component
  is stateful.  So a placement percentage would be a number this toolchain
  invented, not one it measured.

The evidence the toolchain can produce is a **latency pair**: the same converted
chunk with `ComputeUnit.CPU_ONLY` and with `ComputeUnit.CPU_AND_NE`, timed after
warmup on one state object.  A graph whose work runs on the ANE is materially
faster with it enabled; a graph that does not gets the same number twice.  The
probe reports both medians, their ratio, a 1.5 threshold stated next to the
numbers, and the plan's raw lines verbatim so a reader can disagree with the tool.
It never prints a placement share.

The stateful path is also **intermittent**: the same package loads and predicts in
one fresh process and returns `error code: -14` in others, minutes apart, at load
11 and at load 340 alike.  Every heavy step therefore runs in a **fresh process**
with up to ten attempts; the artifact records how many it took, each attempt's
error text, and the load average and swap at every attempt.  A run that never
completed reports `inconclusive`, never a pass.

One consequence worth stating plainly: the converter's own gate,
`assert_ane_only`, fails when the plan attributes no ANE operation to any op --
which is exactly what this host returns for every model.  A 24K conversion here
needs `--no-placement-check`, and the latency pair above is what stands in its
place.  That is a policy decision, recorded rather than papered over.

## Status of the placement evidence (2026-10-24 07:10)

The 24K placement probe (`scripts/dory_24k_placement_probe.py`: seven blocks, one
sliding and one full attention, the real 24,576-row state union, unquantized
because quantization does not change placement) could not produce a compute plan
on this machine: **every stateful model fails with `error code: -14` while Core ML
builds the execution plan**, which is DANE-5's BLOCKER B and is *not* specific to
24,576 rows.  The evidence, all measured in one sitting at load 20-33:

| model | context | plan |
|---|---|---|
| a 3-op stateless model | — | builds: `ane 1, gpu 0, cpu 0` |
| a trivial 1-layer model with a `(4, 8)` state | — | `-14` |
| the 7-block probe, one 1,024-row full cache | 1,024 | `-14` |
| the 7-block probe, one 24,576-row full cache | 24,576 | `-14` |
| the 7-block probe, the 16,384 + 8,192 tiles | 24,576 | `-14` |

So the tiles-vs-one-cache question is **open, and it is the machine, not the
model** -- and, from the later evidence above, it is also the *toolchain*.  The
probe takes about two minutes per variant once the machine is quiet, retries in a
fresh process so one `-14` is not the answer, and is still the first thing to run
in an exclusive window.
