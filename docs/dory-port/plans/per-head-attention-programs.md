# The architecture the measurements force

*Written after the state-budget measurement. This supersedes the five-component design for
long contexts; at 4,096 the five-component design still applies.*

## The constraint

A Core ML program that does real work may keep roughly **34 MiB of state** resident
(measured: 33.8 MiB loads, 45.0 MiB fails; one 24 MiB state loads, two 48 MiB do not).
Weights do not compete with that budget — 64 MiB of weights alongside a 24 MiB state loads —
and the number of carriers is not the constraint (60 tiny carriers load).

## What that means at 24,576 tokens

| cache | size per (layer, head) | carriers that fit per program |
|---|---|---|
| full attention, 24,576 rows × 256 dim × K+V | **24.0 MiB** | **1** |
| sliding window, 1,280 rows × 256 dim × K+V | **1.25 MiB** | ~24 |

Dory has 9 full-attention layers × 4 heads = 36 full caches (864 MiB) and 27 sliding layers
× 4 heads = 108 sliding caches (135 MiB). The only shape that fits:

* **36 attention programs**, one per (layer, head), each holding exactly one 24 MiB cache.
* **~6 sliding programs**, each holding ~24 sliding caches (30 MiB).
* **MLP and projection programs** carry no state at all and are limited only by the per-file
  weight budget, so they chunk by layer group as they do today.

## Cost to expect

Every program costs a dispatch. One token through one loop pass is therefore ~36 + ~6 + a few
MLP dispatches ≈ 50 dispatches, and at two loop passes ≈ 100 per token. At 0.15-0.30 ms of
dispatch overhead each, that is 15-30 ms per token — **35-70 tok/s**, before any compute.
That is the ceiling this architecture imposes, and it is worth measuring before building it:
the first experiment is a dispatch-cost microbenchmark, not a conversion.

## The alternative worth pricing first

If dispatch overhead makes the above too slow, the lever is **smaller caches**, not fewer
programs: an int8 cache halves every number (one (layer, head) becomes 12 MiB, so two heads
fit per program, halving the dispatch count), and a sliding-only design at 24K costs 135 MiB
total — but both change the model's numerics, so they need the KLD gate behind them.

## What is already proven

* One program holding one 24 MiB 24K cache and computing real attention over it: **loads and
  predicts** (probe `ms_probe`, state (2,1,24576,256), q·Kᵀ + mask + softmax + probs·V).
* Weights up to 64 MiB alongside that state: **loads**.
* Per-(layer, head) numerics: the refactor in `ticket/DANE-5` is bit-identical to the
  aggregated layout at 4,096 and equal at 24,576, so splitting by head changes nothing in
  the math — only the packaging.

## Measured dispatch cost (the number that decides TPS)

One predict on a 24K per-head attention program — state (2,1,24576,256), real q·Kᵀ + mask +
softmax + probs·V — costs **1.167 ms** (20 iterations, warmed, CPU_AND_NE).

That single number prices the whole design:

* 36 full-attention programs per loop pass => **42 ms**; at two loop passes **84 ms**.
* Adding the sliding and MLP dispatches, one token at 24,576 tokens is **~90-120 ms**,
  i.e. **8-12 tok/s**, and that is with the model already converted and correct.

So **24K and high TPS are in direct conflict on this stack.** The work per token at 24K is
not the problem — 248 MB of cache read per loop pass is ~2.5 ms of bandwidth — the problem is
that the cache has to be spread across dozens of programs, and each program costs about a
millisecond of latency that does not shrink with its size.

Levers, in order of expected effect, all of which change either the model or the numerics:

1. **int8 cache** — halves every cache, so two heads fit per program: 18 programs, ~21 ms per
   loop pass, and roughly double the throughput. Needs the KLD gate to price the quality cost.
2. **One loop pass at long context** — halves the dispatch count outright; Dory's loop count is
   a runtime choice, and the KLD penalty of 2 loops -> 1 loop is measurable with the existing
   reference (loops-1 floors are already recorded).
3. **Shorter context** — at 8,192 a per-head cache is 8 MiB, so four heads fit per program:
   9 programs instead of 36, and ~4x the throughput.

What this rules out is the thing worth ruling out: a single-program or few-program 24K design.
The state budget forbids it, and no packaging trick changes that.

