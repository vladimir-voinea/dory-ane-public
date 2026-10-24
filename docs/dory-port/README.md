# Dory1.0-4B on the Apple Neural Engine

A working port of **NVIDIA Nemotron-Labs Dory1.0-4B** (a 4B "looped"/depth-recurrent
model) to the Apple Neural Engine through a fork of
[ANEMLL](https://github.com/Anemll/Anemll) 0.3.5, plus the measurements we had to
make because none of them existed.

Everything here is real: the numbers are measured, the code is the code that produced
them, and the things that do not work are listed as not working.

---

## The headline result

**A Core ML *state* on this platform has a hard per-state ceiling, and it is small.**

We could not load a single attention-bearing chunk of Dory until we understood this.
The model's KV cache was declared as two aggregated Core ML states of 279 MiB. The
Neural Engine refused the model — "Failed to build the model execution plan", error
`-14` — for a day, while the same package ran fine on CPU.

The bisect that found it (all on `CPU_AND_NE`, all with the same computation, all on
macOS 27.2 with coremltools 8.3.0):

| cache declared as | total | result |
|---|---|---|
| 1 state × 12 MiB | 12 MiB | loads, predicts |
| 1 state × 16 MiB | 16 MiB | loads, predicts |
| 1 state × 18 MiB | 18 MiB | loads, predicts (3/3) |
| 1 state × 24 MiB | 24 MiB | loads, predicts |
| 1 state × 32 MiB | 32 MiB | **fails** |
| 1 state × 36 MiB | 36 MiB | **fails** (3/3) |
| 2 states × 18 MiB | 36 MiB | loads, predicts |
| 4 states × 9 MiB | 36 MiB | loads, predicts |
| 18 states × 1 MiB | 18 MiB | loads, predicts |
| 144 MiB as a graph **input/output** (no state) | 144 MiB | loads, predicts |

So: **the limit is per single state, not per program, and state count is free.** 36 MiB
in one state fails; the same 36 MiB split across two or four states passes. A cache too
large to be a state can still be passed as a tensor, at the cost of streaming it every
call.

Read the full write-up, including everything we ruled out with controls, in
[findings/ane-state-ceiling.md](findings/ane-state-ceiling.md).

## Why it matters for long context

At 24,576 tokens Dory's KV cache is 864 MiB (nine full-attention layers) plus 135 MiB
(sliding) = **999 MiB**. One layer's K+V at that length is 96 MiB — over the ceiling on
its own. One *head*'s K+V is **24 MiB**, which sits inside the measured-good band.

So the design that makes 24K reachable on the ANE is to declare the cache per
**(layer, head)**: 9 layers × 4 heads = 36 states of 24 MiB, all of them legal, none of
them offloaded. That is the change in flight at the time of writing.

## What is verified, and what is not

Verified numerically (against llama.cpp PR #30306, BF16, same overrides):

- The Dory architecture itself, re-implemented in PyTorch and reproducing the reference
  to KLD ≈ −5e−5 with 99.7–99.8 % top-1 agreement at 1, 2 and 4 loop passes.
- The **ANE-safe FP16 form** of that model — folded scales, `|alpha|` nGPT residual,
  36 physical caches with `last_loop` semantics — seven gates passing with a worst case
  of −6e−5 mean KLD and 98.44 % top-1 against thresholds of 0.01 / 98 %.
- Ten single-function Core ML packages converted at ctx 4096 / ub 64, 6.4 GiB total,
  each audited to within 1.6 % of a calibrated size prediction, with a provenance block
  recording the exact source blobs and toolchain.

Not yet measured on the ANE: per-op placement (the platform's `MLComputePlan` returns
no device assignment at all on this OS build — every op reports "other"), Core ML KLD,
decode throughput, and 24K recall. Those are blocked on the per-(layer, head) cache
layout described above, not on the model.

## Repository layout

| path | what |
|---|---|
| [docs/dory-port/spec.md](docs/dory-port/spec.md) | the specification the port was built against |
| [docs/dory-port/tickets/](docs/dory-port/tickets) | the per-ticket acceptance criteria |
| [docs/dory-port/notes/math-dory.md](docs/dory-port/notes/math-dory.md) | Dory's math, read out of the reference implementation |
| [docs/dory-port/notes/anemll-conventions.md](docs/dory-port/notes/anemll-conventions.md) | ANEMLL's chunk/state conventions, with file:line citations |
| [docs/dory-port/notes/reference-scoring.md](docs/dory-port/notes/reference-scoring.md) | how the KLD yardstick is generated and why batching matters |
| [docs/dory-port/findings/](docs/dory-port/findings) | the measurements worth reading on their own |
| [docs/dory-port/STATUS.md](docs/dory-port/STATUS.md) | living status of the port |
| `anemll/models/dory_ane_model.py` | the ANE-safe FP16 model |
| `anemll/ane_converter/dory_converter.py` | the converter |
| `anemll/dory_chat.py` | the runtime (state, loops, prefill, sampling) |
| `anemll/openai_server/` | an OpenAI-compatible server over the runtime |
| `anemll/eval/kld.py` | the scorer used for every number above |

## Reproducing the headline table

```bash
# requires coremltools 8.3.0, torch 2.5.0, numpy < 2 (coremltools 8.3 predates numpy 2)
python -m pytest tests/test_dory_*.py          # the port's own tests
scripts/dory_24k_placement_probe.py --help     # the latency/placement probe
```

The state-ceiling ladder itself is a set of tiny models with one variable — the state
shape — described step by step in the findings page, so it can be re-run and re-checked
on any machine.

## Attribution and licence

The tree is a fork of [ANEMLL](https://github.com/Anemll/Anemll) (0.3.5 beta), which is
MIT licensed; see the upstream README and licence terms. Dory1.0-4B is
NVIDIA Nemotron-Labs' model; the weights are not redistributed here, and no converted
packages or reference score files are checked in. This repository contains source,
documentation and measurement results only.
