<!-- tracker: DANE-8 (issue id 741), project DANE, column Todo -->
**Blocked by:** nothing  |  **Blocks:** nothing

## Parent

DANE-1 — Spec: Dory1.0-4B on the M4 Max Neural Engine at 24K context

## What to build

The measurements that decide the default configuration, and the default itself. Sweep the LUT widths, measure speed, memory and energy for the Neural Engine build and for the GPU llama.cpp build under the same workloads, and set the winning configuration as the converter's and the server's default. The owner reads one table and knows what the Neural Engine costs and buys.

Measure with Studio's models unloaded, since memory pressure distorts the numbers. `powermetrics` needs sudo, so the agent prepares the exact commands and workload and the owner runs that step.

## Acceptance criteria

- [ ] Mean KLD and top-1 (decode mode) for LUT4, LUT6 and LUT8 blocks, with a LUT6 LM head, at loops 1, 2 and 4.
- [ ] Decode and prefill tok/s at 2K, 8K, 16K and 24K for loops 1, 2 and 4, plus resident memory, for the Neural Engine build.
- [ ] The same table for the GPU llama.cpp build (Q8_0, `last_loop`) on the same prompts.
- [ ] Energy per generated token for both builds at 24K and 2 loops, from `powermetrics` runs the owner performs with commands this ticket supplies.
- [ ] The chosen configuration meets every gate in the spec and is the default in the converter, the meta file and the server.
- [ ] A report is posted as a comment on this ticket: one table comparing Neural Engine and GPU, and a one-paragraph recommendation.