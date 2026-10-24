<!-- tracker: DANE-5 (issue id 738), project DANE, column Todo -->
**Blocked by:** nothing  |  **Blocks:** nothing

## Parent

DANE-1 — Spec: Dory1.0-4B on the M4 Max Neural Engine at 24K context

## What to build

The whole path end to end at a small context (1,024 tokens). One command converts NVIDIA's BF16 checkpoint into the five multifunction chunks of spec D6 — embeddings, input block, recurrent block, output block and the split LM head — with LUT6, and writes the meta file. A chat CLI then streams a reply from the Neural Engine:
- the loop count (1–4) is chosen at start and can be changed mid-session, which resets the cache and says so;
- reasoning is on by default through Dory's chat template;
- generation stops on EOS.

This is the third slice through the seam: the Core ML pipeline's log-probabilities go through the reference scorer in **decode** mode.

## Acceptance criteria

- [ ] One command produces the chunks and the meta file from the BF16 safetensors, and stops with a clear message if any file would exceed ~2 GB.
- [ ] Every op of every chunk is placed on the Neural Engine (Core ML compute plan and ANEMLL's ANE profiler), and the converter checks this itself.
- [ ] The Core ML pipeline scores mean KLD ≤ 0.05 and top-1 ≥ 93% against the decode-mode reference at 2 loops, on the evaluation text cut into 1K windows.
- [ ] The chat CLI streams a coherent answer with reasoning on at loops 1, 2 and 4; switching the loop count mid-session resets the cache and says so.
- [ ] The first-load ANE compile is cached and a second start is measurably faster.
- [ ] Decode tok/s at 1K context for loops 1, 2 and 4 is recorded on this ticket.