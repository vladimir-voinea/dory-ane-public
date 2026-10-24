<!-- tracker: DANE-6 (issue id 739), project DANE, column Todo -->
**Blocked by:** nothing  |  **Blocks:** nothing

## Parent

DANE-1 — Spec: Dory1.0-4B on the M4 Max Neural Engine at 24K context

## What to build

The pipeline from the tracer bullet, extended to 24,576 tokens (spec D7):
- full-attention caches of 24,576 rows; if the real chunks drop off the Neural Engine at that size, use the spec's fallback of a 16,384-row plus an 8,192-row tile with a log-sum-exp merge;
- 1,024-token sliding caches with fill and rotate functions;
- 64-token batched prefill with visible progress;
- a clear error when a prompt exceeds the context.

The user-visible result: paste a 24K-token document into the chat CLI, ask about its far end, and get the right answer from the Neural Engine.

## Acceptance criteria

- [ ] Every op of every chunk is still placed on the Neural Engine at 24,576 rows, or with the tile fallback, and the converter records which.
- [ ] Six codes planted at depths from 0 to 24K in one 24K prompt are all recalled at 2 loops.
- [ ] The decode-mode gate (mean KLD ≤ 0.05, top-1 ≥ 93%, 2 loops) holds on evaluation windows positioned beyond 16K tokens.
- [ ] Decode reaches ≥ 8 tok/s at 24K with 2 loops, and prefill tok/s at 24K is recorded. If decode is below 6 tok/s, stop and report on this ticket before tuning further.
- [ ] A prompt longer than 24,576 tokens gets a clear error, not a crash and not silent truncation.
- [ ] Resident memory at 24K is recorded on this ticket.