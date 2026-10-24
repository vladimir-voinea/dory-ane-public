# Task graph — DANE (Dory1.0-4B on the M4 Max Neural Engine)

Source of truth: issue tracker project `evo.DANE` (dory-ane). Spec = DANE-1. Tickets = DANE-2..DANE-8.

```
DANE-2 (reference scorer + workspace)
   └─> DANE-3 (PyTorch FP32 model matches llama.cpp)
          └─> DANE-4 (ANE-safe FP16 + last_loop caches)
                 └─> DANE-5 (tracer bullet: convert + chat on ANE @1K)
                        ├─> DANE-6 (24K context, sliding rotation, recall)
                        └─> DANE-7 (OpenAI server + Studio Connected)
                               └─> both ─> DANE-8 (shipping config: quality/speed/energy)
```

Frontier is therefore one ticket wide until DANE-5, then two wide (DANE-6 ∥ DANE-7), then DANE-8.

Integration branch: `integration/dory-ane` (cut from origin/main = 0.3.5_beta = f4ad26d).
Integration worktree: `/Users/vladimir/dev/dory-ane-worktrees/integration`.
Per-ticket worktrees: `/Users/vladimir/dev/dory-ane-worktrees/<TICKET>` on branch `ticket/<TICKET>`.
