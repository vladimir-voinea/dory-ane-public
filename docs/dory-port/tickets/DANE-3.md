<!-- tracker: DANE-3 (issue id 736), project DANE, column Todo -->
**Blocked by:** nothing  |  **Blocks:** nothing

## Parent

DANE-1 — Spec: Dory1.0-4B on the M4 Max Neural Engine at 24K context

## What to build

An ANEMLL-style Dory model class that loads NVIDIA's BF16 safetensors and runs a whole sequence in full precision with no cache (a full recompute), with the loop count as an argument. It implements spec D3 exactly:
- the 16 / 40 / 16 block layout with alternating attention and MLP;
- the nGPT residual with `|α|`;
- gated attention, with q and k L2-normalized per head and the `sqk` scale;
- the MLP with its `suv` scales;
- NEOX RoPE with θ 1e4 (sliding) and 1e7 (full), and the 1,024-token sliding window;
- loop entry and exit;
- the logit scale.

It lives beside ANEMLL's Gemma 3 model and follows its config conventions. llama.cpp + PR #30306 is the authority wherever this ticket or the spec is ambiguous. This is the first slice through the seam: the model's log-probabilities go through the reference scorer in **exact** mode.

## Acceptance criteria

- [ ] The model loads the BF16 safetensors with every tensor accounted for: no missing and no unused weights.
- [ ] At loops 1, 2 and 4, the scorer reports top-1 agreement ≥ 99% and mean KLD ≤ 0.005 against the exact-mode reference.
- [ ] Greedy generation from a short chat prompt, rendered with the repo's chat template, produces the same first 32 tokens as llama.cpp at 2 loops.
- [ ] Changing the loop count needs no change to the weights or the model, only the argument.