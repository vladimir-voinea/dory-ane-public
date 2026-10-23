# The Dory yardstick: evaluation set, log-probability format, scorer

Every port of Dory1.0-4B in this project - the PyTorch FP32 model, its
ANE-safe FP16 form, the converted Core ML pipeline - is judged at one seam:
**per-position next-token log-probabilities on a fixed held-out text, scored
against llama.cpp running the BF16 GGUF**. This page is the contract.

Everything below is one command away:

    python scripts/make_dory_references.py \
        --llama-perplexity ~/.unsloth/llama.cpp-dory/build/bin/llama-perplexity \
        --bf16-gguf ~/.cache/huggingface/hub/models--cloudnathan5--Dory1.0-4B-GGUF/snapshots/*/Dory1.0-4B-BF16.gguf \
        --q8-gguf  ~/.cache/huggingface/hub/models--cloudnathan5--Dory1.0-4B-GGUF/snapshots/*/Dory1.0-4B-Q8_0.gguf

It writes the reference artifacts outside the repository (default
`~/dev/dory-ane-artifacts/refs`), builds the Q8_0 proof and the
decode-vs-exact gap report, and leaves per-artifact `.meta.json` sidecars plus
JSON results in `refs/scores/`. Weights and dumps are never committed.

## The evaluation set

| | |
|---|---|
| Text | Project Gutenberg eBook #1342, *Pride and Prejudice* (public domain), chapters I and XLVI |
| Size | 8192 tokens = 2 contexts of 4096 |
| Scored | 4094 positions (the second half of each context) |
| Files | `tests/fixtures/dory_eval/eval_text.txt` (34 KB), `eval_tokens.txt` (8192 ids), `PROVENANCE.json` |
| Tokenizer | Dory's own `tokenizer.json`, `add_special_tokens=False` (the GGUF sets `add_bos_token = false`) |
| Rebuild | `python -m anemll.eval.eval_set --source pg1342.txt --tokenizer <Dory snapshot>` |

Why two contexts and 8192 tokens: `llama-perplexity` scores only the
**second half** of each context, because the first half is needed as context
(the same trick as the transformers perplexity docs). 16x4096 tokens would give
32k scored positions - what the published table used - but 8192 tokens give 4094
scored positions for one eighth of the artifact size, which is enough to
separate 0.003 from 0.03 KLD by orders of margin. The first and last positions
of a context are never scored, and every context starts from an empty KV cache.

The text is literary prose rather than the chat/code/math mix of NVIDIA's
Nemotron post-training data, which is what the published KL table used (that
text is not distributed, so it cannot be reproduced here). Consequence: the
reference perplexity on this set is ~20 rather than ~2.3, and absolute KLD
values are not numerically identical to the published table - compare them in
order of magnitude, and compare candidates against each other on the same set.
Provenance, offsets and checksums are in `PROVENANCE.json`.

## Reference log-probability artifacts

Built with llama.cpp at PR #30306, BF16 GGUF, in the two modes the spec names:

| name | mode | loops | batching | why |
|---|---|---|---|---|
| `bf16_exact_loop1.kld` | exact | 1 | `-b 4096 -ub 1024` | full recompute, loop budget 1 |
| `bf16_exact_loop2.kld` | exact | 2 | `-b 4096 -ub 1024` | **the gate reference** |
| `bf16_exact_loop4.kld` | exact | 4 | `-b 4096 -ub 1024` | full recompute, loop budget 4 |
| `bf16_decode_loop2_batch1.kld` | decode | 2 | `-b 1 -ub 1` | one token per step, as a decoder runs |
| `bf16_decode_loop2_batch64.kld` | decode | 2 | `-b 64 -ub 64` | 64-token prefill batches |

* **exact** = `dory.recurrent_kv_cache_mode=per_loop`: every loop keeps its own
  KV cache, so the logits do not depend on how tokens are batched and equal a
  full recompute. This is what the full-precision PyTorch port is scored against.
* **decode** = `dory.recurrent_kv_cache_mode=last_loop`: one KV cache per
  recurrent layer, overwritten each loop, so the result *does* depend on the
  batching the candidate actually uses. This is what a cached ANE or Core ML
  pipeline is scored against, at the batching it really runs.

Each artifact is the file `llama-perplexity --kl-divergence-base` writes
(1.07 GB at this evaluation size) plus a `<name>.kld.meta.json` sidecar
recording the exact command, loop count, KV mode, batch sizes, GGUF checksum,
evaluation-set checksums, PPL and build time.

Dump layout, so nobody has to reverse-engineer it again:

    b"_logits_"            8 bytes
    uint32 n_ctx, uint32 n_vocab, uint32 n_chunk
    int32  tokens[n_chunk * n_ctx]
    rows:  float32 scale, float32 min_log_prob, uint16 code[n_vocab]
           (plus one pad uint16 when n_vocab is odd)
           logp[v] = scale * code[v] + min_log_prob

There is one row per scored position: the second half of each context, contexts
in order. `anemll.eval.kld.read_llama_cpp_kld` reads it lazily, one block of
rows at a time.

## Candidate format (the one thing every ticket must emit)

A candidate is a **single .npz** with these arrays:

| array | dtype | shape | meaning |
|---|---|---|---|
| `logprobs` | float32 (float16 accepted) | [4094, 131072] | row r = log-softmax over the whole vocabulary at scored position r |
| `positions` | int32 | [4094] | absolute index into `tokens` of each row; row r predicts `tokens[positions[r] + 1]` |
| `tokens` | int32 | [8192] | the evaluation ids from `eval_tokens.txt`, verbatim |
| `meta` | 0-d string (JSON) | - | `format`, `model`, `loops`, `kv_mode`, `batch_size`, `n_ctx` |

The rows are the second half of every context, contexts in order, so with
2 x 4096 the positions are 2048..4094 then 6144..8190 - exactly what
`EvalSet.scored_positions()` returns and exactly what the reference
contains. Emitting a different set of positions is an error, not a rescoring.

Write it with the only writer, so the contract stays in one place:

    import numpy as np
    from anemll.eval import load_eval_set, write_candidate

    evalset = load_eval_set()                       # 8192 ids, chunk_size 4096
    tokens = evalset.token_ids
    positions = evalset.scored_positions()          # 4094 absolute positions
    logprobs = np.empty((evalset.n_rows, 131072), dtype=np.float32)
    for chunk in range(evalset.n_chunks):
        ids = tokens[chunk * evalset.chunk_size:(chunk + 1) * evalset.chunk_size]
        logits = model(ids)                         # your implementation, [4096, vocab]
        # positions first_scored .. chunk_size-2, each predicting the next token
        rows = logits[evalset.first_scored:evalset.chunk_size - 1]
        logprobs[chunk * evalset.rows_per_chunk:(chunk + 1) * evalset.rows_per_chunk] = log_softmax(rows)

    write_candidate("cand.npz", logprobs, positions, tokens,
                    meta={"model": "dory-pytorch-fp32", "loops": 2, "kv_mode": "per_loop",
                          "batch_size": 4096, "n_ctx": evalset.chunk_size})

float32 is the default. A 4094-row candidate is 2.15 GB that way and 1.07 GB at
`dtype="float16"`, which is what DANE-3's artifacts use; float16 costs about
1e-5 on a mean KLD of 0.003 and about 1e-4 on tiny ones, so use it for size and
float32 when you need the last digits. The reference itself is only 16-bit
(llama.cpp quantises to 2.4e-4 steps), so nothing below 1e-5 is real.

`score_arrays()` and `write_candidate()` validate that every position has a
successor token, so an off-by-one in the row mapping fails loudly.

## Scoring

    python -m anemll.eval.kld score         --reference ~/dev/dory-ane-artifacts/refs/bf16_exact_loop2.kld         --candidate cand.npz         --json report.json

It prints - and can gate on (`--max-kld 0.05 --min-top1 0.93` makes it exit 1
when a threshold is missed):

* **mean KLD** = `KLD(reference || candidate)`, the quantity llama.cpp's
  `--kl-divergence` reports: `sum_v p_ref(v) * (logp_ref(v) - logp_cand(v))`,
  over tokens with `logp_ref > -16` (llama.cpp's floor, change it with
  `--clip`); `--direction cand->ref` flips the direction.
* **top-1 agreement**: share of scored positions where the candidate's argmax
  token equals the reference's.
* **worst positions**: the rows with the largest KLD, with both argmax tokens
  and the actual target token - where to look first when a candidate regresses.
* The JSON report adds PPL for both sides, the 99th percentile and max per-row
  KLD, and the artifact metadata.

The scorer accepts `.kld` or `.npz` on either side, so llama.cpp files can be
compared with each other directly (that is how the decode-vs-exact gap is
measured). It refuses to compare artifacts whose `tokens` or `positions` differ, or whose
row counts or vocabulary sizes differ.

## Measured on this machine (M4 Max, macOS 27.2)

| candidate | reference | rows | mean KLD | top-1 | PPL cand / ref |
|---|---|---|---|---|---|
| **Q8_0 GGUF, exact, 2 loops** | BF16 exact loop2 | 4094 | **0.000420 ± 0.000008** | **98.534 %** | 20.205 / 20.238 |
| BF16 decode, 1 token per step | BF16 exact loop2 | 4094 | 0.006976 ± 0.000365 | 95.994 % | 19.864 / 20.238 |
| BF16 decode, 64-token batches | BF16 exact loop2 | 4094 | 0.002853 ± 0.000108 | 97.533 % | 19.987 / 20.238 |
| BF16 exact, loops 1 (context) | BF16 exact loop2 | 4094 | 0.016019 | 93.307 % | 20.729 / 20.238 |

Reference PPL on this set: **20.73 / 20.24 / 20.19** at 1 / 2 / 4 loops.
Every artifact is 1,073,283,076 bytes (1.07 GB); the six of them plus the JSON
reports are 6.4 GB.

**The last two rows are the floor for any `last_loop` candidate.** A cached model that
runs `per_loop` is exact; one that runs `last_loop` cannot do better than the
batching it uses: 0.0029 KLD / 97.5 % top-1 with 64-token prefill batches, 0.0070 /
96.0 % with one token per step. On this text the caching penalty is 7-17x the Q8_0
quantisation error, so DANE-4/DANE-5 should treat ~0.003-0.007 as their real
headroom, not 0.

**The scorer is verified against llama.cpp itself.** Scoring the Q8_0 candidate
dump with `llama-perplexity --kl-divergence --kl-divergence-base bf16_exact_loop2.kld`
prints `Mean KLD 0.000420 ± 0.000008`, `Maximum KLD 0.012116`, `99.0% KLD 0.001976`,
median 0.000333, `Same top p 98.534 %` - identical to the scorer's output to six
decimals. The plumbing check (BF16 scored against its own dump) gives 100.000 %
top-1 and a maximum KLD of 6.4e-5, which is the reference's own 16-bit rounding.

### Why Q8_0 reads 0.00042 here and 0.00297 in the published table

Same weights, same loops, same method - different text, and KLD is text
dependent. The published table used ~32k positions of NVIDIA post-training
chat/code/math data (PPL ~2.3); this set is 4094 positions of 19th-century
prose (PPL 20.2). To measure how much of the gap that explains, the same flags
were run on a templated variant of the same book - 8192 tokens of Dory chat
transcript (`<|im_start|>` / `<|im_end|>` turns, reasoning off):

| text | reference PPL | Q8_0 mean KLD | Q8_0 top-1 |
|---|---|---|---|
| prose, 2 x 4096 (this set) | 20.24 | 0.000420 | 98.534 % |
| chat-templated, 2 x 4096 | 11.69 | 0.013124 | 98.046 % |
| published (unavailable text) | 2.28 | 0.002972 | 98.852 % |

So the published value sits between the two texts we can build, and the top-1
agreement (98.5 % / 98.0 % / 98.85 %) barely moves. The prose set is the one
kept as the yardstick because its per-row KLD is well behaved (median 0.00033,
p99 0.0020, max 0.012, stderr 8e-6), while chat template tokens make a handful
of rows worth 5-13 nats and swamp the mean. The conclusion for the tickets is
unchanged: Q8_0 is nearly lossless, a `last_loop` port is bounded by ~0.003-0.007,
and the published 0.003 is the same order as both.

### Interoperability

The candidate format is the one the parallel scoring notes describe
(`logprobs` / `positions` / `tokens` / `meta`), so files written by that
`score.py` and by `write_candidate` are interchangeable. Checked on someone
else's artifacts: scoring a candidate .npz written by another ticket against its
own dump reproduces that ticket's numbers exactly (mean KLD 0.012344 vs
0.012342, top-1 99.218 % vs 99.218 %, PPL 4.8869 vs 4.8869 - the difference is
that candidate's float16 storage).

## Using it from the later tickets

* **DANE-3** (PyTorch FP32): emit `mode="exact"` candidates at loops 1, 2 and 4
  and score them against `bf16_exact_loop{1,2,4}.kld`. Gate: top-1 >= 99%,
  mean KLD <= 0.005.
* **DANE-4/DANE-5** (FP16 + Core ML on the ANE): emit `mode="decode"` candidates
  with the pipeline's own batching and score them against the matching decode
  reference. Gate: mean KLD <= 0.05, top-1 >= 93% - and the decode floor in the
  table above is the best any `last_loop` candidate can reach.
* Nothing here touches the ANE: the reference is a file, so scoring is offline
  and repeatable.

## Reproducing

    # 1. the fixtures (once; the committed ones are already there)
    curl -O https://www.gutenberg.org/cache/epub/1342/pg1342.txt
    python -m anemll.eval.eval_set --source pg1342.txt --tokenizer <Dory BF16 snapshot>

    # 2. the reference artifacts (~2 min for the five, plus the Q8_0 proof)
    python scripts/make_dory_references.py --llama-perplexity <build>/bin/llama-perplexity         --bf16-gguf <BF16.gguf> --q8-gguf <Q8_0.gguf>

    # 3. tests
    python -m pytest tests/test_dory_kld.py tests/test_dory_eval_set.py
