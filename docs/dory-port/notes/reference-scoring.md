# Reference log-probabilities from llama.cpp, and scoring against them

**Status:** authoritative for DANE-2 and for every ticket that reports KLD or top-1.
Everything below was read from the source or measured on this Mac on 2026-10-23 with the Q8_0
GGUF; nothing is inferred from memory. Measured numbers carry the flags that produced them.

**The one sentence:** the reference is a `_logits_` file written by
`llama-perplexity --save-all-logits` (uint16-quantised log-probabilities, second half of each chunk
only), and the candidate is an `.npz` holding float32 `[positions, 131072]` log-probabilities over
exactly the same positions.

---

## 0. TL;DR - the commands

```bash
# --- paths -------------------------------------------------------------------
PPL=/Users/vladimir/.unsloth/llama.cpp-dory/build/bin/llama-perplexity
BF16=<path to Dory1.0-4B-BF16.gguf>          # not on disk yet; another agent owns it
Q8=/Users/vladimir/.cache/huggingface/hub/models--cloudnathan5--Dory1.0-4B-GGUF/snapshots/9abcb0453c0382e6852fb57893102a7ed0fe17be/Dory1.0-4B-Q8_0.gguf
EVAL=<eval text, at least 4096 tokens for the settings below>

# --- exact mode (per_loop, full recompute), loops 1 / 2 / 4 -------------------
for L in 1 2 4; do
  "$PPL" -m "$BF16" -f "$EVAL" -c 1024 -b 1024 -ub 1024 --chunks 4 \
         -ngl 999 -fa on \
         --override-kv "dory.num_loops=int:$L,dory.recurrent_kv_cache_mode=str:per_loop" \
         --save-all-logits ref-bf16-exact-loop$L.kld
done

# --- decode mode (last_loop), 2 loops, (a) one token per step ----------------
"$PPL" -m "$BF16" -f "$EVAL" -c 1024 -b 1 -ub 1 --chunks 4 -ngl 999 -fa on \
       --override-kv "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop" \
       --save-all-logits ref-bf16-decode-loop2-step1.kld

# --- decode mode (last_loop), 2 loops, (b) 64-token batches ------------------
"$PPL" -m "$BF16" -f "$EVAL" -c 1024 -b 64 -ub 64 --chunks 4 -ngl 999 -fa on \
       --override-kv "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop" \
       --save-all-logits ref-bf16-decode-loop2-b64.kld

# --- score any GGUF candidate against one of those files ---------------------
"$PPL" -m candidate.gguf -c 1024 -ngl 999 -fa on \
       --override-kv "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:per_loop" \
       --kl-divergence-base ref-bf16-exact-loop2.kld --kl-divergence
```

The last command needs **no** `-f`: under `--kl-divergence` llama-perplexity decodes the token ids
stored inside the base file (perplexity.cpp:1716-1720), so a scoring run cannot drift from the
reference text.

---

## 1. What is authoritative, and where it lives

| Thing | Path |
|---|---|
| Reference implementation | /Users/vladimir/.unsloth/llama.cpp-dory - commit 01ad20223 ("merge ggml-org/llama.cpp#30306 (dory)") |
| Dory model graph | `src/models/dory.cpp` (200 lines) |
| HF-to-GGUF converter | `conversion/dory.py` (114 lines), registered on `DoryForCausalLM` |
| Perplexity / KL tool | `tools/perplexity/perplexity.cpp` (2075 lines); flags in `common/arg.cpp` |
| Reference binary | `build/bin/llama-perplexity` (built; `libllama-perplexity-impl.dylib`) |
| Weights | Q8_0 GGUF cached (section 0); BF16 GGUF **not yet on disk** |

Dory is a looped (depth-recurrent) model, which matters everywhere below: it declares 72 physical
blocks but expands them into **virtual layers**, n_virtual = n_input + n_recurrent * n_loops +
n_output (src/models/dory.cpp:29) - 72 at 1 loop, 112 at 2, 192 at 4.

---

## 2. The two switches: loop count and KV mode

**There is no command-line flag for either.** Both are GGUF metadata read once at model load
(src/models/dory.cpp:7-13):

| GGUF key | Type | Values | Effect |
|---|---|---|---|
| `dory.num_loops` | uint32 | 1..4 (n_loops > 4 throws, dory.cpp:30) | how many times the recurrent block runs |
| `dory.recurrent_kv_cache_mode` | string | `"per_loop"` or `"last_loop"` (anything else throws) | one KV per *virtual* layer vs one KV per *physical* recurrent block |

They are set at conversion time by conversion/dory.py:100-102 from `n_recurrent_loops` and
`recurrent_kv_cache_mode` in the HF `config.json` (NVIDIA's public config says **2**; the community
GGUF was converted with **4**).

At run time both are overridden with `--override-kv` (common/arg.cpp:2965: "types: int, float, bool,
str"), comma-separated in one argument:

```bash
--override-kv "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:per_loop"
```

**Always pass both.** The cached Q8_0 GGUF ships with

```
dory.num_loops                = 4
dory.recurrent_kv_cache_mode  = 'per_loop'
```

(dumped from the file), so a run without the override is a **4-loop** run - silently comparing the
wrong thing.

What the mode does in the graph (dory.cpp:73, and the comment at dory.cpp:169):

```cpp
cache_layer.push_back(recurrent && recurrent_kv_cache_mode == "last_loop" ? p : il);
// Repeated writes replace only the current chunk; older tokens retain final-loop KV.
```

so in `last_loop` every loop pass of the recurrent block **overwrites** the same KV rows, and a token
attends over (a) the final-loop KV of every token outside its current batch and (b) the current-loop
KV of the tokens inside its current batch. Everything in section 4 follows from that.

`-ngl` counts *virtual* layers: pass `-ngl 999`, never a physical layer count (`-ngl 36` at 4 loops
would leave 156 virtual layers on the CPU - the community README warns about this).

---

## 3. Exact mode - `per_loop`, full recompute

`per_loop` gives every virtual layer its own KV rows, so the result is the model as trained and does
not depend on batching (measured in section 6). llama-perplexity clears the KV cache at every chunk
(perplexity.cpp:551), so each chunk is an independent document decoded from position 0 - a full
recompute by construction.

### The command

```bash
"$PPL" -m "$BF16" -f "$EVAL" -c 1024 -b 1024 -ub 1024 --chunks 4 \
       -ngl 999 -fa on -t 10 \
       --override-kv "dory.num_loops=int:$L,dory.recurrent_kv_cache_mode=str:per_loop" \
       --save-all-logits ref-bf16-exact-loop$L.kld
```

| Flag | Why exactly this |
|---|---|
| `-c 1024` | chunk length and KV window per chunk; half context, half scored (section 5) |
| `-b 1024` | one llama_decode per chunk (num_batches = n_ctx/n_batch = 1); must equal -c (trap below) |
| `-ub 1024` | one physical ubatch per chunk - irrelevant in per_loop, essential in last_loop (section 4) |
| `--chunks 4` | uses the first 4*1024 tokens; the tool asserts tokens >= 2*n_ctx |
| `-ngl 999` | offload every virtual layer to Metal |
| `-fa on` | pinned; auto resolves to on here, off shifts results (3.2e-6 KLD, 0.037 nats; section 6) |
| `-t 10` | thread count; measured **identical** output at -t 4 and -t 10, so it is speed only |
| `--save-all-logits` | writes the reference file; alias of --kl-divergence-base (arg.cpp:2524); logs "perplexity: saving all logits to ..." |
| `-f` | the eval text; params.escape = false (perplexity.cpp:1994), so the file is used literally |

**Trap - `-b` must equal `-c`.** llama_perplexity() recomputes the context:
n_parallel = max(1, n_batch/n_ctx); params.n_ctx = n_parallel * n_ctx (perplexity.cpp:2011-2015).
`-c 1024 -b 4096` silently makes the context 4096 tokens and decodes four chunks per call as four
sequences. It still "works" in per_loop, but it is not the configuration a candidate reproduces.

### What one run costs (measured, Q8_0 at 2 loops, Metal, M4 Max)

| Setting | Time |
|---|---|
| `-c 1024 -b 1024`, one chunk (1024-token prefill) | 1.8 s |
| 4 chunks, whole run including model load | 8.7 s |
| `-c 1024 -b 64`, one chunk | 3.3 s |
| `-c 1024 -b 1`, one chunk (1024 single-token steps, about 31 ms per token) | 31.8 s |

BF16 is twice the bytes - budget roughly double. The three exact references cost about a minute and
**1.6 GB** of disk; the size table is in section 5.

---

## 4. Decode mode - `last_loop` with the candidate's own batching

Same command, two changes: the KV mode and the batch size.

```bash
# (a) one token per step       -> candidate decode loop
"$PPL" -m "$BF16" -f "$EVAL" -c 1024 -b 1  -ub 1  --chunks 4 -ngl 999 -fa on \
       --override-kv "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop" \
       --save-all-logits ref-bf16-decode-loop2-step1.kld

# (b) 64-token prefill batches -> candidate prefill loop
"$PPL" -m "$BF16" -f "$EVAL" -c 1024 -b 64 -ub 64 --chunks 4 -ngl 999 -fa on \
       --override-kv "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop" \
       --save-all-logits ref-bf16-decode-loop2-b64.kld
```

**The batch boundary is `-ub`, not `-b`.** Measured on Q8_0 at 2 loops, 256-token chunk, every row
compared against the exact reference:

| run | mean KLD vs exact | top-1 | max abs dlogp | PPL |
|---|---|---|---|---|
| per_loop -b 256 (exact, base) | 0 | 100 % | 0 | 5.7131 |
| per_loop -b 64 | 0 | 100 % | 0 | 5.7131 |
| per_loop -b 256 -ub 64 | 0 | 100 % | 0 | 5.7131 |
| per_loop -b 1 | 1.6e-6 | 100 % | 0.030 | 5.7117 |
| last_loop -b 256 (whole chunk, one ubatch) | **0** | 100 % | 0 | 5.7131 |
| last_loop -b 256 -ub 64 | 3.46e-3 | 97.64 % | 7.89 | 5.6977 |
| last_loop -b 64 | **3.46e-3** | 97.64 % | 7.89 | 5.6977 |
| last_loop -b 1 | 4.72e-3 | 97.64 % | 7.49 | 5.7282 |
| per_loop, -fa off | 3.2e-6 | 100 % | 0.037 | 5.7146 |

Three facts to carry forward:

1. **One ubatch covering the whole chunk makes last_loop identical to per_loop** (zero difference at
   every position). last_loop is only a different model when the chunk is cut into ubatches.
2. **-b 256 -ub 64 and -b 64 -ub 64 are byte-identical.** n_batch beyond the ubatch size changes
   nothing.
3. **The floor a last_loop candidate can reach is about 0.005 KLD / 97.6 % top-1** at 2 loops (rows
   6-8) - the same order as the Q8_0-vs-BF16 quantisation error. DANE-2 asks for this number per loop
   count; record it.

Decode mode is not cheap to produce: the reference must decode in the candidate's batching, so
`-b 1` costs O(n^2) work (31 s per 1024-token chunk at 2 loops on Q8_0).

---

## 5. The on-disk reference format

Written by `perplexity()` (perplexity.cpp:458-468, 520-526, 607-621) and read back by
`kl_divergence()` (perplexity.cpp:1688-1720). Little endian throughout.

```
offset  type                     field
0       char[8]                  "_logits_"
8       int32                    n_ctx        # the -c chunk length used to write it
12      int32                    n_vocab      # 131072 for Dory
16      int32                    n_chunk      # --chunks
20      int32[n_chunk * n_ctx]   tokens       # the evaluated ids, chunk after chunk
...     rows                     n_chunk * (n_ctx//2 - 1) rows of nv uint16,
                                  nv = 2*((n_vocab+1)//2) + 4 = n_vocab + 4 for an even vocab
```

Each **row** (perplexity.cpp:79-107) is

```
bytes 0..3    float32  scale         = (max_logit - min_logit) / 65535
bytes 4..7    float32  min_log_prob  = min_logit - max_logit - log_sum_exp
bytes 8..     uint16   q[n_vocab]    ->  logp[v] = min_log_prob + scale * q[v]
```

with min_logit = max(min_logit, max_logit - 16): **the stored log-probabilities are clipped at
max_logit - 16**, everything below decodes to q = 0 and reads as min_log_prob. For Dory's even
131072 vocab a row is exactly 262152 bytes, the last 4 of them unused.

**Rows, and what they mean.** first = n_ctx/2; each chunk contributes n_ctx - 1 - first rows
(perplexity.cpp:613) covering the *input* positions first .. n_ctx-2 of that chunk, and row i holds
the distribution that predicts tokens[first + i + 1]. The first half of every chunk is context only
and is never scored. Total rows = n_chunk * (n_ctx//2 - 1).

**File size** = 20 + 4*n_chunk*n_ctx + rows * 262152. Measured to the byte:

| -c | chunks | rows | bytes | note |
|---|---|---|---|---|
| 256 | 1 | 127 | 33,294,348 | probe |
| 1024 | 1 | 511 | 133,963,788 | probe |
| 1024 | 4 | 2044 | 535,855,092 | the recommended eval shape |
| 4096 | 16 | 32,752 | 8.59 GB | cloudnathan5's published settings |

### Minimal reader (verified: reproduces the tool's own PPL and KLD)

```python
import numpy as np

def read_kld(path):
    with open(path, "rb") as fh:
        hdr = fh.read(20)
    assert hdr[:8] == b"_logits_"
    n_ctx, n_vocab, n_chunk = (int(x) for x in np.frombuffer(hdr[8:20], dtype="<i4"))
    nv = 2 * ((n_vocab + 1) // 2) + 4
    rows = n_chunk * (n_ctx - 1 - n_ctx // 2)
    tokens = np.fromfile(path, dtype="<i4", offset=20, count=n_chunk * n_ctx).astype(np.int64)
    raw = np.memmap(path, dtype="<u2", mode="r",
                    offset=20 + 4 * n_chunk * n_ctx, shape=(rows, nv))
    h = np.ascontiguousarray(raw[:, 0:4]).view("<f4").reshape(-1, 2)   # scale, min_log_prob
    logprobs = raw[:, 4:4 + n_vocab].astype(np.float32) * h[:, 0:1] + h[:, 1:2]
    first = n_ctx // 2
    positions = np.concatenate([np.arange(c * n_ctx + first, c * n_ctx + n_ctx - 1)
                                for c in range(n_chunk)]).astype(np.int64)
    return dict(n_ctx=n_ctx, n_vocab=n_vocab, n_chunk=n_chunk, tokens=tokens,
                logprobs=np.asarray(logprobs), positions=positions, first=first)
```

**Alignment to the text.** positions[p] is the index into the tokenized eval text of the row's input
position; the row predicts tokens[positions[p] + 1]. Chunks are consecutive and non-overlapping,
each decoded from position 0 with a cleared cache, so position 1023 of chunk 0 is never predicted
and position 1024 opens chunk 1 as a context-only token.

**Verified** on a 611-token English text, `-c 256 --chunks 1`, Q8_0 at 2 loops:

* the token ids in the file are byte-identical to AutoTokenizer output with
  `add_special_tokens=False` - **no BOS is inserted**, because tokenizer.ggml.add_bos_token = false
  (perplexity.cpp:455, 567);
* exp(-mean logp[tokens[pos+1]]) over the 127 rows = **5.7131**, exactly the "Final estimate: PPL"
  the tool printed - which proves the row -> position -> target alignment;
* sum(exp(logp)) per row = 1.000-1.015 (uint16 grid noise, step about 2.4e-4 nats).

---

## 6. Exact mode is batching-independent - measured

`per_loop` at `-b 256`, `-b 64` and `-b 256 -ub 64` produced **bit-identical** files. `-b 1`
differed by 1.6e-6 mean KLD (one position at 0.03 nats) purely because single-token mat-vec and
batched mat-mul are different Metal kernels; top-1 was 100 % everywhere. `-fa off` is a real, if
tiny, change (3.2e-6 KLD, 0.037 nats) - pin `-fa on`. Thread count (`-t 4` vs `-t 10`) changed
nothing at all.

So: **exact mode = per_loop, any -b/-ub, -fa on**; candidates should agree to about 1e-5.

---

## 7. The evaluation text

* Needs **at least 2 x n_ctx** tokens or the tool refuses (perplexity.cpp:478); only the first
  n_chunk * n_ctx tokens are used, the rest is dropped.
* Tokenised with add_special_tokens=true but the GGUF says add_bos_token=false, so the ids are plain
  tokenizer.encode(text, add_special_tokens=False) - verified against HF on a sample.
* Scored positions per chunk: n_ctx//2 - 1. A **4096-token** text at `-c 1024 --chunks 4` gives
  4 x 511 = **2044 scored positions**, with the first 512 tokens of each chunk acting as context. If
  the gates need about 4K *scored* positions the text must be about 8K tokens (use `--chunks 8`).
* cloudnathan5's own text is held-out data from Nemotron-Post-Training-Dataset-v1 rendered with
  Dory's chat template, "disjoint from calibration" (their README, Evaluation). Ours should look like
  that - chat/QA/code prose, not wikitext - because KLD is text-dependent.

Their BF16 PPL on that text is 2.340 / 2.284 / 2.291 at 1 / 2 / 4 loops: a sanity number, not a gate.

---

## 8. Candidate format and scoring

### 8.1 The one candidate format (`*_logprobs.npz`)

```python
np.savez_compressed(
    path,
    logprobs  = float32 [N, 131072],   # log p_candidate(v | prefix) at each scored position
    positions = int32   [N],           # absolute positions, identical to ref["positions"]
    tokens    = int32   [T],           # the full tokenized eval text the candidate evaluated
    meta      = json.dumps({           # informational, checked by the scorer
        "format": "dory-logprobs-1", "model": "...", "loops": 2,
        "kv_mode": "per_loop" | "last_loop", "batch_size": 1024 | 64 | 1,
        "n_ctx": 1024, "chunks": 4, "text_sha256": "...", "dtype": "float32"}),
)
```

Producer rules: the same tokens as the reference, the same chunking (each chunk from position 0 with
a fresh cache), rows for positions n_ctx//2 .. n_ctx-2 of every chunk, full vocab in model order.
float32 is the default; float16 halves the file (1.07 GB -> 536 MB at 2044 rows) at a cost of about
1e-4 KLD - fine for candidates, never for the reference.

### 8.2 The formulas (llama.cpp's own, perplexity.cpp:191-252)

For each scored position i, with q = exp(logq) the reference row and p = exp(logp) the candidate row:

```
KLD_i  = sum over v of  q_v * (log q_v - log p_v)   restricted to { v : log q_v > -16 }
top1_i = [ argmax_v log p_v == argmax_v log q_v ]   (ties -> lowest id on both sides)
PPL    = exp( mean_i( -log p_i[target_i] ) )
```

Report mean KLD +- stderr, median, p90, p99, max, top-1 %, PPL(cand), PPL(ref),
RMS dp = sqrt(mean_i (p_i[target] - q_i[target])^2), and the **worst positions** (top-5 by KLD, with
position, KLD and target token id).

The -16 restriction is not optional: it is both llama.cpp's rule and the consequence of the reference
being clipped at max_logit - 16. It drops mass below e^-16 (about 1.1e-7) without renormalising,
which is what makes our numbers comparable to the published table.

### 8.3 The scorer, and proof that it matches llama.cpp

`score.py` (in `scratch/ref-c/`) implements exactly the above. Verified end to end: reference =
per_loop file, candidate = the same model's `last_loop -b 64` file converted to the npz format.

| | llama-perplexity --kl-divergence | our score.py |
|---|---|---|
| Mean KLD | 0.003471 +- 0.000521 | 0.003469 +- 0.000521 |
| 99.0 % KLD | 0.028396 | 0.028399 |
| Maximum KLD | 0.034883 | 0.034886 |
| Same top p | 97.638 % | 97.638 % |
| Mean PPL(Q) / (base) | 5.697748 / 5.713067 | 5.697744 / 5.713068 |

The 5e-7 difference is only because the candidate there is a re-read `.kld` (already quantised)
while llama.cpp scores raw float logits.

```python
# score.py --base ref.kld --cand cand.npz   (read_kld() as in section 5, then)

def load_candidate(path):
    z = np.load(path, allow_pickle=False)
    return dict(logprobs=z["logprobs"].astype(np.float32, copy=False),
                tokens=z["tokens"].astype(np.int64),
                positions=z["positions"].astype(np.int64),
                meta=json.loads(str(z["meta"])) if "meta" in z else {})

def score(ref, cand):
    if cand["logprobs"].shape[1] != ref["n_vocab"]:
        sys.exit("vocab mismatch")
    if not np.array_equal(ref["tokens"], cand["tokens"]):
        sys.exit("token mismatch: the candidate must reproduce the reference tokenization")
    if not np.array_equal(ref["positions"], cand["positions"]):
        sys.exit("position mismatch: emit exactly the reference rows")
    logq, logp = ref["logprobs"], np.maximum(cand["logprobs"], -100.0)
    m = logq > -16.0
    q = np.exp(logq, where=m, out=np.zeros_like(logq))
    kld = np.where(m, q * (logq - logp), 0.0).sum(axis=1)
    top1 = logp.argmax(1) == logq.argmax(1)
    target = ref["tokens"][ref["positions"] + 1]
    return kld, top1, target
```

A candidate that is itself a GGUF never touches the npz: score it with
`--kl-divergence --kl-divergence-base <ref>.kld` (section 0), which is the same arithmetic.

---

## 9. Reproducing cloudnathan5's published numbers

Their `tools/eval_kld.sh` (saved at `scratch/ref-c/eval_kld.sh`) is the whole method, verbatim in
substance: one base file per loop count, then every quant scored against it.

```bash
CTX=4096; CHUNKS=16
# base (BF16 GGUF), once per loop count -> eval/base-bf16-loop$L.kld (~8.6 GB, deleted afterwards)
"$PPL" -m "$BF16" -f "$TEXT" -c 4096 -b 4096 -ub 1024 --chunks 16 -ngl 999 \
       --override-kv "dory.num_loops=int:$L" --kl-divergence-base "eval/base-bf16-loop$L.kld"
# candidate
"$PPL" -m "$QUANT" -f "$TEXT" -c 4096 -b 4096 -ub 1024 --chunks 16 -ngl 999 \
       --override-kv "dory.num_loops=int:$L" --kl-divergence \
       --kl-divergence-base "eval/base-bf16-loop$L.kld"
```

The printed lines `Mean KLD`, `99.0% KLD`, `Maximum KLD` and `Same top p` are scraped into
`eval/kld_results.csv` by `tools/summarize_kld.py` (which regenerates `eval/kld_table.md`). They do
not override the KV mode - their BF16 GGUF is per_loop - and they need no `-f` for the scoring half.

Published Q8_0, exact mode (`eval/kld_table.md` and `eval/kld_results.csv`):

| loops | mean KLD | top-1 | 99 % KLD | max KLD |
|---|---|---|---|---|
| 1 | 0.002031 | 98.934 % | 0.025343 | 6.565920 |
| 2 | **0.002972** | **98.852 %** | - | - |
| 4 | 0.004186 | 98.837 % | - | - |

The spec quotes 0.003 / 98.9 % for Q8_0 and 0.030 / 96.0 % for Q4_K_M at 2 loops; both match this
table (Q4_K_M is 0.029759 / 95.979 %).

**Their eval text is not published.** `eval/heldout.txt` returns 404; the repository carries only
`eval/kld_table.md` and `eval/kld_results.csv`. Exact reproduction of 0.002972 is therefore
impossible. The honest proof is:

1. build our own fixed 4096-token chat-formatted text (section 7);
2. produce `ref-bf16-exact-loop2.kld` from the **BF16 GGUF**;
3. score **Q8_0** against it with `--kl-divergence`, per_loop, num_loops=2;
4. accept **0.002-0.006 KLD and at least 98.5 % top-1** as "near the published figures", and record
   the exact number. A result of 0.03 would mean the scorer is wrong, not the model.

Until the BF16 GGUF lands, Q8_0 scored against itself must give exactly 0.000000 and 100.000 % -
that is the plumbing smoke test, and it passes today.

One conversion detail: their GGUFs were built with `conversion/dory.py` **patched** to keep the 1-D
scale vectors in F32 (`llama.cpp/llama.cpp-dory-f32-scales.patch`, saved at
`scratch/ref-c/published/dory-f32-scales.patch`). The unpatched converter writes BF16/F16 scales for
a BF16/F16 GGUF. The graph casts them to F32 anyway, so it is a CUDA speed fix and not a numerics
fix - but a BF16 GGUF built with the patch is the closer match to theirs.

---

## 10. Tokenizer and prompt facts

From `hf/tokenizer.json`, `hf/tokenizer_config.json`, `hf/generation_config.json`, `hf/config.json`
(downloaded from nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16) and the GGUF tokenizer section; ids
verified with transformers 4.56 (PreTrainedTokenizerFast, 131072 entries, BPE, pre-tokenizer
"pixtral").

| item | value |
|---|---|
| BOS | **1** = `<s>` (never added automatically: add_bos_token=false) |
| EOS | **11** = `<|im_end|>` (tokenizer.ggml.eos_token_id; generation_config.eos_token_id = [2, 11]) |
| also end-of-text | **2** = `</s>` (in generation_config, not the tokenizer's eos) |
| padding | 11 |
| unk | 0 |
| `<|im_start|>` | **10** |
| ` thinking`, the added special token the template emits | **12** |
| ` thinking`, the plain-BPE merge (what tokenizing raw text gives) | 11483 |
| `</think>`, the added special token the template emits | **13** |
| newline | 1010 |
| vocab / context | 131072 / 262144 |

### Chat template (GGUF `tokenizer.chat_template`; identical to `hf/chat_template.jinja`)

System + user, **reasoning on** (enable_thinking=True, the default). The escapes below are literal
template text: a real newline follows each tag.

```
<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{USER}<|im_end|>\n<|im_start|>assistant\n thinking\n
```

**Reasoning off** (enable_thinking=False): same prefix, generation prompt is
`<|im_start|>assistant\n` followed by `</think>` and **no trailing newline**.

```
<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{USER}<|im_end|>\n<|im_start|>assistant\n</think>
```

Verified ids for a two-message example with reasoning on:

```
[10, 25708, 1010, 4568, 1584, 1261, 20351, 27089, 1046, 11, 1010, 10, 3263, 1010,
 7493, 1395, 1032, 1050, 1043, 1050, 1063, 11, 1010, 10, 1503, 19464, 1010, 12, 1010]
```

Read the tail carefully: reasoning on ends `... 10, 1503, 19464, 1010, 12, 1010`, i.e. `<|im_start|>` + `ass` + `istant` + newline + ` thinking` + newline; reasoning off ends `... 1010, 12, 13`, i.e. the same `\n` + ` thinking` + `</think>`. So the markers are tokens **12** (` thinking`) and **13** (`</think>`),
both with the leading space included; **11483** is the ordinary BPE merge of " thinking" and is what you get when
tokenizing that text outside the template - a candidate that hard-codes 11483 will be off by one token.

Other template facts:

* the system block is **always** emitted; with no system message it renders
  `<|im_start|>system\n<|im_end|>\n` - empty content, not omitted;
* assistant turns before the last user turn have their thinking collapsed to ` thinking</think>`
  (truncate_history_thinking defaults to true);
* medium_effort=True appends `\n\n{reasoning effort: efficient}` to the last user turn;
* tools render as `<tools>/<function>/<parameter>` XML inside the system block, and calls come back
  as `<tool_call>\n<function=NAME>\n<parameter=P>\nvalue\n</parameter>\n</function>\n</tool_call>`;
* stop on **`<|im_end|>` (11)**; `</s>` (2) is also listed in generation_config.json.

### Sampling defaults (NVIDIA's recommendation)

`generation_config.json`: temperature 1.0, top_p 0.95, do_sample true; the GGUF carries
general.sampling.temp = 1.0 and general.sampling.top_p = 0.95; the spec adds top_k 0, min_p 0.
llama-server's own defaults differ (top_k 40, min_p 0.05, visible in its generation_settings
echo), so spell them out:

```bash
llama-server -m Dory1.0-4B-Q8_0.gguf -ngl 999 -c 32768 \
  --temp 1.0 --top-p 0.95 --top-k 0 --min-p 0 \
  --override-kv "dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:per_loop"
```

Greedy (temperature 0, top_k 1) for reproducible tests.

---

## 11. Gotchas

1. **The shipped GGUF is 4 loops, per_loop.** Every run needs
   `--override-kv dory.num_loops=int:L,dory.recurrent_kv_cache_mode=str:MODE`; forgetting it silently
   scores at 4 loops.
2. **`-ub` is the batch boundary in last_loop, not `-b`.** `-b 256 -ub 64` is byte-identical to
   `-b 64`; `-b 256 -ub 256` collapses to per_loop. The default `-ub` is 512 (common.h:475), so a
   "64-token batch" run without `-ub 64` is really a 512-token-batch run.
3. **`-b` must equal `-c` for perplexity**, or the context is multiplied by n_parallel
   (perplexity.cpp:2011-2015).
4. **Never pass `--ppl-stride`**: it switches to `perplexity_v2()`, which writes no logits file at
   all (perplexity.cpp:443).
5. **The reference is clipped at max_logit - 16.** Score with the same restriction or the KLD will
   not match the published table.
6. **`-fa off` changes numerics** (3.2e-6 KLD, 0.037 nats). `auto` resolves to on for this model on
   Metal; pin `-fa on` on both sides.
7. **`-ngl 999`, not a layer count**: `-ngl` counts virtual layers (112 at 2 loops, 192 at 4).
8. **No BOS.** add_bos_token=false, so the reference text is not BOS-prefixed; a candidate that
   prepends `<s>` is misaligned at every row.
9. **Chunks are independent.** The cache is cleared per chunk, so chunk k never sees chunk k-1; a
   candidate that runs the text as one stream disagrees at every chunk boundary.
10. **Only the second half of each chunk is scored** (n_ctx//2 - 1 rows), and the last token of a
    chunk is never predicted.
11. **Prompt-cache reuse does not affect the reference** (perplexity clears the cache per chunk) but
    it does affect the server: a repeated one-token request reported cache_n = prompt_tokens - 5 and
    re-evaluated the tail. Numerically harmless; it only makes server timings meaningless.
12. **The server cannot produce full-vocab log-probabilities.** `echo` is rejected in
    server-common.cpp:1170 ("Only no echo is supported") - on this build it was silently ignored on
    `/v1/completions` and a normal completion came back - and `n_probs` / `top_logprobs` return only
    the top-K. Verified: `n_probs: 8` on `/v1/completions` returned 8 entries. That is a spot-check
    tool (top-1, and the gold token's logprob when it is in the top-K), never a KLD source. Passing
    `top_logprobs` alone yielded a single entry in our test; pass the non-standard `n_probs` field
    explicitly.
13. **Memory.** The BF16 GGUF is about 8 GB and Metal keeps a large residency set; unload Studio's
    models and any other llama.cpp process before producing references, or you will measure swap.
14. **Disk.** One reference file is 511 MB per 2044 positions, and 8.6 GB at cloudnathan5's 16x4096
    settings. Five references (3 exact + 2 decode) at `-c 1024 --chunks 4` = 2.7 GB; together with
    the 8 GB BF16 GGUF, 36 GB free gets tight.
15. **`--save-all-logits` truncates** its target (ofstream, binary). Never point it at a reference
    you want to keep.
16. **`-f` is ignored under `--kl-divergence`**: the tokens come from the base file, so a wrong text
    argument is not noticed.

---

## 12. Unknowns that block DANE-2

1. **The BF16 GGUF is not on disk.** Every reference file needs it. Until another agent finishes it,
   only Q8_0-against-Q8_0 plumbing checks can run (they pass).
2. **cloudnathan5's eval text is unpublished** (`eval/heldout.txt` -> 404), so their exact
   0.002972 / 98.852 % cannot be reproduced. Decision needed: use our own fixed 4096-token
   chat-formatted text and read the Q8_0 proof as "0.002-0.006 KLD, at least 98.5 % top-1"
   (recommended), or ask the publisher for the text.
3. **Whether their BF16 GGUF used the f32-scales patch** is unknown. It cannot change the numbers
   (the graph casts scales to F32), only speed on CUDA.
4. **Decode-mode cost at the real eval size** is about 2 min per loop count on Q8_0 and 4-5 min on
   BF16 (`-b 1`), so plan minutes, and do not run it while the conversion work is running.

---

## Appendix - scratch artefacts (all under `scratch/ref-c/`)

| file | what |
|---|---|
| `eval_kld.sh`, `summarize_kld.py` | the published methodology, verbatim |
| `hf_eval/kld_table.md`, `hf_eval/kld_results.csv` | the published results |
| `published/dory-f32-scales.patch`, `published/adapt_config.py`, `published/quantize_all.sh`, `published/README.md` | converter patch, config adapter (GGUF default 4 loops vs HF 2), quant recipes, model-card text |
| `hf/` | NVIDIA tokenizer.json, tokenizer_config.json, special_tokens_map.json, generation_config.json, config.json, chat_template.jinja (no weights) |
| `read_kld.py`, `score.py`, `kld_to_npz.py`, `compare_kld.py` | tested reference reader and scorer |
| `test-text.txt` (611 tokens), `text8.txt` (4888 tokens) | probe texts |
| `t256-*.kld` (13 files, 32 MB each) | the probe references behind the measurement table; the larger timing-probe files were deleted to save disk |
| `cand-lastloop2-b64.npz` | the npz used to prove `score.py` against llama.cpp |
| `logs/` | full output of every run quoted above |


---

# LEAD CORRECTION (2026-10-24 00:15) — the exact reference needs `-ub = n_ctx`

Section 4 above ("exact mode is batching-independent") holds only while the 1024-token sliding
window is inactive. **Measured by the DANE-3 implementer on the BF16 GGUF, per_loop, `-c 4096 --chunks 1`, loops 1:**

| `-ub` | PPL |
|---|---|
| 64 / 256 | 4.9790 (bit-identical to each other) |
| 512 | 4.9610 |
| 1024 | 4.9623 |
| 2048 | 4.9662 |
| 4096 | 4.9655 |

and a ref-vs-ref comparison at loops 2 between `-ub 1024` and `-ub 4096` gives **mean KLD 0.0076,
top-1 99.41 %, max 10.7** — the same order as every gate we are trying to measure. At ctx 512 and 1024
every ubatch setting is bit-identical; the dependence starts exactly when the sliding window begins
masking. llama.cpp computes SWA attention with batch-boundary effects, so a chunk split into ubatches
is *not* a full recompute.

**Ruling, binding for every ticket that reports KLD or top-1:**
- **exact mode** = `per_loop` + **one ubatch covering the whole chunk** (`-b N -ub N` with `-c N`) +
  `--chunks k` for k independent contexts. Only this is a full recompute.
- **decode mode** = `last_loop` with the candidate's own batching (`-b 1 -ub 1` for decode steps,
  `-b 64 -ub 64` for 64-token prefill) — that boundary is the point of decode mode.
- Any reference built with `-ub < n_ctx` must be regenerated before it is used as a gate.

The floor numbers above (last_loop 3.5e-3..4.7e-3 KLD / 97.5-98 % top-1 vs exact) were produced with
256-token chunks, below the window; they remain indicative but must be re-measured with the fixed
exact reference.
