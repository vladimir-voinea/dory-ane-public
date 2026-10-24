# The ANE-safe Dory form: caches, chunk contract and gate evidence (DANE-4)

Ticket: DANE-4 (blocked by DANE-3, feeds DANE-5). Spec: D4 (`last_loop`),
D5 (ANE-safe formulation), D7 (caches).  Math authority:
`~/.unsloth/llama.cpp-dory` @ 01ad20223, `src/models/dory.cpp`; see
`notes/math-dory.md`.

Code:

| file | what it is |
|---|---|
| `anemll/models/dory_ane_model.py` | the cached FP16 ANE-safe module (PyTorch, the shape DANE-5 traces) |
| `anemll/utils/dory_caches.py` | cache inventory, ring-slot writes, host-side masks |
| `tests/test_dory_ane_model.py` | cache mechanics, GQA, folding, per-block oracle, loop/decode semantics, cache reset |
| `scripts/dory_ane_gate_suite.py` | all KLD gates in one process (weights stay hot) |
| `scripts/dory_ane_score.py` | one gate against one reference artifact |
| `scripts/dory_ane_folding_check.py` | the folding gate (folded vs unfolded FP32 on real weights) |
| `scripts/dory_ane_fp16_preflight.py` | ANEMLL's FP16 preflight criteria for an arch transformers cannot load |

## 1. The formulation (spec D5)

* **L2** is ANEMLL's layer-norm RMSNorm divided by sqrt(dim):
  `F.layer_norm([x, -x], eps=1e-6/dim) * 1/sqrt(dim)`, which is exactly
  `x / sqrt(sum(x^2) + 1e-6)` — llama.cpp's `build_gdn_l2_norm`
  (`models.h:14-18`, math-dory.md 4.1/G6).  No raw FP16 sum of squares: the
  layer-norm kernel accumulates in F32, which is what keeps the 7,168-channel
  MLP norm safe (G7).  Measured pre-norm `silu(g)*u` maximum over eval text:
  311.5, against FP16_MAX 65504.
* **Linears are 1x1 convolutions** in ANEMLL's (N, C, 1, W) layout
  (`DoryANEConv1x1`), weights stored as (out, in, 1, 1).
  `compute_mode="matmul"` is an *evaluation-only* path: a 1x1 conv is a matmul,
  and PyTorch's CPU FP16 conv kernel is ~120x slower than its GEMM kernel
  (1.7 vs 208 GFLOPS measured).  Both accumulate in FP32; the traced/ANEMLL
  form must use `compute_mode="conv"`.
* **Static scales folded in FP32, exactly** (`fold_scales=True`): 
  `suv_gate*sqrt(2560)` and `suv_up*sqrt(2560)` into the MLP gate/up rows,
  `(s_gate*sqrt(gate_init)/init_std)^2` into the attention gate rows,
  `logit_scale/init_std` into the LM-head rows.  `sqk` and `|alpha|` stay
  elementwise (they follow a normalisation).
* **Two RoPE tables**, NEOX rotate-half over all 256 dims: theta 1e4 for the 27
  sliding blocks, 1e7 for the 9 full ones, precomputed to `context_length+1`
  rows and sliced with `narrow(current_pos, q_len)` (positions are always
  contiguous, so no gather).
* **Grouped-query attention by reshaping the queries per KV head**:
  `(B,H,Q,D) -> (B,KV,rep*Q,D)`, batched matmul against `(1,KV,K,D)`, logits
  viewed back as `(B,H,Q,K)` so the mask broadcasts, then reshaped for the
  value matmul.  No repeat of K or V, and the cache is only read.

## 2. Caches (spec D4 / D7)

**36 physical attention caches** — 27 sliding + 9 full (math-dory.md G2), not
20: the input and output zones own theirs too.  Under `last_loop` only the 20
recurrent-zone ones are written once per loop pass.  Two state buffers, K and V
interleaved along dim 0 (ANEMLL's convention):

| buffer | shape at ctx 24,576 | bytes FP16 |
|---|---|---|
| `kv_cache_full` | (18, 4, 24,576, 256) | 864 MiB |
| `kv_cache_sliding` | (54, 4, 1,280, 256) | 141 MiB |

`sliding_rows = PAD(min(ctx, sliding_window + batch_size), 256)` = 1280 at
ubatch 64 — the same sizing llama.cpp uses (G3).  A ring of exactly 1024 rows
could not hold "window + current batch".

**Writes are single-row (decode) or n-row slice (prefill) updates** of the
buffer: `cache[slot:slot+1, :, cache_pos:cache_pos+seq, :] = k`.  The sliding
window advances by **index arithmetic** (`cache_pos = current_pos % rows`),
never by shifting or rewriting the cache: attention is a softmax over a *set* of
keys, so the ring's physical order does not matter.  This is deliberate — a
shift-rotate ring (Gemma 3's scheme) rewrites 27 x 2 MB per token, which spec D5
forbids.  A prefill batch never straddles the ring seam because the write offset
is a multiple of `batch_size` and `rows` is a multiple of 256.

**Loop semantics need no loop argument.**  Overwriting the current rows on each
pass *is* `last_loop`: older rows keep the final loop's K/V, the current rows
show the current loop's.  The host calls the recurrent chunk N times with the
same `current_pos`/`cache_pos`.  The loop-entry `L2(h + h_in)` and the loop-exit
`L2(h)` live inside that chunk (static `loop_entry`/`loop_exit` flags).

**A loop-count change resets both buffers** (`set_loops`, tested).

## 3. Chunk I/O (what DANE-5 traces)

| input | shape | dtype |
|---|---|---|
| `hidden_states` | (1, Q, 2560) | float16 |
| `causal_mask` | (1, 1, Q, context_length + sliding_rows) | float16 |
| `current_pos` | (1,) absolute position | int32 |
| `cache_pos` | (1,) `current_pos % sliding_rows` | int32 |
| `h_in` | (1, Q, 2560), recurrent chunk only | float16 |
| out `output_hidden_states` | (1, Q, 2560) | float16 |

The mask is one tensor with two segments: columns `[0:context_length)` are the
full-attention causal mask, `[context_length:]` the sliding mask **in ring-slot
order**.  Both are built on the host by `dory_caches.build_causal_mask`; the
model slices them per layer type.  Slots that are unwritten or older than the
window are `-inf`.

State-name note from DANE-5: coremltools 8.3 rewrites `model.kv_cache_full` to
`model_kv_cache_full` (dots are not allowed), so every component must declare
the **same** two state names for the union rule to hold.

## 4. Acceptance criteria — evidence

**Device line for every gate number below:** "ANE-safe FP16 form, evaluated on
MPS (Metal), `lm_head_parts=16`" — not ANE and not CPU.  The gate measures the
*formulation* (fp16 weights, L2 formulation, ring caches, folding, split head),
not Metal: the same module run on CPU (fp16 `matmul` kernels, `lm_head_parts=8`)
and on MPS (fp16 1x1-convolution kernels, `lm_head_parts=16`) over the same 128
tokens agrees to **mean KLD 2.8e-4 / p90 7.0e-4 / max 4.9e-3** with **100 % top-1
agreement** — 36x below the 1e-2 gate threshold — and both devices predict the
next token identically (59.06 % top-1 on 127 positions of the eval text, 0 %
same-token, i.e. no positional shift).  Only the kernel backend differs between
the two runs; every number in this table would be reported the same way if the
backend were the ANE.

All numeric evidence below is from NVIDIA's real BF16 safetensors
(`backbone.*`, 507 tensors, 3,975,229,440 parameters).  Correctness numbers do
not depend on machine load; timings do, so each carries the load average that
produced it.

| criterion (ticket) | status | evidence |
|---|---|---|
| folding reproduces the unfolded FP32 outputs within FP32 rounding | **PASS** | blocks 0,1,6,7,16,17,22,55 (sliding+full attention, MLP, all zones) max_rel **3.7e-6**; lm_head rows 0..8191 max_rel **2.2e-7** (`folding.json`) |
| FP16 cached decode at 2 loops: mean KLD <= 0.01, top-1 >= 98 % vs the one-token decode reference | **PASS** | rows **2048..2303**, mean KLD **-0.00002**, top-1 **99.61 %**, max KLD 0.001, 895 s at loadavg 297 |
| FP16 64-token prefill at 2 loops: same thresholds vs the 64-token reference, FULL row set | **PASS** | rows **2048..4094** mean KLD **-0.00002**, top-1 **99.76 %**; rows **6144..8190** mean KLD **-0.00002**, top-1 **99.56 %**; max KLD 0.001 both, finite=True, 42.5 s and 38.8 s at loadavg ~187 |
| loops 1 and 4 pass the same checks | **PASS** | decode 1 loop: rows 2048..2303 KLD **-0.00004**, top-1 99.61 % (522 s, loadavg 44); decode 4 loops: same rows KLD **-0.00005**, top-1 99.61 % (942 s, loadavg 8); prefill 1 loop: KLD **-0.00006**, top-1 99.61 % (9.9 s); prefill 4 loops: KLD **-0.00003**, top-1 **98.44 %** (30.1 s) |
| no inf or NaN anywhere over the evaluation set | **PASS** | FP16 preflight over 48 eval tokens: NaN 0, Inf 0, > FP16_MAX 0, max activation 311.5; every gate row additionally records `finite=True` (all seven runs) |
| ANEMLL's FP16 preflight passes | **PASS** (adapted runner) | params 3,975,229,440; max \|weight\| 13.25; 0 tensors exceed FP16 range; precision run 0 NaN / 0 Inf / 0 overflow; verdict PASS.  `fp16_preflight.sh` itself cannot load Dory (no transformers class, no remote code), so `scripts/dory_ane_fp16_preflight.py` applies its exact criteria |
| changing the loop count resets the caches, with a test | **PASS** | `test_changing_the_loop_count_resets_the_caches`: buffers zeroed on a change, not on a repeat, and a run after the change is bit-identical to a fresh model |
| neither the cache writes nor GQA rewrite or repeat the whole cache | **PASS** | `test_decode_write_touches_only_one_row`, `test_prefill_write_is_one_contiguous_slice`, `test_attention_never_materialises_a_repeat_of_kv` (largest tensor created is B\*H\*Q\*K, never H\*K\*D), `test_gqa_attention_matches_a_materialised_repeat` |
| every checkpoint tensor accounted for | **PASS** | 507 consumed, 0 missing, 0 duplicated |
| l2/unit-norm invariant | **PASS** | after the input chunk and after each loop pass, every row's norm is exactly 1.0 |

Resource facts measured on this Mac (shared, loadavg 25-800 during the day):
FP16 weights 7.95 GB; caches 292 MiB at ctx 4096 (1.0 GiB at 24,576); peak
process RSS **7.6 GB**; 28-40 s to load 507 tensors from the safetensors.

### 4.1 The two bugs the gates found (both worth knowing downstream)

**Full-attention caches are addressed by absolute position.**  The write used
the sliding *ring* offset for both cache kinds.  They coincide until
`context_length > sliding_rows`, so every short-context test and every tiny
fixture passed; at position 2048 the ring offset is 768, so the 9 full-attention
layers wrote the token into row 768 and then attended a cache of stale rows.
The 4,094-row prefill gate read **2.05 mean KLD / 44.6 % top-1** before the fix
and **-0.00002 / 99.76 %** after.  Fixed by `write_start = current_pos if
self.is_full else cache_pos`; guarded by
`test_full_cache_is_addressed_by_absolute_position`, which compares two ring
geometries and fails if the bug is reintroduced.  *If a future change makes the
ring and the absolute position diverge anywhere else, this is the test that
catches it.*

**`reset_caches()` must not clear the loop selection.**  It set
`_active_loops = None`, and the fallback was a literal `2`.  Every gate calls
`reset_caches()` before it runs, so the 1-loop and 4-loop gates silently ran 2
loops while being scored against 1-loop and 4-loop references — they read 0.028
and 0.0028 KLD and looked like a model defect.  Fixed in `8de924f`; the default
is now `config.num_loops` and
`test_reset_caches_keeps_the_loop_selection` pins it.  *Lesson for the Core ML
side: a loop count that is carried in mutable state must be asserted, not
assumed, at the moment a measurement is taken.*

## 5. Running the remaining gates

One process, model loaded once, cheapest first.  `--window-rows 256` scores the
contiguous window at absolute positions 2048..2303 of the first context;
`:full` scores all 4,094 rows of both contexts.

~~~bash
cd /Users/vladimir/dev/dory-ane-worktrees/DANE-4
export PYTHONPATH=$PWD
PY=/Users/vladimir/dev/Anemll/env-anemll/bin/python
A=~/dev/dory-ane-artifacts/DANE-4

# everything, in the approved order
$PY scripts/dory_ane_gate_suite.py --threads 6 \
    --gates prefill2:full,decode2:window,decode1:window,decode4:window,prefill1:window,prefill4:window \
    --out-json $A/gate_suite.json | tee $A/gate_suite.log
~~~

Each gate reads its own reference and is skipped with a message if the artifact
is missing:

| gate | reference | rows |
|---|---|---|
| prefill2 | `bf16_decode_loop2_batch64.kld` | full 4094 (approved core gate) |
| decode2 | `bf16_decode_loop2_batch1.kld` | 2048..2303 (approved window) |
| decode1 / decode4 | `bf16_decode_loop{1,4}_batch1.kld` | 2048..2303 |
| prefill1 / prefill4 | `bf16_decode_loop{1,4}_batch64.kld` | 2048..2303 |

Single-gate form: `scripts/dory_ane_score.py --reference <ref> --mode decode --rows 256`.

Regenerating the non-KLD evidence (no model needed for the folding gate beyond
2 x ~1.7 GB of FP32 subset weights):

~~~bash
$PY scripts/dory_ane_folding_check.py --out $A/folding.json
$PY scripts/dory_ane_fp16_preflight.py --tokens 48 --context-length 128 --out $A/fp16_preflight.json
$PY -m pytest tests/test_dory_ane_model.py -q     # 34 tests, no weights needed
~~~

## 6. What DANE-5 must know before tracing

1. Build the model with `compute_mode="conv"` (the matmul mode is only the fast
   eager-evaluation path).  DANE-5 has already traced a 16-block chunk with both
   state buffers, the ring-slice write, `ct.convert(... states=...)` at iOS18 and
   a coremlcompiler pass.
2. Trace the three chunks separately: blocks 0-15, 16-55
   (`loop_entry=True, loop_exit=True`), 56-71.  The loop count stays host-side.
3. Every component must carry the union of both state buffers, with identical
   names (coremltools may rewrite the dots).
4. `causal_mask` is `context_length + sliding_rows` wide and host-built; the
   runtime must build it with `dory_caches.build_causal_mask(start, q_len, ...)`
   and must send `cache_pos = current_pos % sliding_rows`.
5. meta.yaml needs keys 0.3.5 does not have: `kv_mode: last_loop`,
   `sliding_cache_rows`, `context_length`, `loop_range {min,max,default}` and a
   `components:` mapping (name -> file -> functions -> calls per token), because
   the port has five components and 4+N calls per token instead of the
   one-chunk-per-file layout ANEMLL infers from filenames.