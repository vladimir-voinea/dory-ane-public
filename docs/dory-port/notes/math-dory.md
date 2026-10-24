# Dory1.0-4B — authoritative math + tensor notes for the ANE port

**Status:** complete for the questions in DANE-3 / DANE-4. Every claim below is either (a) cited to a line in the authoritative source, or (b) measured from a real Dory artifact. Nothing here is inferred from the spec. Where the spec disagrees, see §10.

**Authority:** `/Users/vladimir/.unsloth/llama.cpp-dory` @ `01ad20223` (merge of ggml-org/llama.cpp#30306, "dory").
Primary files, all read in full unless noted:

| file | what it decides |
|---|---|
| `src/models/dory.cpp` (200 lines) | hparam validation, physical↔virtual layer map, the whole graph |
| `src/models/models.h:14-18` | `build_gdn_l2_norm` — the L2 norm used everywhere |
| `src/llama-graph.cpp` | `build_lora_mm` (1685), `build_qkv` (1870), `build_attn_mha` (3025), iswa `build_attn` (3536), masks (534, 742) |
| `src/llama-arch.cpp` | GGUF tensor names (521-527), key names (265-275, 307), arch name `"dory"` (9) |
| `src/llama-model.cpp` | `llama_model_rope_type` = NEOX for DORY (3581), DORY KV-cache filter (3232-3237) |
| `src/llama-kv-cache-iswa.cpp:69-105` | sliding cache sizing, base/SWA split |
| `src/llama-hparams.h:527-560` | `is_masked_swa` (the window rule) |
| `conversion/dory.py` (114 lines) | HF→GGUF: hparam mapping, required-config gates, `.weight` suffix rule |
| `gguf-py/gguf/{constants,tensor_mapping}.py` | GGUF key/tensor names, HF name candidates |

**Artifacts inspected (no weights downloaded):**

* `cloudnathan5/Dory1.0-4B-Q8_0.gguf` — already in the HF cache. Full KV + all 507 tensor names/shapes/dtypes dumped; the F32 scale vectors were read and measured.
* `cloudnathan5/Dory1.0-4B-BF16.gguf` — first 128 KB fetched by HTTP `Range` to read its metadata block only (identical hparams).
* `nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` — `config.json`, `generation_config.json`, `special_tokens_map.json`, `tokenizer_config.json` downloaded; the **safetensors JSON header** (56 584 bytes) fetched by two `Range` requests to get the exact HF tensor names/shapes. No tensor payload was fetched (the file is 7 950 515 472 bytes; we read 56 592 bytes of it).
* Everything is in `scratch/hf-a/` (`config.json`, `tokenizer_config.json`, `generation_config.json`, `special_tokens_map.json`, `st_head.bin` = the safetensors header, `bf16_head.bin` = the BF16-GGUF metadata prefix, `api.json`).

---

## 1. The model at a glance (measured, not assumed)

| quantity | value | source |
|---|---|---|
| arch name | `dory` | GGUF `general.architecture` |
| blocks (physical) | 72 | `dory.block_count` |
| hidden | 2560 | `dory.embedding_length` |
| MLP width | 7168 | `dory.feed_forward_length` |
| q heads / kv heads | 16 / 4 | `dory.attention.head_count{,_kv}` |
| head_dim (k, v and rope dims — all the same) | 256 | `dory.attention.key_length`, `value_length`, `rope.dimension_count` |
| q width / kv width | 4096 / 1024 | 16·256 / 4·256 |
| vocab | 131072 | `token_embd.weight` shape |
| context | 262144 | `dory.context_length` |
| zones | input 16 / recurrent 40 / output 16 | `dory.{input,recurrent,output}_layer_count` |
| loops | 2 (HF config) · **4 in both published GGUFs** | `n_recurrent_loops` / `dory.num_loops` |
| KV mode | `per_loop` in both published GGUFs | `dory.recurrent_kv_cache_mode` |
| biases | none anywhere | `attention_bias:false`, `mlp_bias:false`; no `*_b` tensors in the safetensors |
| softcap / clamp / alibi / yarn | none | no GGUF keys; `f_clamp_kqv`=0, `attn_soft_cap`=false |
| final norm | **none** | no `output_norm` tensor; converter requires `ngpt_enable_final_norm=False` |
| embedding scale | **none** | `build_inp_embd(tok_embd)` with default `tok_scale=1.0`; no `embedding_scale` key |
| tied embeddings | **untied** | converter raises on `tie_word_embeddings`; separate `lm_head.weight` |

### 1.1 The three constants, and where they really come from

| GGUF key | value (exact f32) | where it comes from |
|---|---|---|
| `dory.init_std` | 0.01769999973475933 | `config.initializer_range = 0.0177` |
| `dory.alpha_init` | 0.05000000074505806 | `config.ngpt_alpha_init = 0.05` |
| `dory.gate_init` | 50.59644317626953 | **derived**: `config.attention_output_gate_logit_scale_init` is `null`, so `conversion/dory.py:107` uses `hidden_size ** 0.5` = √2560 |

llama.cpp validates each (`dory.cpp:34-37`) and stores them in `llama_model_dory::{init_std,alpha_init,gate_init}`. They are used **only** as divisors/multipliers inside the graph; they are not folded into the weights at conversion (the converter does not touch tensor values at all — `dory.py:109-114` is a rename plus a `.weight` suffix).

Derived factors worth precomputing:

```
Kq   = sqrt(head_dim)/init_std        = 16 / 0.0177      = 903.9548
Kg   = sqrt(gate_init)/init_std       = 7.11312 / 0.0177 = 401.8710    (sqrt(50.59644256) = 7.113119)
Ka   = alpha_init/init_std            = 0.05 / 0.0177    = 2.824859
Ksuv = sqrt(n_embd)                   = sqrt(2560)       = 50.59644256
Klm  = 1/init_std                                        = 56.497175
```

### 1.2 Measured magnitudes of the scale vectors (from the Q8_0 GGUF, which stores them F32)

These matter for FP16 range planning (§9):

| vector | measured | after its in-graph factor |
|---|---|---|
| `attn_qk_scale` (blk 0) | [-0.04395, +0.06177], absmean 0.0294 | `sq = (sqk·Kq)²` ∈ [~0, **3117.6**]; blk 6 (full attn) max 2443.8 |
| `attn_gate_scale` (blk 0) | absmax 0.3613, absmean 0.0307 | `sg = (s_gate·Kg)²` ∈ [0.75, **21085**] |
| `residual_alpha` (blk 0) | [-0.0078, +3.0469], absmean 0.180 | `|α| = |a|·Ka` ∈ [0.0024, **8.607**] (per-channel vector; blk 71 max 0.715 → 2.02) |
| `ffn_up_scale` (blk 1) | [0.322, 13.25], mean 1.56 | `·Ksuv` ∈ [16.3, **670.4**] |
| `ffn_gate_scale` (blk 1) | [0.220, 4.125], mean 1.99 | `·Ksuv` ∈ [11.1, 208.7] |
| `output_scale` (logit_scale) | [0.914, 3.516], mean 2.023 | `·Klm` ∈ [51.6, **198.6**] |
| `token_embd` rows | **unit L2 norm** (1.0001 in Q8_0) | no scaling applied by llama.cpp |
| `output` (lm_head) rows | **unit L2 norm** (1.0000) | — |

Two consequences: (i) every activation that enters a block is unit-norm, because every block output is L2-normalised and the embedding is already unit-norm; (ii) `|α|` can exceed 1, so the residual blend is an *extrapolation* for those channels — it must not be clamped.

---

## 2. The 72-block layout (derived from `config.layer_types`, cross-checked against `dory.layer_pattern`)

`config.json` `layer_types` is 72 entries; `conversion/dory.py:34` maps `sliding_attention→'S'`, `full_attention→'*'`, `mlp→'-'`, and this becomes `dory.layer_pattern`, which llama.cpp requires to be exactly `n_physical` long (`dory.cpp:32`). Both artifacts agree:

```
S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-S-S-S-*-
```

i.e. the 8-block period `[sliding, mlp, sliding, mlp, sliding, mlp, full, mlp]` repeated 9×.

**Counts (exact):** 27 sliding-attention, 9 full-attention, 36 MLP.
**Full-attention block indices: 6, 14, 22, 30, 38, 46, 54, 62, 70.**
Split by zone: input 6 sliding + 2 full + 8 MLP; recurrent 15 sliding + 5 full + 20 MLP; output 6 sliding + 2 full + 8 MLP.

The table (blk = physical index = HF `backbone.layers.N` = GGUF `blk.N`):
| blk | zone | kind | pat | RoPE | window |
|---:|---|---|---|---|---|
| 0 | input | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 1 | input | MLP | - | none | - |
| 2 | input | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 3 | input | MLP | - | none | - |
| 4 | input | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 5 | input | MLP | - | none | - |
| 6 | input | attention | * | NEOX 256d, theta=1e7 | full |
| 7 | input | MLP | - | none | - |
| 8 | input | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 9 | input | MLP | - | none | - |
| 10 | input | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 11 | input | MLP | - | none | - |
| 12 | input | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 13 | input | MLP | - | none | - |
| 14 | input | attention | * | NEOX 256d, theta=1e7 | full |
| 15 | input | MLP | - | none | - |
| 16 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 17 | recurrent | MLP | - | none | - |
| 18 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 19 | recurrent | MLP | - | none | - |
| 20 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 21 | recurrent | MLP | - | none | - |
| 22 | recurrent | attention | * | NEOX 256d, theta=1e7 | full |
| 23 | recurrent | MLP | - | none | - |
| 24 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 25 | recurrent | MLP | - | none | - |
| 26 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 27 | recurrent | MLP | - | none | - |
| 28 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 29 | recurrent | MLP | - | none | - |
| 30 | recurrent | attention | * | NEOX 256d, theta=1e7 | full |
| 31 | recurrent | MLP | - | none | - |
| 32 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 33 | recurrent | MLP | - | none | - |
| 34 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 35 | recurrent | MLP | - | none | - |
| 36 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 37 | recurrent | MLP | - | none | - |
| 38 | recurrent | attention | * | NEOX 256d, theta=1e7 | full |
| 39 | recurrent | MLP | - | none | - |
| 40 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 41 | recurrent | MLP | - | none | - |
| 42 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 43 | recurrent | MLP | - | none | - |
| 44 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 45 | recurrent | MLP | - | none | - |
| 46 | recurrent | attention | * | NEOX 256d, theta=1e7 | full |
| 47 | recurrent | MLP | - | none | - |
| 48 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 49 | recurrent | MLP | - | none | - |
| 50 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 51 | recurrent | MLP | - | none | - |
| 52 | recurrent | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 53 | recurrent | MLP | - | none | - |
| 54 | recurrent | attention | * | NEOX 256d, theta=1e7 | full |
| 55 | recurrent | MLP | - | none | - |
| 56 | output | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 57 | output | MLP | - | none | - |
| 58 | output | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 59 | output | MLP | - | none | - |
| 60 | output | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 61 | output | MLP | - | none | - |
| 62 | output | attention | * | NEOX 256d, theta=1e7 | full |
| 63 | output | MLP | - | none | - |
| 64 | output | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 65 | output | MLP | - | none | - |
| 66 | output | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 67 | output | MLP | - | none | - |
| 68 | output | attention | S | NEOX 256d, theta=1e4 | 1024 |
| 69 | output | MLP | - | none | - |
| 70 | output | attention | * | NEOX 256d, theta=1e7 | full |
| 71 | output | MLP | - | none | - |

Theta selection is data-driven, not positional: `dory.rope.profile[p]` is `2` for sliding and `1` for full/MLP (`conversion/dory.py:51`), and the graph picks `profile[p]==2 ? theta_2 : theta_1` (`dory.cpp:159`) where `theta_1 = dory.rope.freq_base = 1e7` and `theta_2 = dory.rope.freq_base_2 = 1e4`. The measured profile array is `[2,1,2,1,2,1,1,1, ...]` — note MLP entries also carry `1` (the converter rejects any value outside {1,2} for *every* layer), but the graph only consults it inside the attention branch.

### 2.1 Physical → virtual layer map (this is the loop)

`dory.cpp:56-67` builds `physical[]` and overrides `hparams.n_layer_all = n_input + n_recurrent*n_loops + n_output`, so the graph iterates over **virtual** layers:

```
il = 0                .. 15          -> physical 0  .. 15      (input zone, run once)
il = 16               .. 55          -> physical 16 .. 55      (recurrent zone, loop 1)
il = 56               .. 95          -> physical 16 .. 55      (loop 2)
il = 96               .. 135         -> physical 16 .. 55      (loop 3)
il = 136              .. 175         -> physical 16 .. 55      (loop 4)
il = 16+40N           .. 16+40N+15   -> physical 56 .. 71      (output zone, run once)
```

with `N = dory.num_loops`. Consequences:

* `n_virtual = 16 + 40N + 16`; for `il < 72`, `physical[il] == il` (the map is the identity over the first pass and over the output zone start) — which is why cache indices and SWA flags can be indexed by either number for `il < 72`.
* The tensors are loaded once per **physical** layer: `load_arch_tensors` loops `il < n_layer` (=192) and re-uses `layers[first[p]]` for every repeat (`dory.cpp:89-118`), so there is exactly one weight set per physical block.
* `n_loops` is validated `1..4` (`dory.cpp:30`). `LLAMA_MAX_LAYERS = 512` (`llama-hparams.h:11`) is not the binding limit; the explicit `n_loops > 4` check is.

---

## 3. Tensors: the complete HF → GGUF mapping

507 tensors = 3 top-level + 36 attention blocks × 8 + 36 MLP blocks × 6. Verified against the safetensors header (names + shapes) and the GGUF tensor list (names + shapes + dtypes) — they correspond one-to-one.

**There is no permutation, no transpose, no merge/split, no reordering, and no scaling of any weight.** `DoryModel.modify_tensors` (`conversion/dory.py:109-114`) overrides the base class and only renames; it does not call `super()`. (Contrast LLaMA, which permutes q/k for NORM rope — Dory needs no permute because it uses NEOX rope and the checkpoint is already in that layout.) Converting to GGUF writes the HF matrix unchanged; ggml then reads it as `ne = [in_features, out_features]`, i.e. the *reverse* of the stored shape, which is what `ggml_mul_mat(w, x)` expects.

| # | HF name | GGUF name | role | HF shape | GGUF shape | transform |
|---|---|---|---|---|---|---|
| 1 | `backbone.embeddings.weight` | `token_embd.weight` | token embedding | (131072, 2560) | (2560, 131072) | rename only |
| 2 | `lm_head.weight` | `output.weight` | **untied** LM head | (131072, 2560) | (2560, 131072) | rename only |
| 3 | `lm_head.logit_scale` | `output_scale.weight` | per-vocab logit scale | (131072,) | (131072,) | rename + `.weight` |
| 4 | `backbone.layers.N.mixer.q_proj.weight` | `blk.N.attn_q.weight` | W_q | (4096, 2560) | (2560, 4096) | rename only |
| 5 | `backbone.layers.N.mixer.k_proj.weight` | `blk.N.attn_k.weight` | W_k | (1024, 2560) | (2560, 1024) | rename only |
| 6 | `backbone.layers.N.mixer.v_proj.weight` | `blk.N.attn_v.weight` | W_v | (1024, 2560) | (2560, 1024) | rename only |
| 7 | `backbone.layers.N.mixer.o_proj.weight` | `blk.N.attn_output.weight` | W_o | (2560, 4096) | (4096, 2560) | rename only |
| 8 | `backbone.layers.N.mixer.g_proj.weight` | `blk.N.attn_gate.weight` | attention output gate | (4096, 2560) | (2560, 4096) | rename only |
| 9 | `backbone.layers.N.mixer.sqk` | `blk.N.attn_qk_scale.weight` | q per-(head,dim) scale | (4096,) | (4096,) | rename + `.weight` |
| 10 | `backbone.layers.N.mixer.s_gate` | `blk.N.attn_gate_scale.weight` | gate scale | (4096,) | (4096,) | rename + `.weight` |
| 11 | `backbone.layers.N.attn_alpha` | `blk.N.residual_alpha.weight` | nGPT residual α | (2560,) | (2560,) | rename + `.weight` |
| 12 | `backbone.layers.N.mixer.gate_proj.weight` | `blk.N.ffn_gate.weight` | W_gate | (7168, 2560) | (2560, 7168) | rename only |
| 13 | `backbone.layers.N.mixer.up_proj.weight` | `blk.N.ffn_up.weight` | W_up | (7168, 2560) | (2560, 7168) | rename only |
| 14 | `backbone.layers.N.mixer.down_proj.weight` | `blk.N.ffn_down.weight` | W_down | (2560, 7168) | (7168, 2560) | rename only |
| 15 | `backbone.layers.N.mixer.suv_gate` | `blk.N.ffn_gate_scale.weight` | gate branch scale | (7168,) | (7168,) | rename + `.weight` |
| 16 | `backbone.layers.N.mixer.suv_up` | `blk.N.ffn_up_scale.weight` | up branch scale | (7168,) | (7168,) | rename + `.weight` |
| 17 | `backbone.layers.N.mlp_alpha` | `blk.N.residual_alpha.weight` | nGPT residual α | (2560,) | (2560,) | rename + `.weight` |

Rows 4-11 exist for the 36 attention blocks; rows 12-17 for the 36 MLP blocks. Rows 11 and 17 share one GGUF name because a physical block is either attention or MLP, never both: `tensor_mapping.py:227` lists both HF names under `RESIDUAL_ALPHA`.

Notable details:

* **No norm tensors exist** — no `attn_norm`, no `ffn_norm`, no `output_norm`. All normalization is parameter-free (nGPT). A loader that expects a per-block norm pair will find nothing.
* The 6 scale/α tensors are 1-D and are *not* quantized: `tensor_force_quant` keeps them at the file's float type — `BF16` in a BF16 GGUF, `F16` in an F16 GGUF, and **`F32` otherwise** (`dory.py:11-24`), which is why the Q8_0 GGUF carries them as F32 (measured above; handy for us: the exact values survive quantization).
* `lm_head` is untied and neither matrix is transposed; `output.weight` is a full (2560, 131072) matrix — at LUT6 this is the single biggest chunk and must be split for ANE (spec D6 already says ≥2 parts).
* `config.json` `architectures: ["DoryForCausalLM"]` and `@ModelBase.register("DoryForCausalLM")` is the key the converter dispatches on. `model_type` is `dory`.
* The converter **refuses** a config unless `dory_format_version=1`, `ngpt`, `fngpt`, `ngpt_enable_final_norm=False`, `attention_output_gate=True`, `mlp_hidden_act=silu`, no biases, `initial_state=input_copy`, `input_transform=residual`, `output_transform=replace`, `partial_rotary_factor{,_2}=1.0`, `mixer_type=None`, equal SWA/global head geometry, `tie_word_embeddings=False` (`dory.py:37-89`). For the public config the first group is synthesised from `layer_types` / `rope_parameters` and then *checked* against the config where the key exists.

---

## 4. The exact computation, in order

Notation. `n` = number of tokens in this ubatch; all activations are `h ∈ R^2560` per token. `L2(x) = x / sqrt(sum(x²) + 1e-6)` over the vector's leading dimension (see §4.1). `⊙` is elementwise. `σ` is the logistic sigmoid. `silu(x) = x·σ(x)`.

### 4.0 Tokens → hidden state

```
h0 = token_embd[token]        # [2560] — NO sqrt(dim) scale, NO L2, NO bias
```

`build_inp_embd(model.tok_embd)` is called with the default `tok_scale=1.0` (`dory.cpp:126`) and there is no `embedding_scale` in the GGUF, so nothing is applied. The rows happen to be unit-norm (§1.2), which is what makes the input consistent with the rest of the stack — but **do not normalize explicitly**: llama.cpp does not, and the projection into block 0 sees the raw row.

### 4.1 The L2 norm — exact definition

`models.h:14-18`:

```
build_gdn_l2_norm(ctx, x, eps) = scale( rms_norm(x, eps/ne0(x)), 1/sqrt(ne0(x)) )
```

`ggml_rms_norm` computes `x / sqrt(mean(x²) + eps_in)`, so with `eps_in = 1e-6/dim` and the trailing `1/sqrt(dim)`:

```
L2(x) = x / sqrt( sum(x²) + 1e-6 )      # eps on the SUM of squares, not on the mean
```

Callers always pass `1e-6f` (`dory.cpp:132`), so **every** norm in the model uses ε = 1e-6 on the squared norm. This is exactly the "RMSNorm divided by √dim" formulation the spec asks for — but note the eps placement (§9, §10 G6).

### 4.2 An attention block (physical `p`, pattern `S` or `*`)

`dory.cpp:153-172`. Per token, with `h` the block input `[2560]`:

```
 1. q = W_q h        [4096]      W_q = blk.p.attn_q        (4096x2560)
    k = W_k h        [1024]      W_k = blk.p.attn_k        (1024x2560)
    v = W_v h        [1024]      W_v = blk.p.attn_v        (1024x2560)
 2. reshape: q -> 16 heads x 256, k,v -> 4 heads x 256   (head-major, dim contiguous)
 3. RoPE, NEOX (rotate-half) over ALL 256 dims, applied to q and k only, NOT to v:
        theta = 1e7 if pattern[p] == '*' else 1e4         (per layer, see the table in §2)
        freq_scale = 1.0, ext_factor = 0.0, attn_factor = 1.0   -> no YaRN, no scaling
 4. q <- L2(q) per head (over the 256 dims of that head)
    k <- L2(k) per head
 5. q <- q * sq        sq[h,d] = ( attn_qk_scale[p][h*256+d] * Kq )^2      (per head AND per dim)
 6. gate = sigmoid( (W_g h) * sg )     sg[h,d] = ( attn_gate_scale[p][h*256+d] * Kg )^2
 7. o = softmax( (q.k^T)*(1/sqrt(256)) + mask ) . v     (GQA: each of the 4 k/v heads serves 4 q heads)
        mask: causal; for sliding blocks additionally mask key p0 when (p1 - p0) >= 1024
        the q.k^T product and the softmax accumulate in F32
 8. o <- L2( o * gate )   — over all 4096 channels at once, NOT per head
 9. cur = W_o o       [2560]      W_o = blk.p.attn_output (2560x4096)
```

Order matters: RoPE comes **before** the L2, and the `sqk` scale comes **after** it (`dory.cpp:160-164`). q's L2 is taken on the rotated vector.

### 4.3 An MLP block (physical `p`, pattern `-`)

`dory.cpp:146-152`:

```
 1. g = W_gate h  [7168]      W_gate = blk.p.ffn_gate  (7168x2560)
    u = W_up   h  [7168]      W_up   = blk.p.ffn_up    (7168x2560)
 2. g <- g * ( ffn_gate_scale[p] * sqrt(2560) )    # note sqrt(n_embd), NOT sqrt(n_ff)
    u <- u * ( ffn_up_scale[p]   * sqrt(2560) )
 3. y = L2( silu(g) * u )      # L2 over the 7168 channels
 4. cur = W_down y [2560]      W_down = blk.p.ffn_down (2560x7168)
```

### 4.4 The nGPT residual update — after **every** block

`dory.cpp:145, 174-177`:

```
residual = h                      # the block input, captured before the block
f        = cur                    # the block output from 4.2 or 4.3
a        = L2(residual)
b        = L2(f)
alpha    = |residual_alpha[p]| * Ka          # per-channel vector [2560], Ka = 0.05/0.0177
h        = L2( a + alpha * (b - a) )         # third norm, on the blended vector
```

Algebraically `a + α(b-a) = (1-α)·a + α·b` with `a`, `b` unit vectors, then renormalised. `|α|` is applied per channel and can exceed 1 (`|α|` up to 8.6 measured in block 0), so this is a slerp-style extrapolation, not a convex mix. The outer L2 is **not** redundant: without it the update would change the norm and every downstream block would see a mis-scaled input.

So **3 L2 norms per block** (a, b, blended), 36 MLP and 36 attention blocks per pass.

### 4.5 Loop entry and exit

`dory.cpp:139-144` and `178-180`:

```
h_in = h                          # captured ONCE, at il == n_input: the state entering the
                                  # recurrent zone, i.e. the output of block 15 (already
                                  # L2-normalised by 4.4)

for loop L in 1..N:
    if start of the recurrent zone:
        h = L2( h + h_in )        # loop entry, before block 16
    run blocks 16..55 in order    # each does 4.2/4.3 + 4.4
    if end of the recurrent zone:
        h = L2( h )               # loop exit, extra norm, after block 55's residual
```

Two facts an implementer must not miss:

1. **At L = 1 the entry add is a no-op up to ε.** `h_in` is captured from the same `h` that the entry add immediately re-reads, so the sum is `2h` and L2 is scale-invariant: `L2(2h) = L2(h)`, differing only in how ε enters. llama.cpp still performs it, and so should we if we want bit-similar F32 agreement, but it cannot be a source of divergence.
2. The exit L2 runs after **every** pass, including the last one. It is not "the final norm" — there is no other final norm anywhere.

### 4.6 Output block and LM head

Blocks 56..71 run once, exactly as in 4.2/4.3/4.4. Then (`dory.cpp:183-190`):

```
h      = the state after block 71's residual update        [2560], unit norm
logits = ( W_lm . h ) * ( output_scale / init_std )        [131072]
         W_lm = output.weight (131072x2560), output_scale = lm_head.logit_scale
```

No bias, no final norm, no softmax inside the model. `out_ids` may gather a subset of rows before the head (`dory.cpp:183-185`) — a decoding optimisation, not part of the math.

### 4.7 One-line pseudocode for the whole forward

```
h = emb[token]
for p in 0..15:      h = residual_update(h, attn_or_mlp(p, h))     # 4.4
h_in = h
for L in 1..N:
    h = L2(h + h_in)
    for p in 16..55: h = residual_update(h, attn_or_mlp(p, h))
    h = L2(h)
for p in 56..71:     h = residual_update(h, attn_or_mlp(p, h))
logits = (W_lm @ h) * logit_scale / init_std
```

---

## 5. Constants: value, provenance, exact application site

| constant | value | GGUF key | applied at |
|---|---|---|---|
| `init_std` | 0.0177 (f32 0.01769999973475933) | `dory.init_std` | divisor in the q scale (`sqrt(hd)/init_std`), the gate scale (`sqrt(gate_init)/init_std`), the α factor (`alpha_init/init_std`), the logit scale (`1/init_std`) |
| `alpha_init` | 0.05 | `dory.alpha_init` | `alpha_init/init_std` = 2.8249 in 4.4 |
| `gate_init` | 50.59644317626953 = f32(√2560) | `dory.gate_init` | `sqrt(gate_init)/init_std` = 401.87, then **squared** (`dory.cpp:167`) |
| `eps_l2` | 1e-6 (hard-coded, not a GGUF key) | — | every L2, as `x/sqrt(sum x² + 1e-6)` |
| `theta_1` | 1e7 | `dory.rope.freq_base` | RoPE base for `profile==1` layers = the 9 full-attention blocks |
| `theta_2` | 1e4 | `dory.rope.freq_base_2` (also `rope.freq_base_swa`) | RoPE base for `profile==2` layers = the 27 sliding blocks |
| `n_swa` | 1024 | `dory.attention.sliding_window` | mask rule below |
| `n_loops` | 2 (config) / 4 (both GGUFs) | `dory.num_loops` | §2.1 |
| `kv_mode` | `per_loop` in both GGUFs | `dory.recurrent_kv_cache_mode` | §6 |
| `swa_type` | forced `STANDARD` | — (set in `dory.cpp:68`) | mask rule |
| `n_ctx` | 262144 | `dory.context_length` | RoPE context bound only (no scaling) |

### 5.1 The sliding-window rule, exactly

`llama-hparams.h:527-539` (`LLAMA_SWA_TYPE_STANDARD`):

```
mask key at position p0 for a query at position p1   <=>   (p1 - p0) >= n_swa
```

With `n_swa = 1024`: **a query at p1 attends to p0 ∈ [p1-1023, p1]** — the last 1024 positions *including itself*: 1024 keys at most, offset 1023 allowed, offset 1024 masked. (Equivalently `p0 > p1 - 1024`; be careful which side of the boundary you put the inequality on — off by one here shifts every sliding layer's result.) Future positions are masked by the causal rule first (`p0 > p1`, `llama-graph.cpp:558`), then the SWA rule.

A query near the start (p1 < 1023) simply sees fewer keys. There is no attention sink, no ALiBi, no softcap.

### 5.2 Sliding-cache sizing in llama.cpp (relevant to the ANE ring buffer)

`llama-kv-cache-iswa.cpp:69-105`:

```
size_base = n_ctx
size_swa  = GGML_PAD( min(size_base, n_swa + n_ubatch), 256 )
```

so the sliding cache is **not** 1024 rows: it is `n_swa + n_ubatch` rounded up to a multiple of 256 — 1280 rows for `n_ubatch=64`, 1536 for 512. The extra `n_ubatch` rows exist so the current batch can be written before the oldest rows are recycled. A 1024-row ring must therefore write-then-attend within the chunk, or keep 1024 + B rows.

---

## 6. Loop semantics and the KV caches

### 6.1 What the two modes do

`dory.cpp:70-79` decides, per virtual layer, which *cache slot index* the layer's K/V go to:

```
recurrent      = (physical[p] in the recurrent zone)
cache_layer[il] = (recurrent && mode == "last_loop") ? p : il
```

* **`per_loop`** — every virtual layer gets its own slot (`cache_layer[il] = il`). Attention layers per pass = 36, so with N loops there are `36 + 20(N-1)` attention caches (96 at N=4). Each loop's K/V lives in its own cache, so a loop pass always attends over K/V that *that same loop* produced for every earlier position. Result: **identical to a full recompute of the whole sequence**, independent of batching. This is the "exact" mode.
* **`last_loop`** — the 40 recurrent virtual copies all point at the same 20 physical recurrent slots (`cache_layer[il] = p`). The input and output zones are unaffected (`cache_layer[il] = il` there, and those `il` values are already < 72). So the number of caches is loop-count independent: 36 physical attention caches (27 sliding + 9 full), of which only the 20 recurrent ones are written more than once per step.

Under `last_loop`, within one graph, the same cache cells are written N times, once per loop pass, in layer order. Because `ggml_build_forward_expand(gf, cpy_k)` places each write before the attention that follows it (`llama-graph.cpp:3570-3609`) and the graph runs strictly in order, each attention reads:

* **rows of the current ubatch** — the K/V written by *this* loop pass a few nodes earlier (loop L's K/V), because the write for loop L replaced loop L-1's values in the same cells;
* **rows from earlier steps** — whatever was written into those cells the last time they were current, which is the **final loop's** K/V of that step.

That is exactly the spec's description, and `dory.cpp:169` says so in a comment: *"Repeated writes replace only the current chunk; older tokens retain final-loop KV."*

Consequences:

* the cache **shape does not depend on the loop count** (one cache per physical attention block), which is what makes a runtime loop count possible in the first place;
* but the *contents* of the older rows depend on the loop count and on where the ubatch boundaries fell. Changing N or the batch size invalidates the cache.

### 6.2 Memory (for the ANE plan)

Per full-attention block: `n_ctx × 1024 dims × 2 B` per tensor, ×2 for K and V → 48 MiB + 48 MiB = **96 MiB at 24576 rows**; ×9 blocks = **864 MiB**, matching spec D7.

Per sliding block: `rows × 1024 × 2 B × 2` → 1280 rows = 5.24 MiB.

| mode | full caches | sliding caches | KV total (FP16, 24576 ctx, ubatch 64) |
|---|---|---|---|
| `last_loop` | 9 × 96 MiB = 864 MiB | 27 × 5.24 = 141 MiB | ≈ 1.0 GiB |
| `per_loop`, N=2 | 14 × 96 = 1344 MiB | 42 × 5.24 = 220 MiB | ≈ 1.5 GiB |
| `per_loop`, N=4 | 24 × 96 = 2304 MiB | 72 × 5.24 = 377 MiB | ≈ 2.6 GiB |

### 6.3 What a host implementation must do, per decode step and per prefill batch

**Decode step for a token at position p (one row):**

1. `h = emb[token]`; run input blocks 0..15. Each attention block writes row p of *its own* cache (K after RoPE, V) and attends over rows `[max(0,p-1023), p]` for sliding / `[0, p]` for full.
2. `h_in = h`.
3. For L = 1..N:
   a. `h = L2(h + h_in)` (loop 1: no-op);
   b. for each recurrent attention block `p`: write row p of cache slot `p` (this *overwrites* the row written by loop L-1 in this same step), then attend over the whole cache — which now contains loop-L K/V for row p and final-loop K/V for rows < p;
   c. run the intervening MLP blocks;
   d. `h = L2(h)` at the end of the zone.
4. Run output blocks 56..71 (their caches are written once, at row p).
5. `logits = (W_lm h) ⊙ logit_scale / init_std`.

Nothing else needs saving: no per-loop hidden state, and no copy of `h_in` beyond the step. `h_in` is a per-token temporary.

**Prefill batch of B rows at positions p0..p0+B-1:** identical, except every cache write is B rows, and row i of the batch attends to rows `[max(0, p_i-1023), p_i]` that are (a) inside the batch → current-loop K/V, (b) older → final-loop K/V from the previous batch. So one chunk of prefill is *not* the same as B decode steps: rows near the chunk start share their neighbours' current-loop K/V.

**Batch-boundary caveat (important for the reference harness).** With `last_loop` the result depends on the *ubatch* boundaries, because the set of tokens that share "current-loop" K/V is exactly the ubatch. llama.cpp's default `n_ubatch` is 512. Therefore:

* the intended port (64-token prefill chunks) will **not** match a default llama.cpp run token-for-token in decode mode;
* the decode-mode reference must be generated with a matching `-ub 64` (llama-perplexity / llama-cli accept `--ubatch-size`), or the comparison must be made in `per_loop` mode;
* within a *single* ubatch that contains the whole sequence, `last_loop` ≡ `per_loop` ≡ full recompute.

**How big the boundary effect actually is** (measured by the reference-harness note, `notes/reference-scoring.md` §4, Q8_0, 2 loops, 256-token chunk, `llama-perplexity`): `last_loop` with one ubatch covering the chunk is **bit-identical** to `per_loop` (0 difference at every position), while `last_loop` with `-ub 64` differs from it by **KLD 3.46e-3 / top-1 97.64 %**. For scale, dropping flash attention costs KLD 3.2e-6. So the ubatch boundary is ~1000× more consequential than the attention kernel on this model — it is the dominant term in any decode-mode comparison and must be pinned deliberately, not inherited.

### 6.4 Choosing the loop count and KV mode for a reference run

`n_loops` and the KV mode are read **only** from GGUF metadata (`dory.cpp:9-10`); there is no `--loops` flag and no per-request knob. They can nevertheless be changed **per process, without touching the file**, through llama.cpp's metadata-override mechanism: `--override-kv KEY=TYPE:VALUE` (`common/arg.cpp:2965-2970`) fills `llama_model_params.kv_overrides`, and every `get_key` consults it (`llama-model-loader.cpp:431-436`). Any integral target — `n_loops` is `uint32` — is overridden with `int:` (`llama-model-loader.cpp:233-241`), string targets with `str:`:

```
--override-kv dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop
```

That is how a 2-loop / `last_loop` reference is produced (and how the owner's Studio runs it). Two caveats:

* the override must be spelled with the *right* type tag; a mismatch is only a warning ("Bad metadata override type for key …") and the key silently keeps its GGUF value, so **always check the loader's log line** `Using metadata override ( int ) 'dory.num_loops' = 2` before trusting a reference run;
* it is a model-load-time setting, so a *mid-session* loop-count change — as the spec wants for the port — is a model reload for the reference, which is another reason the port's loop count must be a host-side loop over one loaded model.

The persistent alternative (needed if a tool cannot pass overrides) is a metadata rewrite, which costs a full copy of the file: `gguf-set-metadata in.gguf dory.num_loops 2`. Budget 4.23 GB for Q8_0 or 7.95 GB for BF16; disk at the time of writing was 18 GB free.

Both published GGUFs (`Dory1.0-4B-Q8_0.gguf` and `Dory1.0-4B-BF16.gguf`) carry `num_loops = 4`, `recurrent_kv_cache_mode = per_loop` — *not* the 2-loop `last_loop` configuration the spec describes as the default, so every command that wants the spec's configuration must pass the override explicitly.

---

## 7. Tokenizer and chat template

**Tokenizer.** `tokenizer_class: PreTrainedTokenizerFast`, one `tokenizer.json`; byte-level BPE (`tokenizer.ggml.model = 'gpt2'`) with the **pixtral** pre-tokenizer (`tokenizer.ggml.pre = 'pixtral'`). 131072 tokens, 269443 merges, 1000 added special tokens (ids 0..999; 18+ are `<SPECIAL_nn>` placeholders). `add_bos_token = false`, `add_eos_token = false`, `add_prefix_space = false`, `clean_up_tokenization_spaces = false`, `model_max_length = 262144`.

Use `tokenizer.json` with HF `tokenizers` / `PreTrainedTokenizerFast` — do **not** re-implement the pre-tokenizer, and do not rely on llama.cpp's vocab dump for anything but parity checks.

**Special ids (verified in `tokenizer_config.json` `added_tokens_decoder` and in the GGUF vocab):**

| id | token | note |
|---|---|---|
| 0 | `<unk>` | `unknown_token_id` |
| 1 | `<s>` | BOS — **never added automatically** |
| 2 | `</s>` | listed in `generation_config.eos_token_id` |
| 11 | `<|im_end|>` | the turn terminator; `tokenizer.ggml.eos_token_id`, `pad_token`, `eos_token` |
| 10 | `<|im_start|>` | every message starts with it |
| 12 / 13 | ` thinking` / `</think>` | single tokens; `</think>` is the reasoning/answer boundary |
| 14 / 15 | `<tool_call>` / `</tool_call>` | tool call delimiters |
| 16 / 17 | `<tool_response>` / `</tool_response>` | inside a `user` turn |
| 3-9 | `[INST] [AVAILABLE_TOOLS] [TOOL_CALLS] ...` | legacy Mistral-style; unused by this template |

**EOS.** `generation_config.json` says `eos_token_id: [2, 11]`, `pad_token_id: 11`; the GGUF records only `eos_token_id = 11`. Stop on **11** primarily, and on **2** as well if you want to mirror the HF generation config.

**Chat template** (from `tokenizer.chat_template` in the GGUF; I rendered it with Jinja2 to get the exact strings; structure described, not pasted):

* Every message is `<|im_start|>{role}\n{content}<|im_end|>\n`. An **empty system block is always emitted** when the caller passes no system message — the rendered prompt still begins with `<|im_start|>system\n<|im_end|>\n`.
* Optional `tools`: if present the system block becomes `<|im_start|>system\n` + `# Tools` + a `<tools>…</tools>` XML block (per function: `<function><name>…</name><description>…</description><parameters>` with one `<parameter><name>` … per property, plus `<required>`) + a fixed instruction telling the model to answer only with `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>`.
* Assistant turns: if the caller supplies `reasoning_content` the content is rendered as `"\n" + reasoning + "</think>" + content`; if the content carries no thinking tags at all it is prefixed with `"\n"` (empty reasoning). `truncate_history_thinking` (default **true**) empties the reasoning of every assistant turn *before* the last user turn, to save context.
* Assistant `tool_calls` render as one `<tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>\n` per call, then `<|im_end|>`.
* Tool results are wrapped in a `user` turn: `<|im_start|>user\n<tool_response>\n{content}\n</tool_response>\n<|im_end|>\n`; consecutive tool messages share one user turn.
* `medium_effort=True` appends `\n\n{reasoning effort: efficient}` to the **last** user message.
* With `add_generation_prompt`:
  * `enable_thinking=True` (the default) → `<|im_start|>assistant\n thinking\n` — the model generates the reasoning immediately;
  * `enable_thinking=False` → `<|im_start|>assistant\n thinking</think>` — empty reasoning, the model answers directly.

So: reasoning ends at token **13** (`</think>`), the turn ends at **11**. A streamer can split on 13 and stop on 11 without any string matching.

---

## 8. Sanity anchors for the reimplementation

Put these in a test, in this order — each isolates one class of bug:

1. `L2(x)` on a unit vector returns the same vector (ε is negligible there) and on `c*x` returns the same vector for any `c>0`.
2. Embedding rows are unit norm → block 0's input is unit norm. If `|h|` after block 0 is not ≈1, the residual update is wrong.
3. After *any* block, `|h| ≈ 1`. This is a free invariant check at every layer.
4. `sq`, `sg`, `α`, `suv`, `logit_scale/init_std` should land in the ranges measured in §1.2. A factor-of-`sqrt(dim)` mistake shows up immediately here.
5. A single-token forward with N=1 must match N=4 **at the first token** (all loop passes read the same one-token cache, and the entry add is a no-op), so logits after one token are almost independent of N — a strong check that `h_in` and the entry/exit norms are wired correctly.
6. Attention logits: `(q̃·k̃)·sq` for unit vectors is in the hundreds (median `sq` ≈ 640, typical `|cos|` ≲ 0.1). If your logits are O(1), you applied `sqk` without squaring, or forgot `sqrt(hd)/init_std`.

---

## 9. Gotchas for a PyTorch reimplementation

**Scale and range (FP16).**

* `suv_up · sqrt(2560)` reaches **670** (measured max), so the MLP's `u` and `silu(g)⊙u` can reach a few hundred and `sum(x²)` over 7168 channels can reach `~1e8`. **FP16 overflows at 65504**, and the sum overflows once any element exceeds ≈3 (`7168·3² = 64512`). So the MLP's L2 must accumulate in F32 — or be computed as `sqrt(mean(x²))` and divided, which keeps the accumulation near 1. That is exactly the ANEMLL "RMSNorm / √dim" trick, and it is numerically *equivalent*, not an approximation, provided ε is handled as in §4.1.
* Same for the 4096-dim attention-output L2: bounded by `|o⊙gate| ≤ 1` so ≈4096 worst case — safe in FP16, but do not rely on it.
* `sq` reaches **3118**; multiplied by a unit-norm q component the product is fine in FP16, but `sq` itself must not be computed as `(sqk·Kq)²` in FP16 after folding `sqk` into q — the reference order is `L2(q)` **then** multiply, and folding `sq` into `W_q` is impossible (it follows a normalisation). Keep it an elementwise multiply.
* `logit_scale/init_std` reaches **199** — fine, but the folded `lm_head` rows then have norms 51..199, which matters for LUT quantisation (per-channel groups absorb it).
* `|α|` up to 8.6 and `a + α(b-a)` can have a large norm before the final L2 — compute the blend in F32. Nothing needs clamping: there is no clamp anywhere in the reference for this arch (`f_clamp_kqv = 0`, no softcap).

**Ordering traps.**

* RoPE → L2 → `sqk`. In that order. Rotating after normalising, or scaling before normalising, changes the result.
* The attention-output L2 is over **all 4096 channels**, i.e. after the heads are concatenated; it is *not* per head. (q's and k's L2s *are* per head, over 256 dims.)
* `gate` multiplies the attention output **before** the L2 and before `W_o`. `s_gate` is squared and laid out head-major, matching the 4096-wide attention output.
* The MLP scale uses `sqrt(n_embd) = sqrt(2560)`, **not** `sqrt(n_ff)`. A `sqrt(7168)` here is a silent 1.67× error.
* `alpha_init/init_std` (≈2.825) multiplies α *before* the absolute value; α is a **per-channel vector**, not a scalar, and the absolute value applies to the product (`ggml_abs(ggml_scale(...))`).
* Three norms per block. Implementers routinely drop the third (the one on the blended vector) because it "looks like normalising something already normalised" — it is not redundant: `(1-α)a + αb` is not unit norm unless `|α|` is 0 or 1.
* The loop-exit L2 fires after **every** pass including the last, and there is no other final norm.
* `h_in` is captured at the *first recurrent block*, i.e. it is the state after block 15 — not the embedding, and not the state after loop 1.

**Epistemic traps.**

* Do not "improve" the model by normalising the embedding or adding a final norm. Both are absent in the reference, and both move KLD far more than any quantisation choice.
* Do not fold the `init_std`-derived factors into the checkpoint and then compare against llama.cpp running the BF16 GGUF with the same factors still in the graph — the reference keeps them all in the graph.

---

## 10. Where the spec and llama.cpp disagree

Ranked by how much damage the spec's wording would do if taken literally.

**G1 — "loop count chosen at run time" is a load-time constant in the reference (material, but solvable with one flag).** Spec D4 wants the loop count chosen at run time, 1–4, with no reconversion — which the port can do, because the tensors are shared across loops. In llama.cpp it is GGUF metadata (`dory.num_loops`, `dory.recurrent_kv_cache_mode`) read at model load; the only way to vary it is `--override-kv dory.num_loops=int:2,dory.recurrent_kv_cache_mode=str:last_loop` (§6.4), i.e. per process, not per request, and a mid-session change is a model reload. Also note **both published GGUFs are 4 loops / `per_loop`** while the HF config default is 2: every reference run must state the override explicitly, and a run without it is silently the 4-loop `per_loop` model — the single easiest way to produce a "wrong reference" here.

**G2 — "one KV cache per physical recurrent attention block" is a third of the truth (material).** That holds only for the 20 recurrent attention blocks. The 16 input-zone and output-zone attention blocks also own caches (they are attention layers and they are the first occurrence of their physical index, so `cache_layer[il] == il`). The correct statement: **36 physical attention caches — 27 sliding + 9 full — of which the 20 recurrent ones are shared across loop passes.** An implementer who follows the spec literally allocates 20 and has nowhere to write the input/output zones' K/V.

**G3 — the sliding cache is not 1,024 rows (moderate).** llama.cpp sizes it `PAD(min(n_ctx, n_swa + n_ubatch), 256)` = 1280 rows for a 64-token batch. A ring of exactly 1024 rows cannot hold "current window + current batch" and still let the batch attend to a full window; the implementer needs 1024 + B rows, or must interleave write and read.

**G4 — BOS/EOS are wrong or incomplete (moderate).** Spec D3 says "BOS 1, EOS 11". Reality: `add_bos_token = false` and the template opens with `<|im_start|>`, so **no BOS is ever prepended** (prepending id 1 would be real prompt corruption, not a no-op); and `generation_config.json` lists **two** EOS ids, 2 (`</s>`) and 11 (`<|im_end|>`), while the GGUF records only 11. Stop on 11, optionally also on 2.

**G5 — the constants are not all "from the GGUF" (minor but load-bearing).** `gate_init` is not in NVIDIA's config at all: `attention_output_gate_logit_scale_init` is `null`, so the converter synthesises `hidden_size ** 0.5` = 50.59644256…, and the GGUF carries its f32 rounding. If a future NVIDIA revision publishes a non-null value, `sqrt(gate_init)` changes and every attention gate changes with it. Treat `dory.gate_init` as the authority, never the literal 50.596.

**G6 — ε placement (minor, but it compounds over ~400 norms per pass).** The spec says "L2 uses ε = 1e-6". In llama.cpp the helper divides ε by the dimension before `rms_norm`, so the effective term is `+1e-6` on the **sum of squares**. A stock `RMSNorm(eps=1e-6)` adds `n·1e-6` (2.56e-3 for the MLP), a ~0.13 % shrink per norm applied ~120 times per loop pass. Budget it in the KLD test rather than discovering it later.

**Confirmed as correct in the spec** (no action): the 16/40/16 zones and 27/9/36 counts; the full-attention indices following from the stated pattern; 2560/7168/16/4/256/131072; NEOX rope over all 256 dims with θ 1e4 / 1e7; the per-(head,dim) `(sqk·√256/init_std)²` scale; the `1/√256` softmax scale; `suv·√2560` on both MLP branches; the gate formula; the nGPT residual with `|α|`; `logit_scale/init_std`; the inclusive-1024 window; the untied head; the `last_loop` description ("older rows keep final-loop KV, current rows see the current loop"); and the exact HF tensor names in D2.

---

## 11. Blocking unknowns / what only a comparison against llama.cpp can settle

1. **Which reference configuration the Lead will actually score against.** The configuration is now expressible (`--override-kv …=int:2,…=str:last_loop`, §6.4), so this is a decision rather than a blocker: exact mode should be `per_loop` (batch-independent, = full recompute); decode mode should be `last_loop` at the port's chosen loop count **and** the port's ubatch size. What is still unknown is whether the Lead's already-built `llama-perplexity` invocation passes the override — if the harness was written against the GGUF defaults it is scoring 4 loops `per_loop`.
2. **The ubatch size of that reference.** `last_loop` is batch-dependent (§6.3); the port's 64-token prefill must be matched by `-ub 64` on the reference side, or the decode-mode tolerance must be re-derived.
3. **Whether NVIDIA's own forward agrees with llama.cpp's `h_in` placement and loop-exit norm.** No PyTorch reference exists in the wild (transformers has no `dory` class, no remote code), so llama.cpp *is* the definition. I did not run it (no BF16 weights on disk; Q8_0 would answer the question but the Lead owns that harness) — everything above is source + metadata + measured scale vectors.
4. **ε sensitivity** (G6) — worth one A/B: run the exact reference once with the true "1e-6 on the sum" and once with a stock RMSNorm ε, and measure the logprob delta. If it is below the gate, ignore G6; if not, match it exactly.
5. **Nothing else.** The graph is small and fully read; there is no code path in `dory.cpp` I have not accounted for. The only remaining statements are the `ggml_prec_set_src` loop at `dory.cpp:193-199`, which only fires for **F32** weights (it forces the activation operand of such matmuls to stay F32) and is therefore a no-op for the BF16 and Q8_0 GGUFs we care about.

---

## 12. How to re-verify any claim here

```
# the whole graph + hparams
sed -n '1,200p'   <repo>/src/models/dory.cpp
sed -n '14,18p'   <repo>/src/models/models.h                       # the L2 norm
sed -n '1685,1700p;1856,1982p;3025,3096p;3455,3625p' <repo>/src/llama-graph.cpp
sed -n '527,560p' <repo>/src/llama-hparams.h                       # the SWA rule
sed -n '69,106p'  <repo>/src/llama-kv-cache-iswa.cpp               # sliding cache size
sed -n '3230,3240p;3497,3600p' <repo>/src/llama-model.cpp          # Dory cache filter + NEOX
sed -n '1,120p'   <repo>/conversion/dory.py

# metadata of any Dory GGUF (arch, constants, pattern, profile, loop count, KV mode)
<repo>/.venv/bin/python -c "from gguf import GGUFReader; r=GGUFReader('<file>.gguf'); [print(k, r.fields[k].contents()) for k in r.fields if k.startswith('dory.')]"

# the exact HF tensor names/shapes without downloading weights (HTTP Range on the safetensors header)
curl -sSL -r 0-7 <url>/model.safetensors | od -An -tu8          # -> header length in bytes
curl -sSL -r 0-$((HEADER_LEN+7)) <url>/model.safetensors        # -> the header JSON only
```

Scratch artifacts used for the measurements above (all inside this repo; nothing outside it was modified):
`scratch/hf-a/{config.json,tokenizer_config.json,generation_config.json,special_tokens_map.json,st_head.bin,bf16_head.bin,api.json}`,
`scratch/gen_table.py` / `scratch/gen_table2.py` (regenerate §2's table from `config.json`).
