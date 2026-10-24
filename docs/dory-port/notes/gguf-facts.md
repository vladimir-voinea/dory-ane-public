# GGUF facts — verified by the Lead from Dory1.0-4B-BF16.gguf metadata (2026-10-23 22:45)

Source: `~/.cache/huggingface/hub/models--cloudnathan5--Dory1.0-4B-GGUF/snapshots/9abcb0453c0382e6852fb57893102a7ed0fe17be/Dory1.0-4B-BF16.gguf` (GGUF v3, 507 tensors, 58 KV fields), dumped with `gguf-py` from the llama.cpp side build. Architecture `dory`.

## Hyperparameters (verbatim keys and values)
| key | value |
|---|---|
| dory.block_count | 72 |
| dory.context_length | 262144 |
| dory.embedding_length | 2560 |
| dory.feed_forward_length | 7168 |
| dory.attention.head_count | 16 |
| dory.attention.head_count_kv | 4 |
| dory.attention.key_length | 256 |
| dory.attention.value_length | 256 |
| dory.rope.dimension_count | 256 (full rotary, all 256 dims) |
| dory.attention.sliding_window | 1024 |
| dory.input_layer_count | 16 |
| dory.recurrent_layer_count | 40 |
| dory.output_layer_count | 16 |
| dory.num_loops | 4 (community GGUF default; NVIDIA HF default is 2) |
| dory.recurrent_kv_cache_mode | per_loop (GGUF default) |
| dory.init_std | 0.01769999973475933 |
| dory.alpha_init | 0.05000000074505806 |
| dory.gate_init | 50.59644317626953 |
| dory.rope.freq_base | 10000000.0 ← llama.cpp's `theta_1` |
| dory.rope.freq_base_2 | 10000.0 ← llama.cpp's `theta_2` |
| dory.rope.freq_base_swa | 10000.0 |
| tokenizer.ggml.model / pre | gpt2 / pixtral |
| tokenizer.ggml.bos_token_id / eos | 1 / 11 |
| add_bos / add_eos | False / False |
| general.sampling.temp / top_p | 1.0 / 0.95 |

## Block layout — decoded from `dory.layer_pattern`
```
S-S-S-*-S-S-S-*  (x9 groups, 72 chars total;  one char per physical block)
```
So, for a physical block index `i` (0..71), `k = i % 8`:
- `k in {0,2,4}` → **sliding-window attention** (window 1024, θ = 1e4)
- `k == 6` → **full attention** (θ = 1e7)
- `k in {1,3,5,7}` → **MLP block**

Counts: 36 attention (27 sliding + 9 full) + 36 MLP = 72. ✓ spec D3.
Ranges: input blocks 0–15 → full at **6, 14**; recurrent blocks 16–55 → full at **22, 30, 38, 46, 54**; output blocks 56–71 → full at **62, 70**. 9 full-attention caches, 27 sliding.

`dory.rope.profile` = `[2,1,2,1,2,1,1,1]` repeated 9x. In `dory.cpp`: `theta = profile[p] == 2 ? theta_2 : theta_1`. Since `theta_2 = 1e4` and `theta_1 = 1e7`, profile 2 → **1e4** (the sliding blocks) and profile 1 → **1e7** (the full blocks). The mapping is *inverted* relative to the key names — do not "simplify" it.

## Tensor inventory (507 = 3 + 36x6 + 36x8)
- 3 globals: `tok_embd`, `output` (lm_head, **untied**), `logit_scale` (per-vocab vector).
- Per MLP block (36), 6 tensors: `ffn_gate`, `ffn_up`, `ffn_down` + `ffn_gate_scale` (suv_gate), `ffn_up_scale` (suv_up), `residual_alpha`.
- Per attention block (36), 8 tensors: `attn_q`, `attn_k`, `attn_v`, `attn_out` + `attn_gate` (wqkv_gate), `attn_qk_scale` (sqk), `attn_gate_scale` (s_gate), `residual_alpha`.

Use this as the "every tensor accounted for" check for the HF→PyTorch loader: 36x6 + 36x8 + 3 = 507.

## The graph, as llama.cpp builds it (`llama.cpp-dory/src/models/dory.cpp`, 200 lines, read in full)
Virtual layers = 16 + 40*N + 16 (N = loop count, 1..4). Physical block `p = physical[il]`.

1. `input_h = cur` captured at `il == n_input` (i.e. the hidden state entering the recurrent stack).
2. At the first virtual layer of every loop (`(il - n_input) % n_recurrent == 0`): `cur = L2(cur + input_h)`.
3. **MLP block**: `g = gate_proj(cur) * (suv_gate * sqrt(n_embd))`; `u = up_proj(cur) * (suv_up * sqrt(n_embd))`; `cur = L2(silu(g) * u)`; `cur = down_proj(cur)`.
4. **Attention block**: q,k,v = projections; if profile≠0 and not MLP: NEOX RoPE with `theta` per the profile mapping (all 256 dims, `n_ctx_orig = n_ctx`, scale 1.0, no freq factors);
   `sq = (sqk * sqrt(head_dim) / init_std)^2` per (head,dim) → `q = L2_per_head_dim(q) * sq`; `k = L2_per_head_dim(k)`;
   `gate = sigmoid(gate_proj(cur) * (s_gate * sqrt(gate_init) / init_std)^2)`;
   `attn = softmax((q·k^T)/sqrt(head_dim)) @ v` over the cache for `cache_layer[il]`;
   `cur = L2(attn * gate)`; `cur = o_proj(cur)`.
   Note: the gate multiplies **before** the L2, and there is **no** per-head output normalization beyond that.
5. **nGPT residual**, for every block: `a = L2(residual)`, `b = L2(cur)`, `alpha = |residual_alpha * alpha_init / init_std|`, `cur = L2(a + (b - a) * alpha)`.
6. At the last virtual layer of every loop (`(il + 1 - n_input) % n_recurrent == 0`): `cur = L2(cur)` (extra loop-exit norm).
7. **Logits**: `logits = output @ cur`, then `logits *= logit_scale / init_std`.
8. `L2` everywhere is `build_gdn_l2_norm(x, 1e-6)` — L2 norm with eps 1e-6 (confirm the exact eps placement in llama-graph.cpp: it is the ANE-safe formulation's seam in D5).

## KV cache modes
- `per_loop`: `cache_layer[il] = il` (each virtual layer gets its own cache) — batching-independent, equals a full recompute. Use for the **exact** reference.
- `last_loop`: `cache_layer[il] = p` for recurrent blocks — one cache per *physical* recurrent attention block, so the N loop passes share it. Comment in the source: "Repeated writes replace only the current chunk; older tokens retain final-loop KV." Use for the **decode** reference and for the ANE port.
- Both are GGUF metadata (`dory.recurrent_kv_cache_mode`), overridable at run time. Studio's live server uses `--override-kv dory.num_loops=int:2`; the same mechanism takes `dory.recurrent_kv_cache_mode=str:last_loop`.

## Tokenizer / template
`tokenizer.chat_template` and the pixtral BPE are in the GGUF and in the HF repo's `tokenizer_config.json` (already in `scratch/hf-a/`). BOS 1, EOS 11, no automatic BOS.
