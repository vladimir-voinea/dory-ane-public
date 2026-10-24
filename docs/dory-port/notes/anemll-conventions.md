# ANEMLL 0.3.5 conventions — the guide a Dory implementer must follow

**Pin.** Everything below is verified against the released ANEMLL **0.3.5_beta**
(`f4ad26d`, branch `integration/dory-ane`) at
`/Users/vladimir/dev/dory-ane-worktrees/integration`. Citations are `path:line`
relative to that checkout. Nothing here is from memory of another version; where a
claim could not be checked in 0.3.5 the text says so explicitly.

**How to use this.** Sections 1-7 are the conventions to copy. Section 8 is the list
of things in 0.3.5 that actively fight a looped-recurrent model, with the cleanest
ANEMLL-shaped way to express each. Anything marked **INVENT** does not exist in
0.3.5 and must be designed by the implementer; everything else has a working
precedent in-tree.

---

## 0. The shape of 0.3.5 in one page

Four architectures ship as **`anemll/models/<arch>_model.py`** (PyTorch reference +
CoreML-traceable module) and **`anemll/ane_converter/<arch>_converter.py`**
(coremltools conversion, CLI, file naming, metadata). Gemma 3 is the newest and is
the model to imitate for anything sliding-window or two-RoPE
(`anemll/models/gemma3_model.py`, `anemll/ane_converter/gemma3_converter.py`).

Three artifact shapes exist:

| Artifact | Functions inside | Produced by | Runtime handle |
|---|---|---|---|
| per-part model | one (`main`) | converter `--part 1` / `3` | `ct.models.MLModel` / `CompiledMLModel` |
| combined chunk | 2: `infer`, `prefill` | `combine_models.py --chunk N` (`combine_models.py:170-228`) | `.predict(..., function_name=...)` |
| Gemma 3 chunk | 4: `infer`, `infer_rotate`, `prefill`, `prefill_rotate` | `combine_models.py --gemma3` (`combine_models.py:346-453`) | same, function chosen by position |

Multifunction files are **never** produced by `ct.convert()`; they are assembled
afterwards with `ct.utils.MultiFunctionDescriptor` plus
`ct.utils.save_multifunction` (`combine_models.py:23-52`). There is no
`FunctionDescriptor` / `functions=` usage anywhere in `anemll/` — confirmed by grep.

The delivered model directory is: `<prefix>_embeddings[_lutN].mlmodelc`,
`<prefix>_lm_head[_lutN].mlmodelc`,
`<prefix>_FFN_PF[_lutN]_chunk_XXofYY.mlmodelc`, `meta.yaml`, `config.json`,
tokenizer files (`docs/model_conversion_guide_for_AI_agents.md:256-268`).

---

## 1. Adding a new architecture, end to end

### 1.1 Files you add, files you touch

Add:

- `anemll/models/dory_model.py` — config class, building blocks, `DoryModel`,
  `DoryForCausalLM`, Dory norms, RoPE modules.
- `anemll/ane_converter/dory_converter.py` — `DoryConverter(BaseConverter)`, part
  conversion wrappers, `parse_lut_arg`, `parse_args`, `test_conversion`, `main()`.
- `tests/test_dory_model.py` — pytest / cosine-vs-HF check in the style of
  `tests/test_qwen_inference.py`, plus a `tests/conv/test_hf_model.sh`-style
  end-to-end script.

Touch (all small, all mechanical):

- `anemll/utils/convert_model.sh:305-340` — the `jq` architecture dispatch chain
  (`model_type` decides which `python3 -m ...converter` runs). Today it maps
  `qwen2*` to `qwen2_5_converter`, `qwen*` to `qwen_converter`, `gemma3*` to
  `gemma3_converter`, else `llama_converter`. **This shell chain is the only
  "discovery" mechanism** — there is no registry, no entry point, no plugin.
- `anemll/utils/convert_model.sh:342-362` — sliding-window detection
  (`.use_sliding_window` / `.sliding_window` in `config.json`). Dory needs
  `sliding_window: 1024` recognised; the Gemma 3 branch shows the shape.
- `anemll/utils/convert_model.sh:366-393` — `SPLIT_LM_HEAD` default (8, or 16 for
  qwen/gemma) and the derived `lm_head_chunk_sizes` array. Dory's 131,072 vocab
  wants 16.
- `anemll/utils/generate_meta_yaml.py:157-161` and `:443-446` — the same
  `split_lm_head` mapping for the meta file.
- `anemll/utils/combine_models.py` — only if the function set differs from
  `{infer,prefill}` / the Gemma-3 four.
- `tests/chat.py` — only if a component takes an input the runtime does not already
  send (`hidden_states`, `position_ids`, `causal_mask`, `current_pos`, optional
  `update_mask`).

### 1.2 The two base classes, and what they actually require

**`anemll/models/base_model.py:9-72`** — `BaseModel(nn.Module)`:

~~~python
BaseModel.__init__(self, config)          # :12  stores self.config
@abstractmethod load_pretrained_weights(self, model_path, enable_conv2d=True,
        enable_vocab_split=False, enable_vocab_split8=True, enable_logits2=True,
        enable_coreml=False, mlp_up_split=1, mlp_down_split=1, enable_debug=False)  # :21-45
@abstractmethod preprocess(self)          # :47
@abstractmethod validate(self)            # :52
@abstractmethod forward(self, *args, **kwargs)  # :57
~~~

In practice `preprocess` / `validate` are never implemented by the shipped models
(`LlamaModel` implements only `load_pretrained_weights` and `forward`), and nothing
calls `validate`. Subclass it for the container (`DoryModel(BaseModel)`), exactly
as `LlamaModel(BaseModel)` does at `anemll/models/llama_model.py:754`.

**`anemll/ane_converter/base_converter.py:3-25`** — `BaseConverter(ABC)` with
`self.model`, `preprocess()` (no-op hook), abstract `convert()`,
`postprocess(num_workers=None)`. Every converter implements `postprocess` for LUT
quantization only.

### 1.3 The model-class API you must reproduce

Four classes per architecture. Study the Gemma 3 quartet — it is the only one built
for heterogeneous caches and is the template for Dory:

1. **`<Arch>Config(**kwargs)`** — a plain object (not a dataclass) that
   `kwargs.get()`s every HF `config.json` field it needs and adds three
   ANEMLL-only fields: `context_length`, `state_length`, and (Gemma 3)
   `attention_size`, `use_split_cache`, `force_rotation_mode`, `batch_size`.
   Precedents: `llama_model.py:54-92`, `gemma3_model.py:67-195`. `from_json` must
   unwrap multimodal `text_config` (`gemma3_model.py:180-195`) — Dory has no vision
   tower today but a nested config costs nothing to accept.
2. **`<Arch>Model(BaseModel)`** — `__init__` builds an `nn.ModuleList` of decoder
   layers and `register_buffer`s the KV cache(s); `load_pretrained_weights`;
   `process_layer_regular` / `process_layer_prefill` (+ `_prefill_rotate` for SWA);
   `process_layers(hidden, position_ids, causal_mask, current_pos, start_layer,
   end_layer, IN_PREFILL=...)`; `forward` / `forward_prefill`.
   Precedent: `gemma3_model.py:767-1886`.
3. **`<Arch>ForCausalLM(nn.Module)`** — embeddings + `self.model` + LM head split
   into N `nn.Conv2d(hidden, vocab//N, 1)`; a `load_pretrained_weights` that loads
   both halves; `forward(input_ids, update_mask, position_ids, current_pos,
   causal_mask, IN_PREFILL=False)`; `prefill_kv_cache(...)`.
   Precedent: `llama_model.py:1139-1419`, `qwen_model.py:986-1200`.
4. **`<Arch>Converter(BaseConverter)`** — `convert(split_part)` dispatch, one
   `convert_<part>` per artifact, `GetTransformerStates`, `postprocess`,
   `test_conversion`, `main`.
   Precedent: `llama_converter.py:38-1586`, `gemma3_converter.py:38-2700`.

**Constructor signature that the converter instantiates.** The converter owns the
model, not the other way round:

~~~python
converter = DoryConverter(model=DoryForCausalLM(config, enable_coreml=True),
                          context_length=..., state_length=..., lut_bits=...,
                          per_channel=8, batch_size=64, num_chunks=N,
                          argmax_in_model=False)
converter.convert(split_part="2")   # -> ct.models.MLModel
~~~

(`llama_converter.py:41-69`, `:71-128`; the traced wrapper is defined *inside* each
`convert_*` method and holds `self.states = GetTransformerStates(...)`, e.g.
`llama_converter.py:778-795`, `gemma3_converter.py:1447-1483`.)

### 1.4 State declaration — the part that must be gotten exactly right

Three things must agree byte for byte, or CoreML fails with "state is unused" /
"state shape mismatch":

1. **A real `register_buffer`** on the traced module, dtype `torch.float16`, shape
   `(2 * n_layers_in_this_state, n_kv_heads, cache_len, head_dim)` — K and V
   interleaved along dim 0 (`llama_model.py:778-800`, `gemma3_model.py:813-863`).
   Layer *i* uses row `[i]` for K and `[i + n_layers]` for V
   (`llama_model.py:883-884`).
2. **A `ct.StateType`** whose `name` is the *buffer's qualified name*, i.e. the
   attribute path from the traced root: `model.model.kv_cache_0`
   (`llama_converter.py:180`) or `model.model.kv_cache_local` /
   `model.model.kv_cache_global` (`gemma3_converter.py:357`, `:369`). The prefix is
   literally the string `"model.model."` because the wrapper is
   `ForCausalLM` -> `.model` (the `<Arch>Model`) -> buffer.
3. **`ct.convert(..., states=wrapper.states)`** on the traced wrapper
   (`llama_converter.py:842`, `:953`, `:1181`; `gemma3_converter.py:1522`,
   `:1745`, `:1232`).

**Union-of-states rule.** Every chunk that will share one runtime `MLState` must
declare the *same* state list, even if that chunk touches no layer of a kind.
Gemma 3 states this outright — "Always declare BOTH state types for all chunks,
even if chunk doesn't have that layer type ... CoreML requires consistent state
declarations" (`gemma3_converter.py:341-345`) — and then dodges CoreML's
unused-state error by touching both caches and adding `0 *` into the output
(`gemma3_converter.py:1474-1481`):

~~~python
dummy_local  = self.model.model.kv_cache_local[0,0,0,0]  * 0.0
dummy_global = self.model.model.kv_cache_global[0,0,0,0] * 0.0
out = out + (dummy_local + dummy_global).view(1, 1, 1)
~~~

This is *the* trick Dory needs: the runtime creates **one** state object and passes
it to every chunk (`tests/chat.py:1274-1333`), so a union is not optional. If
chunks declare different state sets, `create_unified_state` silently falls back to
per-chunk states and "KV cache won't be fully shared"
(`tests/chat.py:1334-1350`) — which for Dory is fatal, because the input block
writes caches the recurrent block must read.

State names seen by the runtime are the attribute paths; `detect_cache_type`
(`tests/chat.py:587-745`) sniffs `kv_cache_local` / `kv_cache_global` /
`kv_cache_0` out of `repr(make_state())` to pick its attention path.

### 1.5 Tracing discipline (every converter does this, in this order)

1. `preprocess()` — `self.model.to(TEST_DEVICE)`, `.eval()`,
   `requires_grad_(False)`, `apply(set_eval_and_freeze)`
   (`llama_converter.py:414-440`). `TEST_DEVICE` is the literal `"cpu"`
   (`llama_model.py:32`).
2. Build the wrapper (`torch.nn.Module`) that returns the part's outputs; set
   `wrapper.states = ...`.
3. `_reset_kv_cache_buffers(wrapper)`, then `torch.jit.trace(...)`, then
   `_reset_kv_cache_buffers(wrapper)` **and** `_reset_kv_cache_buffers(traced)`
   (`llama_converter.py:544-550`, `:817-823`). Skipping this bakes the sample
   trace's cache writes into the graph.
4. `ct.convert(traced, inputs=[...], outputs=[...], states=...,
   compute_precision=ct.precision.FLOAT16,
   compute_units=ct.ComputeUnit.CPU_AND_NE,
   minimum_deployment_target=ct.target.iOS18, convert_to="mlprogram")`
   (`llama_converter.py:838-847`). **All** conversions use iOS 18 / macOS 15
   stateful models; there is no other target in the tree.
5. `postprocess()` for LUT, then `AddMetadata(...)`, then `.save(path)`.

### 1.6 Exact tensor contract of a transformer chunk (copy these names and shapes)

`infer` (`llama_converter.py:803-835`, `gemma3_converter.py:1488-1521`):

| name | shape | dtype |
|---|---|---|
| `hidden_states` | `(1, 1, hidden)` | float16 |
| `position_ids` | `(1,)` | int32 |
| `causal_mask` | `(1, 1, 1, context_length)` — or `attention_size` for Gemma 3 | float16 |
| `current_pos` | `(1,)` | int32 |
| out `output_hidden_states` | `(1, 1, hidden)` | float16 |

`prefill` (`llama_converter.py:911-946`, `gemma3_converter.py:1678-1746`):

| name | shape | dtype |
|---|---|---|
| `hidden_states` | `(1, batch_size, hidden)` | float16 |
| `position_ids` | `(batch_size,)` | int32 |
| `causal_mask` | `(1, 1, batch_size, context_length)` | float16 |
| `current_pos` | `(1,)` | int32 |
| optional `update_mask` | `(1, 1, mask_len, batch_size)` | float16 |
| out `output_hidden_states` | `(1, 1, hidden)` (see caveat) | float16 |

Caveats worth knowing before you copy them:

- `batch_size` is a **compile-time constant** (default 64). Prefill cannot take a
  shorter sequence; the runtime handles the tail by looping single tokens
  (`tests/chat.py:1088-1116`) and forces single-token mode entirely when neither
  `update_mask_prefill` nor `prefill_dynamic_slice` is set
  (`tests/chat.py:1358-1362`).
- Llama-style prefill deliberately **skips the MLP of the last layer** and returns
  `hidden_states[:, 0:1, :]` (`llama_model.py:905-912`, `:1055-1058`) — an
  optimisation meaning "prefill exists to fill the cache, its output is unused".
  Gemma 3 does not do it: its last chunk applies the final norm and returns the
  full batch (`gemma3_converter.py:1470-1472`). Dory's five-component chain must
  choose deliberately (see 8.6).
- `update_mask` is an *optional extra input* added only for the non-rotate,
  non-dynamic-slice prefill variant (`gemma3_converter.py:813`, `:1099`), and the
  runtime probes for it at first use (`tests/chat.py:994-1016`).
- `current_pos` is always a 1-element int32 array even for prefill; it is read
  inside the graph as a scalar, and slicing must use `torch.narrow` with positive
  starts (see 5.1).

### 1.7 Chunk boundaries inside a converter

Balanced layer split, same formula in all three converters
(`llama_converter.py:761-768`, `gemma3_converter.py:1434-1445`):

~~~python
base, rem = divmod(total_layers, num_chunks)
start = chunk_idx * base + min(chunk_idx, rem)
end   = start + base + (1 if chunk_idx < rem else 0)
~~~

Only `gemma3_converter` exposes converting *one* chunk at a time: `--chunk-no N`
(`gemma3_converter.py:531-560`, `:2134`); llama/qwen loop internally. **For Dory
this formula must be replaced** by explicit, structurally chosen ranges 16 / 40 /
16 (8.1) — the converter part functions already accept `start_layer` /
`end_layer` internally (`gemma3_converter.py:1447-1456`), so exposing them on the
CLI is a ~10-line change, not a redesign.

### 1.8 Weight loading conventions

- Key rewrite: HF `model.layers.N...` to `layers.N...` for the inner model
  (`llama_model.py:1240-1263`); Gemma 3 strips `language_model.` then `model.`
  (`gemma3_model.py:1901-1911`).
- Every projection that will be a 1x1 conv is reshaped **on load**, not in the
  module: `v.view(v.shape[0], v.shape[1], 1, 1)` (`llama_model.py:1252`,
  `:1257`; `gemma3_model.py:1926`). `o_proj` stays 2-D in Llama
  (`llama_model.py:1248-1250`) but becomes a `Conv2d` in Gemma 3
  (`gemma3_model.py:1917-1926`) — both work; follow Gemma 3 for Dory (uniform
  layout).
- LM head is split at load time into `lm_head8_1..8` / `lm_head2_1..2` /
  `lm_head1` by `torch.split` on the reshaped weight (`llama_model.py:1208-1223`);
  the split is **even** (`vocab_size // N`) — an uneven vocab would break this.
- Missing keys: `ANEMLL_ALLOW_MISSING_WEIGHTS=1` / `--allow-missing-weights`
  converts anyway, printing a red TODO (`llama_model.py:1229-1236`,
  `convert_model.sh:120`).
- Expected-missing buffers are whitelisted (`kv_cache_0`,
  `rotary_emb.inv_freq`) (`llama_model.py:1269-1272`).

---

## 2. Commands, flags, and what success looks like

### 2.1 One-shot conversion

~~~bash
./anemll/utils/convert_model.sh \
    --model  /path/to/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 \
    --output /path/to/dory-ane-out \
    --prefix dory \
    --context 24576 \
    --batch 64 \
    --chunk 4 \
    --lut1 8 --lut2 6 --lut3 6 \
    --skip-check
~~~

Flag semantics, from the script's own usage block (`convert_model.sh:80-136`) and
defaults (`:39-72`): `--context` (512), `--batch` (64), `--lut1|--lut2|--lut3`
(embeddings none / FFN 4 / LM head 6), `--chunk N|auto` (2), `--max-chunk-mb`
(950), `--prefix` (llama), `--restart 1-8`, `--only 1-8`, `--argmax`,
`--split-rotate`, `--single-cache`, `--dynamic-prefill-slice` (ON) /
`--static-prefill-slice`, `--fp16-scale`, `--clamp`, `--chunk-no`,
`--allow-missing-weights`, `--force-mlprogram-compile`, `--skip-anemll-dedup`,
`--skip-check`.

Steps (`convert_model.sh:150-158`, implemented at `:500-900`): 1 embeddings,
2 LM head, 3 FFN, 4 prefill, 4a FFN rotate, 4b prefill rotate (Gemma 3 only),
5 combine, 6 compile, 7 tokenizer + meta.yaml, 8 chat smoke test.

### 2.2 Part-by-part (what the script actually runs)

~~~bash
# 1 embeddings, 3 LM head (part numbers are fixed by the converters, not by arch)
python -m anemll.ane_converter.dory_converter --part 1 --model M --output O \
    --prefix dory --context-length 24576 --batch-size 64 [--lut 8]
python -m anemll.ane_converter.dory_converter --part 3 --model M --output O \
    --prefix dory --context-length 24576 --lut 6 [--argmax]

# 2 FFN (infer) and 2_prefill, per chunk
python -m anemll.ane_converter.dory_converter --part 2 --model M --output O \
    --prefix dory --context-length 24576 --batch-size 64 --chunk 4 --lut 6
python -m anemll.ane_converter.dory_converter --part 2_prefill --model M --output O \
    --prefix dory --context-length 24576 --batch-size 64 --chunk 4 --lut 6 \
    --dynamic-prefill-slice

# rotate variants only exist in the Gemma 3 converter today
python -m anemll.ane_converter.gemma3_converter --part 2_rotate ...          # infer_rotate
python -m anemll.ane_converter.gemma3_converter --part 2_prefill_rotate ...  # prefill_rotate
~~~

`--part` choices are enumerated per converter: llama
`{1,2,2_prefill,3,all,monolithic,monolithic_prefill}`
(`llama_converter.py:1261-1264`); gemma3 adds `2_rotate`, `2_prefill_rotate`,
`monolithic_rotate`, `monolithic_prefill_rotate`, `full`, `prefill`, `embeddings`
(`gemma3_converter.py:2157`). **INVENT**: Dory's `--part` set must be extended
beyond "layer ranges" — five semantically different components, not two
(`embeddings`, `input`, `recurrent`, `output`, `lm_head`).

### 2.3 LUT quantization — what the flags really do

`--lut 6` / `--lut 6,4` / `--lut 6,0` are parsed by `parse_lut_arg`
(`llama_converter.py:1199-1243`): bits, then optional per-channel group size;
default group size **8**; `0` / `tensor` / `t` select per-tensor. Everything
funnels into coremltools:

~~~python
cto.coreml.OpPalettizerConfig(mode="kmeans", nbits=nbits,
    granularity="per_grouped_channel", group_size=per_channel,
    num_kmeans_workers=num_workers)
# or granularity="per_tensor" when per_channel <= 0
~~~

(`llama_converter.py:442-458`; `gemma3_converter.py:397-415` adds
`weight_threshold`). So "LUT4 / LUT6 / LUT8" = `nbits=4/6/8`; there is **no** LUT6
helper module, no per-layer palettization table, no custom codebook anywhere in
`anemll/` (grep for `lut6|LUT6|per_channel` returns only these config builders and
filenames). Quantization is applied to the whole converted `MLModel` after
`ct.convert`, via `cto.coreml.palettize_weights` (`llama_converter.py:530`).

Per-component overrides exist only for **monolithic** models: `--lut-embeddings`
and `--lut-lmhead` are applied by matching **op names** in the MIL program
(`"embed_tokens" in op_name`, `"lm_head" in op_name`,
`llama_converter.py:496-521`). For chunked conversion the three LUT widths come
out of the three separate converter invocations (`--lut1/2/3`), which is what the
spec's D8 needs.

Known wart to code around: palettize can fail with `ValueError: Pool not running`
on the second chunk; both llama and gemma3 retry single-threaded
(`llama_converter.py:532-537`; `gemma3_converter.py:1530-1535` documents the bug).

### 2.4 LM-head splitting

The split is a *model-construction* decision, not a flag: `ENABLE_VACAB_SPLIT8`
(`llama_model.py:44`) creates `lm_head8_1..8`; Qwen adds a 16-way
(`qwen_model.py:1013-1017`); Gemma 3 uses 16 for its 262K vocab
(`chunk_size = 16384  # 262144 / 16`, `gemma3_converter.py:837`). Outputs are
named `logits1..logitsN` (`llama_converter.py:697-706`), or a single `logits`
when one head. Concat order is by head index (`llama_model.py:1400`, `:1410`),
and the runtime rebuilds the full vocab in that same order using `split_lm_head`
from metadata (`tests/chat.py:1247-1269`), falling back to `lm_head_chunk_sizes`
when the split is uneven (`tests/chat.py:276-357`).

For Dory: **131,072 = 16 x 8192**, so 16 even chunks of 8192 output channels —
safely under both the 16,384 guidance and the 65,536 axis cap. `ENABLE_CONV2D`
and the 8/2/16-way splits are module-level constants in `llama_model.py:41-46`,
i.e. chosen at import time, not by CLI.

In-model argmax (`--argmax`) replaces the logits outputs with `argmax_idx` (int32,
one **local** index per vocab chunk) and `argmax_val` (fp16)
(`llama_converter.py:661-671`, `:692-696`; gemma3 `:1045-1069`, note the comment
"we return LOCAL indices ... global offset computed on the Python/Swift side").
The runtime takes the chunk with the largest value and adds that chunk's offset
(`tests/chat.py:1214-1245`). This is the right lever for a 131K vocab on ANE.

### 2.5 Combine (multifunction) and compile

~~~bash
# 2-function chunk (infer + prefill), deduped by default
python anemll/utils/combine_models.py --chunk 4 --lut 6,4 --prefix dory \
    --input O --output O
# 4-function chunk (infer + infer_rotate + prefill + prefill_rotate)
python anemll/utils/combine_models.py --chunk 4 --lut 6 --prefix dory --gemma3 \
    --input O --output O
# rotate functions split into separate files (2 files per chunk)
python anemll/utils/combine_models.py --chunk 4 --lut 6 --prefix dory --split-rotate ...
~~~

(`combine_models.py:669-690`.) Weight sharing between the functions is *not*
automatic: `_save_multifunction_dedup` runs `prepare_dedup_sources` first, because
k-means converges to different (but semantically identical) LUT bytes per function
and "CoreMLTools dedup compares raw bytes", costing 15-40 % size
(`anemll/utils/dedup_weights.py:6-30`). Do not disable it for Dory —
`--skip-anemll-dedup` exists precisely as an escape hatch.

~~~bash
# compile everything to .mlmodelc
python anemll/utils/compile_models.py all --input O --output O --recursive
python anemll/utils/compile_models.py 2 --lut 6 --chunk 4 --prefix dory --input O --output O
~~~

(`compile_models.py:165-175`.) Mechanically it is
`xcrun coremlcompiler compile <pkg> <dir>` (`compile_models.py:15-42`), with
`--add-mlprogram-if-eligible force` behind `--force-mlprogram`.

### 2.6 Verifying ANE placement — what the evidence looks like

~~~bash
python anemll/utils/ane_profiler.py --model <chunk>.mlpackage --analyze
python anemll/utils/ane_profiler.py --model <chunk>.mlpackage --report --output r.json
python anemll/utils/ane_profiler.py --model <chunk>.mlpackage --compare
python anemll/utils/ane_profiler.py --model <chunk>.mlpackage --benchmark-all -n 20 -w 5
~~~

It compiles the package into a temp dir with `xcrun coremlcompiler` and loads
`MLComputePlan` (`ane_profiler.py:91-143`), then walks every op and asks
`compute_plan.get_compute_device_usage_for_mlprogram_operation(op)`
(`ane_profiler.py:201`). **Pass looks like** (`ANE_PROFILER.md:42-56`):

~~~
Total operations: 4630
  Const/metadata: 2996
  Executable ops: 1634
    ANE: 1622 (99.3%)
    GPU: 12
    CPU: 0
--- GPU Fallback Operations (12) ---
  - ios18.cast:current_pos_to_int16
~~~

The gate the spec wants ("every op of every chunk on the ANE") is `GPU: 0`
**and** `CPU: 0`, i.e. `ane_percentage == 100.0` over executable ops
(`ane_profiler.py:262-280`). Anything else is printed by name, so the offending op
is directly identifiable. Usual suspects: `select`, `ios18.cast`,
`ios18.gather`, `greater_equal`, `while_loop` (`ANE_PROFILER.md:161-170`).

Two hard limits on this evidence: (a) **stateful models cannot be benchmarked or
prediction-tested** by the profiler — random inputs carry no `MLState`, so
`--test` / `--benchmark` fail with "must be an MLState" while `--analyze`,
`--compare`, `--costs` keep working (`ANE_PROFILER.md:251-263`); (b) placement is
per-function, so a 4-function chunk must be profiled **per function** (or the
report read per function).

---

## 3. The meta/config schema in 0.3.5

### 3.1 meta.yaml — chunked layout (`generate_meta_yaml.py:398-509`)

~~~yaml
model_info:
  name: anemll-<MODEL_NAME>-ctx<CONTEXT>
  version: 0.3.5
  description: |
    ...            # free text; Context / Batch / Chunks are interpolated (:452-456)
  license: MIT
  author: Anemll
  framework: Core ML
  language: Python
  architecture: <ARCH>          # from config.json model_type (:305-321)
  parameters:
    context_length: <CONTEXT>
    batch_size: <BATCH>
    lut_embeddings: <bits|none>            # checked against files on disk (:414-420)
    lut_embeddings_per_channel: <int>      # only if given
    lut_ffn: <bits|none>
    lut_ffn_per_channel: <int>             # only if given and not none
    lut_lmhead: <bits|none>
    lut_lmhead_per_channel: <int>
    argmax_in_model: true                  # only when true (:479)
    vocab_size: <int>                      # only from ANEMLL_CONVERSION_INFO
    lm_head_chunk_sizes: [a, b, ...]       # only when known (:488-491)
    num_chunks: <N>                        # :503
    model_prefix: <prefix>
    embeddings: <prefix>_embeddings[_lutN].mlmodelc
    lm_head:    <prefix>_lm_head[_lutN].mlmodelc
    ffn:        <prefix>_FFN_PF[_lutN]_chunk_01ofNN.mlmodelc
    pf:         <prefix>_FFN_PF..._rot.mlmodelc     # --split-rotate only (:495)
    split_lm_head: 8|16
    split_rotate: true                     # --split-rotate only
    sliding_window: <int>                  # --sliding-window only (:498)
    update_mask_prefill: true              # --update-mask-prefill only
    prefill_dynamic_slice: true            # default true (:335-344)
    single_cache: true                     # --single-cache only
~~~

Monolithic variant (`generate_meta_yaml.py:164-222`) replaces the three model
paths with `model_type: monolithic`, `model_prefix`,
`monolithic_model: <prefix>_monolithic_full[_lutN].mlmodelc`, and a `functions:`
list — either `[infer, prefill]` or, with `--rotate`,
`[infer, infer_rotate, prefill, prefill_rotate]` (`:206-216`). `split_lm_head` is
16 for `qwen*` / `gemma*`, else 8 (`:157-161`, `:443-446`).

`ANEMLL_CONVERSION_INFO` (a JSON blob exported by `convert_model.sh:806-860`) is
appended as **YAML comments** only (`generate_meta_yaml.py:240-297`), for humans.

### 3.2 Per-file CoreML metadata (com.anemll.*)

`AddMetadata` (`metadata.py:58-147`) writes, all as strings:

| key | written when |
|---|---|
| `com.anemll.info` | always (`Converted with Anemll vX`) |
| `com.anemll.context_length` | always (`:107-108`) |
| `com.anemll.num_chunks`, `com.anemll.chunk_no` | chunked parts |
| `com.anemll.batch_size` | prefill parts |
| `com.anemll.lut_bits` | quantized parts |
| `com.anemll.argmax_in_model` | only when true (`:122-123`) |
| `com.anemll.vocab_size`, `com.anemll.lm_head_chunk_sizes` | LM head |
| `com.anemll.functions` | combined models (comma-joined) |

Plus `author`, `version`, `short_description` (`:88-89`, `:138-147`).
`AddCombinedMetadata` merges source metadata and unions the function lists
(`metadata.py:222-269`).

### 3.3 What the runtime actually reads

`tests/chat.py:1651-1843` (meta.yaml) and `tests/chat.py:472-585` (per-file):
`model_prefix`, `context_length`, `batch_size`, `num_chunks`, `lut_*` (to rebuild
filenames), `split_lm_head`, `argmax_in_model`, `vocab_size`,
`lm_head_chunk_sizes`, `sliding_window`, `update_mask_prefill`,
`prefill_dynamic_slice`, `split_rotate`, `attention_size` (mask width, defaults to
`context_length`, `:1791`), optional `tokenizer_path`, `monolithic_model`,
`model_type`.

### 3.4 How context, chunk layout and KV mode are recorded — plainly

- **Context length**: `parameters.context_length` in meta.yaml **and**
  `com.anemll.context_length` in every `.mlpackage`. `state_length` is *read*
  (`tests/chat.py:480`, `:1684`) but **is never written** by `AddMetadata` — it
  silently defaults to `context_length`. For Dory (context 24,576 vs cache rows
  24,576) that identity happens to hold; do not rely on it elsewhere.
- **Chunk layout**: `parameters.num_chunks` plus the three path fields and the
  filename convention `_chunk_NNofMM`; `split_lm_head` / `lm_head_chunk_sizes`
  describe the LM-head split. There is no explicit "components" list — the
  *filename* is the interface (`combine_models.py:134-181`).
- **KV mode**: **not recorded at all** in 0.3.5. State layout is discovered at
  runtime from `make_state()` / the spec (`detect_cache_type`,
  `tests/chat.py:587-745`), and every state is `np.float16`
  (`llama_converter.py:163`, `:178`; `gemma3_converter.py:355`, `:368`). There is
  no INT8 KV, no "KV mode" key, no per-cache dtype. Dory's `last_loop` (D4) and
  loop range therefore need **new meta keys** (INVENT): e.g.
  `kv_mode: last_loop`, `loops: {min: 1, max: 4, default: 2}`,
  `tokenizer: tokenizer.json`, `chat_template: chat_template.jinja`, and a
  `components:` mapping (name -> file -> functions -> calls per token) so the
  runtime does not have to hardcode five paths.

---

## 4. The chat runtime: how chunks are driven

`tests/chat.py` is the reference runtime (the Swift CLI mirrors it). Its flow is
the one to copy for Dory's Python runtime and server.

### 4.1 Wiring

- `parse_args` -> `--meta` parse (`:1651-1843`) -> `load_models` (`:746-901`):
  embeddings + LM head are single models; the FFN path is expanded by
  `find_all_chunks` (`:376-380`) and each chunk becomes a **dict**
  `{'infer': model, 'prefill': model[, 'infer_rotate': ..., 'prefill_rotate': ...]}`
  (`:818-822`); rotate functions are loaded only when
  `context_length > sliding_window` (`:843-861`).
- `load_model` (`:382-470`) picks `CompiledMLModel` for `.mlmodelc` and
  `MLModel` for `.mlpackage`, and **probes** the function list before selecting
  one, raising a readable error listing available functions (`:413-427`). Note it
  deliberately omits `compute_units=` for multi-function packages so
  `make_state()` works (`:448-449`).
- **State**: `create_unified_state` (`:1274-1350`) calls `model.make_state()`
  once on the chunk that has the global cache (`global_cache_chunk_idx`) and
  passes that single object to every chunk of every component (`:1081`,
  `:1112`).
- **Masks**: `make_causal_mask(length, start)` builds a `(1,1,L,L)` fp16 mask with
  `-inf` above the diagonal (`:977-983`); the row for token `pos` is sliced per
  step: `causal_mask[:, :, pos-1:pos, :]` (`:1160`).
  `make_update_mask(mask_len, batch_pos, batch_size)` is the
  `(1,1,mask_len,batch_size)` one-hot that tells a prefill chunk where to write
  its rows (`:985-992`).

### 4.2 Prefill, then decode

Prefill (`run_prefill`, `:1026-1118`):

1. For each full batch of 64 (`while batch_pos + batch_size <= context_pos`):
   embed `input_ids[:, pos:pos+64]` (`:1062-1065`), then call each chunk with
   `hidden_states`, `position_ids=arange(pos, pos+64)`,
   `causal_mask[:, :, pos:pos+64, :]`, `current_pos=[pos]`, and (if enabled)
   `update_mask` (`:1074-1085`). The output of chunk *i* is the input of *i+1*.
2. Choose the function: `'prefill_rotate' if batch_pos >= sliding_window and
   has_rotation else 'prefill'` (`:1068`).
3. Tail tokens `< batch_size` go through the **single-token function** one at a
   time (`:1088-1116`), choosing `infer_rotate` once
   `batch_pos >= sliding_window` (`:1095`).

Decode (`generate_next_token`, `:1120-1272`): function choice is `infer_rotate`
when `(pos - 1) >= sliding_window` (`:1136`); then embed one token;
`position_ids=[pos-1]`; `causal_mask[:, :, pos-1:pos, :]`; iterate the chunks with
the shared state; finally the LM head and either argmax (`:1206-1245`) or
concat-and-argmax over `split_lm_head` logits (`:1247-1272`).

`chat_loop` shows the whole sequence end to end: template -> `run_prefill(...)` ->
`while pos < context_length-1: generate_next_token(...)` (`:1408-1522`).
`context_pos` is the prompt length; `pos` starts there.

### 4.3 Tokenizer wiring

`initialize_tokenizer(model_path)` (`:903-...`) loads `AutoTokenizer` from the
directory that meta.yaml sits in (`args.tokenizer = args.d`, `:1712-1717`,
`:1776-1780`), so the tokenizer files are **copied next to the models** by
conversion step 7 (`convert_model.sh:812-830`: `tokenizer.json`,
`tokenizer_config.json`, `tokenizer.model`, `vocab.json`, `merges.txt`,
`chat_template.jinja`, `added_tokens.json`, `special_tokens_map.json`,
`generation_config.json`, `preprocessor_config.json`) plus a minimal
`config.json` written by `anemll.ane_converter.create_config_json`
(`create_config_json.py:10-31`). Prompt formatting uses
`tokenizer.apply_chat_template(messages, add_generation_prompt=True, **kwargs)`
when a template exists, falling back to `[INST] ... [/INST]` (`:1408-1431`).
Stop tokens are discovered by string from the tokenizer (`build_stop_token_ids`,
`:239-...`). For Dory: `tokenizer.json` from the BF16 repo and
`chat_template.jinja` are the two files that matter; the `--no-think` plumbing in
`chat_loop` (`:1352`) is the precedent for toggling reasoning.

---

## 5. The sliding-window fill/rotate prior art (Gemma 3, window 512)

**It is present in 0.3.5.** It is not hidden on a branch: `gemma3_model.py` and
`gemma3_converter.py` in the released tree implement the full scheme, and
`combine_models.py:346-453` assembles the four functions. History confirms the
scheme arrived in `fa9c6cb` ("split KV cache for Gemma3 local vs global attention
layers"), `ae16a45` ("prefill rotation support"), `3079cf6` ("static slicing and KV
cache updates"), and was merged for the beta in `f8f7df7`.

### 5.1 The idea

The sliding cache is a **fixed-length ring implemented as a shift**: it always
holds exactly the last `sliding_window` rows, in order, so local attention never
needs an offset and never needs a mask past `sliding_window`
(`gemma3_model.py:906-926`).

- While `current_pos < sliding_window` the cache is being **filled**: write at
  `[current_pos]` (or via `update_mask`) — `_fill_kv_local`,
  `gemma3_model.py:967-992`, `:1100-1108`.
- Once full, every step **rotates**: drop row 0, append the new row —
  `_update_kv_local`, `gemma3_model.py:1024-1065`:

~~~python
sw = self.config.sliding_window
key_slice   = self.kv_cache_local[key_cache_idx:key_cache_idx + 1]
key_tail    = torch.narrow(key_slice, 2, 1, sw - 1)      # start=1, length=sw-1
shifted_key = torch.cat([key_tail, key_states], dim=2)   # [1,H,sw,D] again
self.kv_cache_local[key_cache_idx:key_cache_idx + 1, :, :, :] = shifted_key
# same for V with value_cache_idx = key_cache_idx + num_local
~~~

The comments are the load-bearing part: positive indices only, because
`cache[:-1] = cache[1:].clone()` fails under `torch.jit.trace` and
`cache[:, :, 1:, :]` produces "offset (-1)" errors on ANE
(`gemma3_model.py:1035-1039`).

`force_rotation_mode` (`gemma3_model.py:144-148`, dispatch at `:1090-1120`) makes
the choice a **compile-time** one, because a data-dependent branch cannot be
traced:

- `None` -> conditional on `current_pos` (eager use only, not traced);
- `False` -> always fill -> **`infer` / `prefill`**;
- `True` -> always rotate -> **`infer_rotate` / `prefill_rotate`**.

The converter sets it on the config immediately before tracing
(`model.model.config.force_rotation_mode = force_rotation`,
`gemma3_converter.py:1425`; prefill equivalent `:1582`), i.e. two traces of the
same weights, then `combine_models.py` welds the four `.mlpackage`s into one file
with four functions, weights deduped (`combine_models.py:414-432`).

### 5.2 The exact host call sequence

From `tests/chat.py` and `run_monolithic_prefill_with_rotation` (`:1974-2110`),
which spells the two-phase split out:

| situation | function called | cache action |
|---|---|---|
| prefill batches with `batch_pos + 64 <= sliding_window` | `prefill` | fill at `pos` via `update_mask` |
| prefill tail (partial batch) **before** the window | `infer` per token | fill |
| prefill batches with `batch_pos >= sliding_window` | `prefill_rotate` | shift by `batch_size`, store last 64 |
| prefill tail **after** the window | `infer_rotate` per token | shift by 1 |
| decode while `pos-1 < sliding_window` | `infer` | fill |
| decode while `pos-1 >= sliding_window` | `infer_rotate` | shift by 1 |

Prefill rotate (`_update_kv_local_prefill`, `gemma3_model.py:1167-1208`) shifts by
`batch_size` and appends the batch — and it deliberately uses
`self.config.batch_size` (a constant) rather than the runtime `seq_len` "for
CoreML tracing compatibility" (`:1178-1194`). The constraint is
`batch_size < sliding_window` (true: 64 < 512; for Dory 64 < 1024).

The prefill **fill** path writes at `[0 : seq_len]` with a *static* slice
(`:1161-1165`), not at `current_pos`, and trims to the last `sliding_window`
tokens if `seq_len > sliding_window` (`:1147-1151`). The dynamic variant exists
behind `prefill_dynamic_slice` and is annotated "not ANE-safe; intended only for
testing" (`gemma3_model.py:160-162`, `:1153-1159`).

The mask: `causal_mask` is built at width `max(context_length, sliding_window)`
(`tests/chat.py:1048`) and attention slices it to the window
(`gemma3_model.py:640-648`).

### 5.3 What must be generalised for Dory

Gemma 3 has **one** local cache (15 layers, window 512) and **one** global cache
(3 layers, full context). Dory has **27 sliding layers** (window 1024) and **9
full** (24,576) inside the same call chain, split across three components. The
generalisable pieces are:

- the layer-to-cache-slot mapping by type: `get_layer_cache_mapping` counts
  same-type layers before the index (`gemma3_model.py:206-233`), used by every
  store path;
- the two-cache union declaration (`gemma3_converter.py:341-371`) — extend to *n*
  caches, one `ct.StateType` per cache with the same name everywhere;
- the rotate formula and its tracing constraints (positive `narrow`, static
  `batch_size`, `cat`), which transfer unchanged to 1024 rows;
- `_kv_slice` / attention window selection by layer type
  (`gemma3_model.py:906-926`, `:619-632`).

**INVENT**: a "rotate only some layers" variant. Gemma 3 traces a whole chunk
either in fill or in rotate mode; with Dory's three components each containing
both kinds, `force_rotation_mode` must become a *per-layer-type* switch (rotate
the sliding caches, always fill the full ones) — a ~3-line change at
`gemma3_model.py:1090-1120`, but it is not there today.

---

## 6. FP16 preflight and numerical validation

### 6.1 What it checks

`anemll/utils/fp16_compatibility_check.py` (998 lines), wrapped by
`anemll/utils/fp16_preflight.sh`:

1. **Weight range** — loads BF16, flags any tensor with `absmax > FP16_MAX`
   (65,504) (`:126-185`, threshold test at `:156`).
2. **Precision run** — BF16, FP16, and FP16-with-FP32-output-reference, each with
   forward hooks that count `NaN`, `Inf`, and values `> FP16_MAX` **per layer**
   (`:194-303`, hooks `:224-240`). This is how it names the first NaN layer.
3. **Residual-accumulation analysis** and **recommended scale alpha**
   (`analyze_residual_accumulation` `:577-...`, `compute_recommended_scaling`
   `:722-...`).
4. **Clamp sweep** — greedy-decodes with activation clamping at
   `[65000, 60000, ..., 20000]` and reports `match_rate` against the BF16
   baseline (`:379-496`).
5. **Verdict** — `ane_compatible`, `compatibility_score` (1.0 / 0.7 / 0.5 / 0.0),
   `issues`, `recommendations` (`:498-575`).

### 6.2 How to run it, and what a pass looks like

~~~bash
./anemll/utils/fp16_preflight.sh --model <path-or-hf-id>        # sweep on by default
python anemll/utils/fp16_compatibility_check.py --model <id> --sweep --save-report r.json
~~~

It auto-activates `env-anemll` if `VIRTUAL_ENV` is unset
(`fp16_preflight.sh:38-58`) and writes
`tests/dev/logs/fp16_preflight_<timestamp>.json` by default (`:78-84`).

A pass prints (`:538-573`):

~~~
ANE Compatible: ✓ YES
Compatibility Score: 100%
Precision Test Results:
  BF16:        ✓ PASS
  FP16:        ✓ PASS
  FP16→FP32:   ✓ PASS
~~~

(verbatim format strings at fp16_compatibility_check.py:545-552; ✓/✗ are literal)

and per-weight `All weights within FP16 range` (`:173-174`). A *failure* names the
first NaN layer and the overflow layers, then suggests either activation clamping
at a specific value or a weight-scaling alpha. ANEMLL's two remedies are documented
in `README.md:398-470`: **weight scaling** (scale `embed_tokens`, fold alpha into
the post-norm gains — "100 % match", zero runtime cost) and **runtime clamping**
(`--clamp`, adds ops). For Gemma 3 the shipped alpha values are 0.48 / 0.82 /
0.1875 for 270M / 1B / 4B-QAT.

### 6.3 The other validation patterns in-tree

- **Unit-level numerics**: `tests/test_qwen_inference.py` loads HF and the ANEMLL
  PyTorch model side by side and compares logits/argmax on one prompt with a
  hand-built causal mask (`:30-60`). The cheapest regression test to copy before
  any conversion.
- **End-to-end FP16, no LUT**: `tests/test_gemma3_1B_fp16_2chunk.py` runs
  `convert_model.sh` with empty `--lut1/2/3` and 4096 context, then
  `tests/chat.py --meta ... --prompt "Hello!"` (`:76-116`). This is the
  "unquantized pipeline smoke test" the spec's FP16 gate wants, and it shows that
  empty LUT strings mean "no quantization".

---

## 7. Follow-this-unless-the-spec-says-otherwise checklist

### 7.1 Converter

- [ ] One `DoryConverter(BaseConverter)`; `convert(split_part)` returns an
      `MLModel` **and** a `test_conversion` helper owns naming, metadata and
      saving (`llama_converter.py:1289-1505`).
- [ ] Every part: `preprocess()` -> reset buffers -> `torch.jit.trace` -> reset
      both -> `ct.convert(..., states=..., FLOAT16, CPU_AND_NE, iOS18,
      "mlprogram")`.
- [ ] Tensor names exactly `hidden_states`, `position_ids`, `causal_mask`,
      `current_pos`, `output_hidden_states`; `input_ids` only for embeddings and
      monolithic (`llama_converter.py:595`, `:1167-1178`, `:714-720`).
- [ ] `position_ids` / `current_pos` are **int32** in `ct.TensorType` even though
      traced as `long` (`llama_converter.py:828-830`).
- [ ] `batch_size` is a compile-time constant; document it in meta.
- [ ] LUT applied after conversion via `postprocess`; per-part widths come from
      separate invocations; per-op overrides only for monolithic.
- [ ] `AddMetadata` on **every** saved model before `.save()`.
- [ ] File naming `<prefix>_<part>[_lutN][_chunk_XXofYY].mlpackage` — the runtime
      and `compile_models.py` both parse it (`combine_models.py:134-181`).
- [ ] `--chunk-no`-style single-chunk conversion for large parts, so a failed
      chunk does not cost the whole run.
- [ ] Palette: `mode="kmeans"`, `granularity="per_grouped_channel"`,
      `group_size=per_channel` (default 8), retry single-threaded on
      "Pool not running".
- [ ] Fail loudly when a `.mlpackage` would exceed ~2 GB
      (`docs/convert.md:10-13`; `--max-chunk-mb` default 950 for auto chunking).

### 7.2 Model class

- [ ] `Config(**kwargs)` with `from_json`, adding `context_length` /
      `state_length`.
- [ ] All linears that feed the ANE are `nn.Conv2d(..., kernel_size=1,
      bias=False, dtype=torch.float16)` and weights loaded as `(out, in, 1, 1)`;
      data flows `(N,C,H,W)` via `x.permute(0,2,1).unsqueeze(2)` before and
      `.squeeze(2).permute(0,2,1)` after (`llama_model.py:382-388`,
      `:1384-1414`).
- [ ] No raw `sum(x*x)` / `norm` for normalisation: use the layer-norm trick
      (`llama_model.py:94-134`, `gemma3_model.py:241-282`), and divide by
      `sqrt(dim)` where a true L2 norm is wanted (see 8.4).
- [ ] RoPE: precomputed `cos_cached` / `sin_cached` buffers sized
      `max(context_length, state_length)*2` (`llama_model.py:180-187`), applied
      rotate-half style on the two halves (`llama_model.py:204-242`), selected
      **per layer** if bases differ (`gemma3_model.py:866-904`).
- [ ] KV writes are `tensor[a:b] = value` on a registered buffer, then read back
      from the buffer — never an out-of-graph cache (`llama_model.py:888-893`).
- [ ] Sliding caches: `narrow` + `cat`, positive indices, static sizes.
- [ ] Attention: mask added as a `(1,1,q,k)` fp16 tensor sliced from the model
      input; softmax is the manual max-subtract form (`llama_model.py:540-547`).
- [ ] Every `register_buffer` that backs a state is zeroed before and after
      tracing.
- [ ] Union of states declared per chunk + "touch" trick for unused ones.
- [ ] No Python `if` on tensor values anywhere in the traced path; all switches
      (`force_rotation_mode`, `prefill_dynamic_slice`) are read from **config** at
      trace time.

### 7.3 Tests

- [ ] `tests/test_dory_model.py` — HF-vs-PyTorch logits on one prompt, in the
      `test_qwen_inference.py` style.
- [ ] A `tests/conv/test_hf_model.sh`-style script: convert ->
      `tests/chat.py --meta ... --prompt ... --max-tokens 50`, asserting text,
      not tensors.
- [ ] FP16 smoke: convert with empty LUT strings, then chat (Gemma 3 precedent).
- [ ] Placement: `ane_profiler.py --analyze` on every function of every chunk,
      requiring `GPU: 0` / `CPU: 0`.
- [ ] `fp16_preflight.sh --model <weights>` recorded in `tests/dev/logs/`.
- [ ] Long-context recall and KLD harnesses belong to the project (INVENT) —
      ANEMLL ships no perplexity/KLD harness in these files.

### 7.4 ANEMLL's hard limits, where they bite Dory

| Limit | Where | Consequence for Dory |
|---|---|---|
| ANE tensor dimension <= 16,384 ("ANE tensors dimension size is limited to 16384") | `llama_model.py:179`, `qwen2_5_model.py:158`; 16-way LM head = 16,384 (`gemma3_converter.py:837`) | LM head chunks <= 16,384 output channels; the 24,576-row KV state is **above** this documented cap — the spec's measured 3.1 ms probe says it works, but D7's 16,384 + 8,192 tiling is the sanctioned fallback |
| ~2 GB per file (macOS), 1 GB iOS | `docs/convert.md:10-13` | the 40-block recurrent chunk at LUT6 is the pressure point; `--chunk auto` targets 950 MB (`convert_model.sh:61`) |
| iOS 18 / stateful models only | every `ct.convert` call | no fallback path to non-stateful |
| FP16 only | all `np.float16` states; `MODEL_DTYPE` (`llama_model.py:22`) | 864 MiB of full-attention state + ~108 MiB sliding, all fp16; no INT8 KV in 0.3.5 |
| No traced conditionals | `force_rotation_mode` design (`gemma3_model.py:144-148`) | every variant doubles as another traced function |
| Practical context ceiling 4,096 for Gemma 3 (512-2048 recommended) | `README.md:11`, `:170`, `:378` | 24,576 is 6x beyond any shipped configuration — expect new limits (state size, load time, mask width) |
| Even vocab split only (`vocab_size // N`) | `llama_model.py:1212-1221` | 131,072 is divisible by 16, so fine; do not introduce an uneven split in the model class |
| Multifunction weight bloat 15-40 % | `dedup_weights.py:6-30` | keep dedup on; a 4-function recurrent chunk without dedup is ~4x weights |

---

## 8. What in 0.3.5 fights a looped-recurrent model, and how to express it

Ordered by how much damage each does if ignored.

### 8.1 Heterogeneous components vs the balanced chunk split

`convert_model.sh` / the converters assume "N equal chunks of one transformer
stack" (`llama_converter.py:761-768`). Dory is 16 + 40 + 16 with different cache
sets and different call multiplicities. *Cleanest expression*: keep the converter's
`convert_part_2`-style wrapper but give it explicit `start_layer` / `end_layer`
(the wrapper already takes them — `gemma3_converter.py:1447-1456`) and add part
names `input` / `recurrent` / `output` mapping to (0,16), (16,56), (56,72). The
balanced formula then disappears for Dory; `num_chunks` in meta becomes a
component count, and the runtime calls each component a known number of times.

### 8.2 One shared MLState vs five components

The runtime creates one state from one chunk and passes it everywhere
(`tests/chat.py:1274-1333`). If any component declares a different set, the
fallback gives each chunk its own state and the caches stop being shared
(`:1334-1350`). *Cleanest*: declare the **union** — 9 full + 27 sliding caches —
in all five components, using the Gemma 3 pattern verbatim
(`gemma3_converter.py:341-371` plus the `dummy_*` touch at `:1474-1481`). The cost
is that every component carries a state it does not use; the cost of *not* doing it
is a silent correctness bug.

### 8.3 Blocks called N times per token — no loop input needed

This is the good news, and it should be written down before someone invents a
`loop_idx` input. `last_loop` semantics (D4) are reproduced *by construction* if
each loop pass writes the current rows into the same slots:

- decode, position `p`: every loop pass writes K/V for `p` into slot `p`,
  overwriting the previous loop's values; attention over `[0..p]` sees older
  positions' **final-loop** values and `p`'s **current-loop** value;
- prefill, rows `[p..p+63]`: identical, with the batch's rows overwritten per loop
  and later rows masked.

So the recurrent chunk needs **no loop argument at all**; the host simply calls it
`N` times with the same `current_pos` and the same state. What *is* needed is
`h_in` (the hidden state entering the recurrent block), because Dory's loop starts
with `h <- L2(h + h_in)`. Two options: an extra fp16 input `h_in (1,1,2560)` /
`(1,64,2560)` on the recurrent component, or a third CoreML state. The input is
more ANEMLL-like (a chunk is a function of inputs plus state, cf.
`hidden_states` / `position_ids` / `causal_mask` / `current_pos`) and costs
nothing; the state version would need the same union treatment everywhere.

Overhead to budget: `4 + N` `predict()` calls per token (embeddings, input,
recurrent xN, output, LM head) — the spec's D11 assumes exactly this.

### 8.4 Two RoPE bases inside one chunk, and L2 norms over 7,168 channels

Gemma 3 already solves the two-base problem: two `RotaryEmbedding` modules
(`self.rotary_emb_global` / `self.rotary_emb_local`, `gemma3_model.py:866-873`),
selected **per layer index** inside `process_layers`, never passed as a model
input, both precomputed as buffers sized to
`max(context_length, state_length)*2` (`gemma3_model.py:340`). Note the cap
interaction: a 24,576-token cos/sin table is 49,152 rows long, which exceeds the
16,384 axis guidance — precompute exactly the rows needed, or index a table of
`state_length` and keep the doubling only where it is really required
(`llama_model.py:180` doubles for no documented reason).

For the L2 norms, 0.3.5 gives you the norm but **not** the `/sqrt(dim)`: the
layer-norm trick (`F.layer_norm(concat([x, -x]))` then drop the mirrored half)
computes `x / rms(x)` = `sqrt(dim) * x / ||x||` (`llama_model.py:103-134`,
`gemma3_model.py:251-282`). A true L2 normalisation is therefore
`norm_fn(x) * (1/sqrt(dim))` — one constant multiply, which can be folded into the
following projection's weights (the same trick as D5's folding of `suv_*` and
`logit_scale`). **INVENT**: no file in 0.3.5 performs that `1/sqrt(dim)` step;
`stable_l2_norm` (`llama_model.py:1116-1137`) is the only L2 helper, it is unused,
and it uses `torch.linalg.norm` whose ANE placement is unverified.

### 8.5 Nine 24,576-row full caches in one chunk

State size is `2 x 9 x 4 heads x 24,576 x 256 x 2 B = 864 MiB`, and the sliding
block adds `2 x 27 x 4 x 1,024 x 256 x 2 B = ~108 MiB`. Both are *state*, not
weights, so they do not count against the ~2 GB file limit but do count against
load time and peak memory at first prediction. The spec's D7 fallback
(16,384 + 8,192 tiles combined with a log-sum-exp merge) has **no precedent in
0.3.5** — nothing in the tree tiles attention. The nearest structural precedent is
the split local/global cache (two states of different lengths in one model,
`gemma3_converter.py:346-371`), so the tiling would be expressed as two more states
and a log-sum-exp combine inside the same chunk (INVENT).

### 8.6 Prefill's "output is unused" optimisation must be dropped

`LlamaModel.forward` returns `hidden_states[:, 0:1, :]` in prefill and skips the
last layer's MLP (`llama_model.py:905-912`, `:1055-1058`), precisely because in
the Llama pipeline prefill only fills the cache and the next token is produced by
an `infer` call. Dory's five-component chain still ends in an output block + LM
head, so if prefill returns a single row the coherent convention is: **prefill is
cache-building only, and the first generated token is produced by one `infer` pass
over the last prompt token** — which is exactly what `tests/chat.py` already does
for the prefill tail (`:1088-1116`). Gemma 3's choice (return the whole batch from
the last chunk, `gemma3_converter.py:1470-1472`) is the alternative if you want one
fewer call. Pick one and make the recurrent component agree for both.

### 8.7 No conditionals -> the rotate/fill split multiplies artifacts

Dory has both sliding and full caches in each of the three block components, so a
naive port traces `infer`, `infer_rotate`, `prefill`, `prefill_rotate` for **each**
of three components = 12 traced graphs and, without dedup, 4 copies of each
component's weights. Mitigations, in order of preference: (a) make
`force_rotation_mode` per-layer-type (5.3), which keeps the four functions but
means `_rotate` only differs in the sliding caches; (b) `--split-rotate` (two
files per chunk, `combine_models.py:566-666`) if a 4-function file approaches
2 GB; (c) keep dedup on always.

### 8.8 Mask width and attention_size

The causal mask is a fixed-width tensor allocated to `context_length` (or
`attention_size`), passed in **every** call (`tests/chat.py:1151-1160`,
`gemma3_converter.py:1493-1495`). At 24,576 that is a 48 KB fp16 payload per call
per chunk — fine — but the *slice* must match the chunk's key width or the add
shapes mismatch; Gemma 3 solves it by slicing to `q_seq_len, k_seq_len` at use
time (`gemma3_model.py:640-648`). Copy that, do not copy Llama's eager
`causal_mask[:, :, :context_length]` (`llama_model.py:617`).

### 8.9 What has no precedent and must be built

- a meta key for **loop count range / default** and for **last_loop KV mode**
  (3.4);
- a **component manifest** in meta (name -> file -> functions -> calls per token) —
  today the runtime hardcodes embeddings / FFN / LM head
  (`tests/chat.py:1739-1775`);
- **explicit layer ranges** as CLI parts (8.1);
- **per-layer-type rotate** (5.3, 8.7);
- **attention tiling + log-sum-exp merge** for a 24,576-row state if placement
  fails (8.5);
- **loop-count invalidation** of the cache (spec D4 / user story 10) — ANEMLL has
  no cache-reset API beyond allocating a fresh `MLState`; note that
  `state_transition.py` (`transition_kv_state:62`, `compact_kv_state:153`,
  `transition_coreml_state:590`) is the only state-rewriting utility, and it is
  about context-length migration, not reset.

---

## Appendix A — the ten files to read before writing code

1. `anemll/models/gemma3_model.py:767-1208` — states, split caches, rotate, masks.
2. `anemll/ane_converter/gemma3_converter.py:289-395` — union state declaration.
3. `anemll/ane_converter/gemma3_converter.py:1406-1535` — infer/rotate trace.
4. `anemll/ane_converter/gemma3_converter.py:1546-1760` — prefill(+rotate) trace.
5. `anemll/models/llama_model.py:1139-1419` — head split + forward contract.
6. `anemll/ane_converter/llama_converter.py:751-972` — the canonical part tracers.
7. `anemll/utils/combine_models.py:23-52, 346-453` — multifunction assembly.
8. `anemll/utils/generate_meta_yaml.py:398-509` — meta schema.
9. `tests/chat.py:977-1350` — masks, prefill, decode, unified state.
10. `anemll/utils/convert_model.sh:300-930` — the real orchestration order.

## Appendix B — glossary of ANEMLL-specific names

- **part** — a converter output: `1` embeddings, `2` FFN/infer, `2_prefill`,
  `3` LM head (`metadata.py:16-21`).
- **chunk** — a contiguous layer range of part 2, with its own files.
- **function** — a CoreML function inside a multifunction `.mlpackage`
  (`infer`, `prefill`, `infer_rotate`, `prefill_rotate`).
- **state** — `ct.StateType` / `MLState`; the KV cache. Names are attribute paths.
- **update_mask** — `(1,1,mask_len,batch)` fp16 one-hot selecting write rows.
- **attention_size** — mask width / global-cache read width (Gemma 3 only).
- **sliding_window** — local cache length, and the position at which the runtime
  switches to the `_rotate` functions.
