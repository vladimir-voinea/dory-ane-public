# Dory1.0-4B on the Apple Neural Engine

Dory is NVIDIA's looped (latent-recurrent) normalized-GPT: 72 physical blocks
that the host runs **N times** (N = 1-4) for every token, with interleaved
3:1 sliding (window 1024) and full attention.  This document describes the
classic-ANEMLL port (spec D1-D11): how the five Core ML components are built,
what the meta file records, and how to run and measure them.

- Converter: `anemll/ane_converter/dory_converter.py`, one command `scripts/convert_dory.py`
- Runtime: `anemll/dory_chat.py`
- Traceable chunk wrappers: `anemll/models/dory_chunks.py`
- Math and caches: `anemll/models/dory_ane_model.py`, `anemll/utils/dory_caches.py` (ticket DANE-4)

## The five components (spec D6)

| Component | Blocks | Called per token | File |
|---|---|---|---|
| embeddings | token embedding | once | `dory_embeddings[_lutN].mlmodelc` |
| input | 0-15 | once | `dory_input_lut6.mlmodelc` |
| recurrent | 16-55 | **N times** (loop count) | `dory_recurrent_lut6.mlmodelc` |
| output | 56-71 | once | `dory_output_lut6.mlmodelc` |
| LM head | `lm_head`, split into 8 x 16,384 | once | `dory_lm_head_lut6.mlmodelc` |

Each file is multifunction: `infer` (one token) and `prefill` (64 tokens),
built as two traces of the same weights and welded with
`MultiFunctionDescriptor` plus ANEMLL's weight dedup, so the two functions share
one copy of the weights.

The block ranges come from Dory's structure, not from ANEMLL's balanced split
(`llama_converter.py:761-768`): an even split would cut through the recurrent
zone and give a chunk a different call multiplicity from its neighbours.
`--recurrent-splits 2` turns the 40-block recurrent component into two files, each
still called once per loop, with the loop entry norm on the first and the loop
exit norm on the last -- the escape hatch when the recurrent component does not
fit in one 2 GB file.

## Conversion

    python scripts/convert_dory.py --model <bf16-dir> --output <dir>

The BF16 directory is NVIDIA's `NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16` (or its
Hugging Face cache snapshot, which is the default).  Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--context-length` | 1024 | rows of a full-attention cache; 24576 is the spec target |
| `--batch-size` | 64 | the prefill batch, a compile-time constant |
| `--lut-blocks` / `--lut-lm-head` | 6 / 6 | LUT palettization bits (spec D8) |
| `--lut-embeddings` | none | spec D8: unquantized or 8 |
| `--lm-head-splits` | 8 | even parts, 16,384 channels each |
| `--recurrent-splits` | 1 | 2 when the recurrent block exceeds 2 GB |
| `--argmax` | off | return local argmax indices from the LM head |
| `--only` | all | convert one component (a failed chunk does not cost the run) |
| `--dry-run` | off | print the plan, the size estimates and the disk check |

Before converting anything the command prints the plan and refuses to start if a
component would exceed **2 GB** (the per-file limit the Neural Engine imposes)
or if the volume would drop below 2 GB free.  After converting it checks every
function of every file against Core ML's compute plan and fails the run if any
op is not on the Neural Engine (spec gate: GPU: 0 and CPU: 0).
`conversion_report.json` records the sizes, the timings and the per-function
placement counts.

## What the meta file records

ANEMLL 0.3.5 records the context length and a chunk count; Dory's runtime needs
more, so `meta.yaml` adds (with `dory_meta: 1` as the schema version):

- `kv.mode: last_loop`, `kv.invalidate_on_loop_change: true`, and the two state
  declarations with their shapes and byte cost (9 full-attention caches of
  `context_length` rows, 27 sliding caches of
  `PAD(min(ctx, window + batch), 256)` rows);
- `loops: {min: 1, max: 4, default: 2}` -- the supported loop range;
- `components.<name>`: file, functions, block range, `calls_per_token` (`1` or
  `loops`), LUT width, inputs and outputs -- so the runtime does not hardcode five
  paths;
- `tokenizer`: the tokenizer file, the chat template (or
  `chat_template_fallback`), the reasoning default and the EOS ids;
- `parameters.zones`, `sliding_rows`, `mask_len` and the attention-block
  inventory, which the runtime needs to build the mask.

`validate_meta()` checks all of it on write and on load.

## The state union (the one thing that silently breaks)

Every component declares **both** cache buffers, even the ones that never touch
a cache (embeddings, LM head), and adds `0 * state[0,0,0,0]` into its output so
Core ML does not prune the unused one.  If the declarations differ, Core ML
silently gives each chunk its own state and the input block's cache writes never
reach the recurrent block -- a correctness bug with no error message.  The trick
is Gemma 3's (`gemma3_converter.py:341-371`, `:1474-1481`).

## Running

    # chat, reasoning on by default, loop count chosen at start
    python -m anemll.dory_chat chat --model-dir <converted-dir> --loops 2

    # one prompt
    python -m anemll.dory_chat chat --model-dir <dir> --prompt "Why is the sky blue?"

    # decode speed at 1K context for loops 1, 2 and 4
    python -m anemll.dory_chat bench --model-dir <dir> --context 1024 --loops 1,2,4

    # the Core ML pipeline scored at DANE-2's seam, in decode mode
    python -m anemll.dory_chat score --model-dir <dir> --chunk-size 1024 \
        --loops 2 --out cand.npz --reference <decode reference>.kld

Mid-session commands: `/loops N` (resets the KV cache and says so -- older rows
keep the *final* loop's K/V, so a change must start from an empty cache),
`/reset`, `/quit`.

Generation stops on EOS (tokens 11 and 2, per the repository's
`generation_config.json`).

### Chat template

The BF16 checkpoint ships the special tokens (`<|im_start|>`, `<|im_end|>`,
`</think@@`, `[INST]`, ...) but **no chat template**: none in
`tokenizer_config.json`, none in the community GGUF metadata, while NVIDIA's README
launches llama.cpp with `--jinja --reasoning on`.  The converter copies a
`chat_template.jinja` when the weights have one (including one embedded in
`tokenizer_config.json`) and the runtime prefers it; otherwise it renders the
documented ChatML fallback with an open think block, which is what the vocabulary
implies.  The choice is recorded in `meta.yaml` (`tokenizer.chat_template` or
`tokenizer.chat_template_fallback`), never hidden in code.

## Measuring

- **Placement**: the converter runs the check itself; to repeat it,
  `python anemll/utils/ane_profiler.py --model <file>.mlpackage --analyze` must
  print `GPU: 0` and `CPU: 0`.  Stateful models cannot use `--benchmark` /
  `--test` (random inputs carry no `MLState`); `--analyze`, `--compare` and
  `--costs` work.
- **Quality**: `score` writes a candidate in DANE-2's `.npz` format and scores it
  with `anemll.eval.kld`; the gate is mean KLD <= 0.05 and top-1 >= 93% at 2
  loops in decode mode.
- **Speed**: `bench` reports decode tok/s per loop count.  Timings are only
  meaningful on a quiet machine -- record `uptime` beside them.

## Ticket evidence (DANE-5 acceptance criteria)

Measured numbers live here; anything not yet measured says so.  Timings are
only comparable on a quiet machine, so each one is recorded with the load
average it was taken under.

| Criterion | Status | Evidence |
|---|---|---|
| One command produces the chunks and meta, and stops with a clear message if a file would exceed ~2 GB | **implemented and unit-tested**, conversion not yet run | `scripts/convert_dory.py`; estimates at 1K context and LUT6: embeddings 653 MB, input 735 MB, recurrent 1837 MB (91 % of the 2 GB limit), output 735 MB, LM head 336 MB. The guard refuses a LUT8 recurrent block (2.29 GB) with a message naming `--recurrent-splits 2` and `--lut-blocks 4` |
| Every op of every chunk on the ANE, converter checks itself | **implemented**, not yet measured on the real chunks | `placement_report()` walks every operation of every function with `MLComputePlan` after `xcrun coremlcompiler`; `assert_ane_only()` fails the run and names the offending ops. Trace + `ct.convert` with both states was verified on a small module (conversion OK, package compiles); the real-dimension run is blocked by machine memory |
| Mean KLD <= 0.05, top-1 >= 93 % at 2 loops, decode mode, 1K windows | **not yet measured - machine blocked** | `python -m anemll.dory_chat score` writes DANE-2's candidate format and calls `anemll.eval.kld.score_files`; the 1K-window decode reference still has to be built (command below) |
| Chat streams a coherent answer with reasoning on at loops 1, 2 and 4; a mid-session loop change resets the cache and says so | **runtime implemented and unit-tested**, no real answer yet | `/loops N` resets the shared `MLState` and prints why; reasoning on by default through the chat template; generation stops on EOS 11 and 2 |
| First-load ANE compile cached, second start measurably faster | **not yet measured - machine blocked** | the converter emits `.mlmodelc` next to `.mlpackage` and the runtime prefers the compiled form; `bench` reports `load_seconds` |
| Decode tok/s at 1K for loops 1, 2 and 4 | **not yet measured - machine blocked** | `python -m anemll.dory_chat bench --context 1024 --loops 1,2,4` |

Facts established while building it (worth knowing before the first real run):

- a 16-block chunk with both cache states traces with `torch.jit.trace` (the
  ring write `cache[slot:slot+1, :, cache_pos:cache_pos+seq, :] = k` survives
  tracing) and converts with `ct.convert(..., states=..., FLOAT16,
  CPU_AND_NE, iOS18, mlprogram)`;
- coremltools 8.3 renames state names (`model.kv_cache_full` ->
  `model_kv_cache_full`, dots are not allowed), consistently across components,
  so the union rule is satisfied by building every component the same way;
- the state union at 1K context is 36 MiB of full caches plus 108 MiB of
  sliding caches, per `meta.yaml`;
- the BF16 repository has **no chat template** (see above), so the runtime
  records and uses a documented ChatML fallback with an open think block.

## Gate commands (run when the machine is quiet)

    # 0. reference: 1K-window decode mode at 2 loops (BF16 GGUF, one token per
    #    step would be batch 1; the Core ML pipeline prefills in 64-token
    #    batches, so the matching reference is batch 64)
    python scripts/make_dory_references.py all \
        --llama-perplexity ~/.unsloth/llama.cpp-dory/build/bin/llama-perplexity \
        --bf16-gguf <BF16.gguf> --out-dir ~/dev/dory-ane-artifacts/DANE-5/refs-1k \
        --ctx 1024 --chunks 8 --only bf16_decode_loop2_batch64 --skip-q8

    # 1. conversion (dry run first: plan, estimates, disk check, no output)
    python scripts/convert_dory.py --context-length 1024 --dry-run
    python scripts/convert_dory.py --context-length 1024

    # 2. placement, independently of the converter's own gate
    for f in ~/dev/dory-ane-artifacts/DANE-5/1024ctx/*.mlpackage; do
        python anemll/utils/ane_profiler.py --model "$f" --analyze | tail -20
    done

    # 3. quality: KLD and top-1 at 2 loops, decode mode, 1K windows
    python -m anemll.dory_chat score \
        --model-dir ~/dev/dory-ane-artifacts/DANE-5/1024ctx \
        --chunk-size 1024 --loops 2 \
        --out ~/dev/dory-ane-artifacts/DANE-5/cand-coremll-lut6-loop2-1k.npz \
        --reference ~/dev/dory-ane-artifacts/DANE-5/refs-1k/bf16_decode_loop2_batch64.kld

    # 4. chat at loops 1, 2 and 4, reasoning on
    for n in 1 2 4; do
        python -m anemll.dory_chat chat --model-dir ~/dev/dory-ane-artifacts/DANE-5/1024ctx \
            --loops $n --prompt "Why is the sky blue?" --max-tokens 200
    done

    # 5. decode speed at 1K (record uptime beside it)
    python -m anemll.dory_chat bench --model-dir ~/dev/dory-ane-artifacts/DANE-5/1024ctx \
        --context 1024 --loops 1,2,4 --tokens 32

    # 6. compile cache: first start vs second start of the same runtime
    python - <<'PY'
    import time, json
    from anemll.dory_chat import DoryRuntime
    d = "~/dev/dory-ane-artifacts/DANE-5/1024ctx"
    for i in (1, 2):
        t = time.time(); r = DoryRuntime(d, verbose=False); print(i, "load", round(time.time()-t, 2), "s")
    PY
