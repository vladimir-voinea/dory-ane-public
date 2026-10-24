# The shipping configuration — what the Neural Engine costs and buys

This page is the DANE-8 verdict: one table that compares the LUT widths on the
Neural Engine, the same workload on the GPU llama.cpp build, and the
configuration this port ships.  Everything below the marker is **generated** by
`scripts/dory_sweep.py report` from measured cells; the pages above and below it
are the method and the decision rule.

## The rule

A configuration ships when it clears every spec gate at every loop count:

| gate | threshold | where it comes from |
|---|---|---|
| mean KLD vs the bf16 decode reference | <= 0.05 | spec, Testing Decisions |
| top-1 agreement | >= 93 % | spec, Testing Decisions |
| decode speed at 24,576 tokens / 2 loops | >= 8 tok/s | spec, D11 |
| the reassess floor | below 6 tok/s, stop and reassess | spec, D11 |

Among the configurations that clear all four, the **fastest at 24K / 2 loops**
ships; ties go to the better worst-case top-1, then to the wider LUT.  The rule
is implemented in one place (`choose_shipping`) and it reads only measured
cells — the same store that produces the table produces the choice.

**Nothing is interpolated.**  If a cell has no measurement the report refuses to
render and prints the command that fills it.  A missing cell is never replaced by
a neighbouring value, an estimate, or "about the same".

## What each number is, and how it was measured

| column | measurement |
|---|---|
| on disk | the bytes of the files the meta names, summed per component (`sizes`) |
| mean KLD, top-1 | the Core ML pipeline's log-probabilities on the fixed evaluation set, scored against DANE-2's bf16 decode reference with `anemll.eval.kld.score_files` — the same scorer every ticket uses, never a second implementation (`score`) |
| decode tok/s | the runtime's own decode step, warm window (the first quarter of the tokens dropped), one 24K-capable session per cell (`speed`) |
| prefill tok/s | the same prompt fed in the runtime's 64-token batches, timed once (`speed`) |
| peak RSS | `ru_maxrss` of the measuring process, i.e. the model's own footprint (`speed`) |
| GPU rows | `llama-server` on the Q8_0 GGUF with `--override-kv dory.num_loops=int:<n>,dory.recurrent_kv_cache_mode=str:last_loop`; one non-streaming `/completion`, its `timings` are the row, `psutil` gives the RSS (`gpu`) |
| energy per token | `powermetrics` samples (ANE Power / GPU Power, mW) x the workload's wall time / generated tokens (`energy`) |

Protocol, so the two builds are comparable: the same fixed evaluation text, the
same target token counts (2048, 8192, 16384, 24576), the same 32 decoded tokens
per cell, both at 2 loops for the head-to-head and at 1/2/4 loops for the
matrices.  Each recorded cell stores the command, the load average, the host and
the sha256 of the artifact it came from, so any number on this page can be traced
to a run.

The Neural Engine numbers are only meaningful on a quiet machine: stateful Core
ML loads fail with error -14 under load, and memory pressure distorts both RSS
and tok/s.  Studio's models must be unloaded before a sweep; the numbers below
say which load average they were taken at.

Disk discipline: the LUT widths do not fit side by side.  Convert one
configuration, record `sizes` and `score` for it, delete the packages, check
`df -h` before the next one.  The speed matrix runs on the shipping
configuration, so only it needs to stay on disk while it is measured.

## How to run the sweep

    python scripts/dory_sweep.py plan                       # what is missing, and the command for each cell
    python scripts/dory_sweep.py sizes  --config lut6 --model-dir <dir>
    python scripts/dory_sweep.py score  --config lut6 --loops 2 --model-dir <dir> \
        --reference ~/dev/dory-ane-artifacts/refs/bf16_decode_loop2_batch64.kld
    python scripts/dory_sweep.py speed  --config lut6 --model-dir <dir>
    python scripts/dory_sweep.py gpu    --gguf <Q8_0.gguf>
    python scripts/dory_sweep.py energy-plan                # the sudo commands for the owner
    python scripts/dory_sweep.py report                     # writes this page's table

## Energy: the owner's one sudo step

`powermetrics` needs root, so that part is the owner's.  The harness prints the
exact commands (`energy-plan`); the shape of it is:

    sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 500 -o ane-24k-2loops.txt &
    PM=$!
    python -m anemll.dory_chat bench --model-dir <dir> --context 24576 --loops 2 --tokens 64
    kill -INT $PM; wait $PM
    python scripts/dory_sweep.py energy --build ane --log ane-24k-2loops.txt \
        --tokens 64 --seconds <bench seconds>

and the same with `gpu` for the Q8_0 build.  What comes back: the powermetrics
log and the bench JSON; the log's samples times the run's wall time divided by
the generated-token count is the joules per token this page reports.  Until that
run happens the energy row says so, and no other row is affected — energy is a
measurement, not one of the gates above.

<!-- BEGIN SWEEP TABLE -->

Not measured yet.  `python scripts/dory_sweep.py report` replaces this block
with the table once every required cell has a measurement; it refuses to write
anything while one is missing, so an empty block here means the sweep has not
run against a converted model yet.

<!-- END SWEEP TABLE -->

## What the choice means in practice

The shipping configuration is baked into three places and they must agree:
the converter's defaults (`scripts/convert_dory.py --lut-blocks/--lut-lm-head`),
the meta file the conversion writes (`parameters.lut_blocks`, `parameters.lut_lm_head`),
and the server that serves it (`GET /v1/models` reports the model id and the loop
range; the loop count stays a per-request choice at 1–4 with 2 as the default).
Changing the sweep's verdict means changing all three, and re-running the score
and speed cells for the new configuration.
