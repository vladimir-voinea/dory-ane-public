"""All ANE-safe Dory gates in ONE process, so the 8 GB of weights stay hot.

Loading the model once per gate pays the load and the page-in cost five times
over; on this shared machine that is what made a decode step cost 8.6 s.
Gates are run cheapest first and every number is written next to the row range,
the wall time and the load average that produced it.

    python scripts/dory_ane_gate_suite.py --rows-mode window --window-rows 256
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from anemll.eval import load_eval_set, write_candidate
from anemll.eval.kld import LogprobSource, read_candidate_npz, read_llama_cpp_kld, score_sources
from anemll.models.dory_ane_model import DoryANEConfig, DoryANEForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dory_ane_score import DEFAULT_WEIGHTS, log_softmax  # noqa: E402

REFS = os.path.expanduser("~/dev/dory-ane-artifacts/refs")
ARTIFACTS = os.path.expanduser("~/dev/dory-ane-artifacts/DANE-4")


def loadavg():
    return os.getloadavg()[0]


def gate_rows(mode, window):
    """(chunk, first_row, n_rows) for every context this mode scores."""
    if mode == "window":
        return [(0, 0, window)]
    return [(0, 0, 2047), (1, 0, 2047)]


def run_prefill_gate(model, evalset, rows_spec, batch):
    """Prefill in fixed batches; keep log-probabilities only for the gate rows."""
    rows, positions, logprobs = 0, [], []
    t0 = time.time()
    for chunk, first_row, n_rows in rows_spec:
        base = chunk * evalset.chunk_size
        first_scored = evalset.first_scored
        want_start = first_scored + first_row
        want_stop = want_start + n_rows
        # the reference processed full 64-token ubatches; round the prefix up so
        # the scored rows sit in the same ubatch boundaries as llama.cpp's
        limit = min(((want_stop + batch - 1) // batch) * batch, evalset.chunk_size)
        ids = evalset.token_ids[base:base + limit]
        model.reset_caches()
        out = []
        pos = 0
        while pos < len(ids):
            take = min(batch, len(ids) - pos)
            if take < batch:
                for step in range(take):
                    lg = model.decode_step(int(ids[pos + step]), pos + step)
                    if want_start <= pos + step < want_stop:
                        out.append(log_softmax(lg[0, 0]))
                pos = len(ids)
                break
            lg = model.prefill(torch.tensor(ids[pos:pos + take]), pos)
            for i in range(take):
                if want_start <= pos + i < want_stop:
                    out.append(log_softmax(lg[0, i]))
            pos += take
            if (pos // take) % 16 == 0:
                print(f"    prefill {pos}/{len(ids)} tokens, {time.time() - t0:.0f}s",
                      flush=True)
        out = np.stack(out)
        rows += out.shape[0]
        logprobs.append(out)
        positions.append(evalset.scored_positions()[chunk * evalset.rows_per_chunk + first_row:
                                                   chunk * evalset.rows_per_chunk + first_row + n_rows])
    return np.concatenate(logprobs), np.concatenate(positions), rows, time.time() - t0


def run_decode_gate(model, evalset, rows_spec):
    rows, positions, logprobs = 0, [], []
    t0 = time.time()
    for chunk, first_row, n_rows in rows_spec:
        base = chunk * evalset.chunk_size
        first_scored = evalset.first_scored
        limit = first_scored + first_row + n_rows
        ids = evalset.token_ids[base:base + limit]
        model.reset_caches()
        out = []
        for pos, tok in enumerate(ids):
            lg = model.decode_step(int(tok), pos)
            if pos >= first_scored + first_row:
                out.append(log_softmax(lg[0, 0]))
            if pos and pos % 256 == 0:
                print(f"    decode {pos}/{len(ids)} tokens, {time.time() - t0:.0f}s", flush=True)
        out = np.stack(out)
        rows += n_rows
        logprobs.append(out)
        positions.append(evalset.scored_positions()[chunk * evalset.rows_per_chunk + first_row:
                                                   chunk * evalset.rows_per_chunk + first_row + n_rows])
    return np.concatenate(logprobs), np.concatenate(positions), rows, time.time() - t0


def score(logprobs, positions, tokens, ref_path, name, out_dir):
    ref = read_llama_cpp_kld(ref_path)
    # slice the reference to exactly the rows the candidate produced
    want = positions
    idx = np.searchsorted(ref.positions, want)
    if idx.size == 0 or not np.array_equal(ref.positions[idx], want):
        raise SystemExit(f"{name}: candidate positions are not a contiguous row range "
                         f"of {ref_path}")
    start = int(idx[0])
    stop = start + len(want)
    sliced = LogprobSource(f"{ref.name}[{start}:{stop}]", len(want), ref.vocab_size,
                           ref.tokens, ref.positions[start:stop], ref.meta,
                           lambda a, b: ref.logp_block(start + a, start + b))
    cand_path = os.path.join(out_dir, f"cand_{name}.npz")
    write_candidate(cand_path, logprobs, positions, tokens,
                    meta={"model": "dory-ane-fp16", "kv_mode": "last_loop", "gate": name},
                    dtype="float16")
    cand = read_candidate_npz(cand_path)
    return score_sources(sliced, cand)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--rows-mode", choices=("window", "full"), default="window",
                    help="default rows mode; a gate may override it as name:mode")
    ap.add_argument("--window-rows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gates", default="prefill2,prefill1,prefill4,decode2,decode1,decode4")
    ap.add_argument("--out-json", default=os.path.join(ARTIFACTS, "gate_suite.json"))
    ap.add_argument("--artifacts", default=ARTIFACTS)
    args = ap.parse_args(argv)

    torch.set_num_threads(args.threads)
    os.makedirs(args.artifacts, exist_ok=True)
    evalset = load_eval_set()
    tokens = np.asarray(evalset.token_ids, dtype=np.int32)
    rows_spec = gate_rows(args.rows_mode, args.window_rows)

    cfg = DoryANEConfig(context_length=evalset.chunk_size, batch_size=args.batch,
                        fold_scales=True, dtype=torch.float16, compute_mode="matmul")
    model = DoryANEForCausalLM(cfg)
    t0 = time.time()
    report = model.load_pretrained_weights(args.weights)
    print(f"[{time.strftime('%H:%M:%S')}] loaded {report['consumed']} tensors in "
          f"{time.time() - t0:.0f}s; loadavg={loadavg():.0f}", flush=True)

    wanted = [g.strip() for g in args.gates.split(",") if g.strip()]
    results = []
    for spec in wanted:
        name, _, mode = spec.partition(":")
        mode = mode or args.rows_mode
        this_spec = gate_rows(mode, args.window_rows)
        kind = "prefill" if name.startswith("prefill") else "decode"
        loops = int(name[-1])
        model.set_loops(loops)
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] gate {name} rows_mode={mode} "
              f"threads={args.threads} loadavg={loadavg():.0f}", flush=True)
        if kind == "prefill":
            lp, pos, n, secs = run_prefill_gate(model, evalset, this_spec, args.batch)
        else:
            lp, pos, n, secs = run_decode_gate(model, evalset, this_spec)
        ref_name = (f"bf16_decode_loop{loops}_batch{args.batch if kind == 'prefill' else 1}.kld")
        ref_path = os.path.join(REFS, ref_name)
        entry = {"gate": name, "kind": kind, "loops": loops, "rows": int(n),
                 "row_range": f"{int(pos[0])}..{int(pos[-1])}", "wall_s": round(secs, 1),
                 "loadavg": round(loadavg(), 1), "reference": ref_name,
                 "threads": args.threads, "rows_mode": mode}
        if os.path.exists(ref_path):
            res = score(lp, pos, tokens, ref_path, name, args.artifacts)
            entry.update({"mean_kld": res.mean_kld, "top1": res.top1_agreement,
                          "kld_max": res.kld_max, "kld_p90": res.kld_p90})
            print(f"[{time.strftime('%H:%M:%S')}] {name}: rows={n} {entry['row_range']} "
                  f"mean_kld={res.mean_kld:.5f} top1={100 * res.top1_agreement:.2f}% "
                  f"({secs:.0f}s, loadavg {entry['loadavg']:.0f})", flush=True)
        else:
            entry["skipped"] = f"missing reference {ref_name}"
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {entry['skipped']}", flush=True)
        results.append(entry)
        with open(args.out_json, "w") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
