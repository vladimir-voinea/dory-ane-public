#!/usr/bin/env python3
"""Build the Dory reference log-probability artifacts, one command.

Every later ticket is scored against these files, so they are built by
llama.cpp (PR #30306) running Dory's BF16 GGUF, exactly the reference the
published KL table uses:

    exact   dory.recurrent_kv_cache_mode=per_loop   -b 4096 -ub 4096
            (per_loop keeps a KV cache per loop, and one ubatch per context is
            a full recompute: splitting the context into ubatches makes the
            logits batch-dependent once the sliding window is active, so the
            ubatch must cover the whole context - see ubatch_for())
    decode  dory.recurrent_kv_cache_mode=last_loop  -b N -ub N
            (one KV cache per recurrent layer, overwritten each loop, so the
            result depends on the batching the candidate actually uses)

Both modes score the second half of each 4096-token context, which is what
llama-perplexity does; see docs/dory-eval.md.

usage:
    python scripts/make_dory_references.py all \
        --llama-perplexity ~/.unsloth/llama.cpp-dory/build/bin/llama-perplexity \
        --bf16-gguf  ~/.cache/huggingface/hub/.../Dory1.0-4B-BF16.gguf

Artifacts are written outside the repository, to --out-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from anemll.eval.eval_set import FIXTURE_DIR, TEXT_FILE, TOKENS_FILE, load_eval_set  # noqa: E402
from anemll.eval.kld import read_llama_cpp_kld, score_files  # noqa: E402

DEFAULT_OUT = Path.home() / "dev" / "dory-ane-artifacts" / "refs"
CTX = 4096
CHUNKS = 2

# name -> (mode, loops, n_batch)
# Decode references exist at every loop count the tickets report (1, 2, 4), at
# both batchings a candidate runs: one token per decode step and 64-token
# prefill.  Each is scored against the exact reference with the SAME loop count.
REFERENCES = {
    "bf16_exact_loop1": ("exact", 1, 4096),
    "bf16_exact_loop2": ("exact", 2, 4096),
    "bf16_exact_loop4": ("exact", 4, 4096),
    "bf16_decode_loop1_batch1": ("decode", 1, 1),
    "bf16_decode_loop1_batch64": ("decode", 1, 64),
    "bf16_decode_loop2_batch1": ("decode", 2, 1),
    "bf16_decode_loop2_batch64": ("decode", 2, 64),
    "bf16_decode_loop4_batch1": ("decode", 4, 1),
    "bf16_decode_loop4_batch64": ("decode", 4, 64),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def human(size: int) -> str:
    return f"{size / 1e9:.2f} GB" if size >= 1e9 else f"{size / 1e6:.1f} MB"


def ubatch_for(mode: str, n_batch: int, ctx: int) -> int:
    """The -ub that makes each mode mean what it says.

    exact: ONE ubatch covering the whole context (-b ctx -ub ctx). A chunk cut
    into several ubatches is not a full recompute, because llama.cpp's sliding
    window (1024 tokens here) masks attention per ubatch. Measured on the BF16
    GGUF, per_loop, -c 4096 --chunks 1, loops 1, PPL by -ub:

        64 / 256  4.9790 (bit-identical to each other)
        512       4.9610
        1024      4.9623
        2048      4.9662
        4096      4.9655

    and at loops 2, -ub 1024 against -ub 4096 (both dumps, same text) reads
    mean KLD 0.0076 / top-1 99.41 % / max 10.7 - the same magnitude as the
    gates this reference is used to measure. Below ctx 1024 every ubatch
    setting is bit-identical, so the split only starts to matter exactly when
    the window begins masking; references built with -ub < ctx must not gate.

    decode: the candidate's own batching is the whole point of the mode, so
    -ub follows -b (-b 1 -ub 1 for decode steps, -b 64 -ub 64 for prefill).
    """
    return max(n_batch, 1) if mode == "decode" else ctx


def run_llama_perplexity(
    binary: Path,
    gguf: Path,
    text: Path,
    out: Path,
    mode: str,
    loops: int,
    n_batch: int,
    ctx: int = CTX,
    chunks: int = CHUNKS,
) -> dict:
    """Run llama-perplexity and leave out (.kld) plus out.meta.json behind."""
    kv_mode = "per_loop" if mode == "exact" else "last_loop"
    n_ubatch = ubatch_for(mode, n_batch, ctx)
    cmd = [
        str(binary),
        "-m", str(gguf),
        "-f", str(text),
        "-c", str(ctx),
        "-b", str(n_batch),
        "-ub", str(n_ubatch),
        "--chunks", str(chunks),
        "-ngl", "999",
        "-fa", "on",  # pinned: auto resolves to on here, off shifts the numbers
        "--override-kv", f"dory.num_loops=int:{loops},dory.recurrent_kv_cache_mode=str:{kv_mode}",
        "--kl-divergence-base", str(out),
    ]
    started = time.time()
    print(
        f"[{out.name}] {mode} loops={loops} kv={kv_mode} "
        f"batch={n_batch} ubatch={n_ubatch} ...",
        flush=True,
    )
    log = out.with_suffix(out.suffix + ".log")
    with open(log, "w") as fh:
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=False)
    duration = time.time() - started
    if proc.returncode != 0:
        tail = "\n".join(log.read_text().splitlines()[-15:])
        raise SystemExit(f"{out.name}: llama-perplexity exited {proc.returncode}\n{tail}")
    ppl = ""
    for line in reversed(log.read_text().splitlines()):
        if "Final estimate: PPL" in line:
            ppl = line.split("PPL =")[-1].split("+/-")[0].strip()
            break
    print(f"[{out.name}] done in {duration:.1f}s, PPL={ppl}, {human(out.stat().st_size)}", flush=True)
    return {"cmd": cmd, "duration_s": round(duration, 1), "ppl": ppl, "log": str(log)}


def verify_tokens(out: Path, eval_set) -> None:
    """The dump must be on exactly the committed evaluation ids."""
    source = read_llama_cpp_kld(out)
    expected = load_eval_set(FIXTURE_DIR).token_ids
    if source.tokens.shape != expected.shape or not np.array_equal(source.tokens, expected):
        raise SystemExit(
            f"{out.name}: llama.cpp tokenised the evaluation text differently from "
            f"{TOKENS_FILE} (got {source.tokens.shape[0]} ids, expected {expected.shape[0]})"
        )
    if source.n_rows != eval_set.n_rows:
        raise SystemExit(
            f"{out.name}: {source.n_rows} scored rows, expected {eval_set.n_rows}"
        )


def build_reference(name, spec, args, eval_set, gguf_sha) -> Path:
    mode, loops, n_batch = spec
    out = args.out_dir / f"{name}.kld"
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    if out.is_file() and meta_path.is_file() and not args.force:
        print(f"[{name}] exists, skipping (use --force to rebuild)", flush=True)
        return out
    info = run_llama_perplexity(
        args.llama_perplexity, args.bf16_gguf, args.eval_text, out, mode, loops, n_batch,
        ctx=args.ctx, chunks=args.chunks,
    )
    verify_tokens(out, eval_set)
    meta = {
        "name": name,
        "mode": mode,
        "loops": loops,
        "kv_cache_mode": "per_loop" if mode == "exact" else "last_loop",
        "n_batch": n_batch,
        "n_ubatch": ubatch_for(mode, n_batch, args.ctx),
        "chunk_size": args.ctx,
        "n_chunks": args.chunks,
        "rows_per_chunk": args.ctx - 1 - args.ctx // 2,
        "n_rows": eval_set.n_rows,
        "eval_set": {
            "dir": str(FIXTURE_DIR),
            "tokens_sha256": sha256(FIXTURE_DIR / TOKENS_FILE),
            "text_sha256": sha256(FIXTURE_DIR / TEXT_FILE),
        },
        "model": {
            "gguf": str(args.bf16_gguf),
            "sha256": gguf_sha,
        },
        "tool": {
            "llama_perplexity": str(args.llama_perplexity),
            "command": info["cmd"],
        },
        "perplexity": info["ppl"],
        "seconds": info["duration_s"],
        "bytes": out.stat().st_size,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    if not args.keep_log:
        Path(info["log"]).unlink(missing_ok=True)
    return out


def q8_proof(args, eval_set, gguf_sha) -> dict:
    """Score the Q8_0 GGUF against the BF16 exact-mode reference at 2 loops."""
    reference = args.out_dir / "bf16_exact_loop2.kld"
    if not reference.is_file():
        print(f"[q8-proof] {reference.name} missing, skipping", flush=True)
        return {}
    candidate = args.out_dir / "q8_0_exact_loop2.kld"
    meta_path = candidate.with_suffix(candidate.suffix + ".meta.json")
    if not candidate.is_file() or args.force:
        info = run_llama_perplexity(
            args.llama_perplexity, args.q8_gguf, args.eval_text, candidate, "exact", 2, 4096,
            ctx=args.ctx, chunks=args.chunks,
        )
        verify_tokens(candidate, eval_set)
        meta = {
            "name": candidate.stem,
            "role": "candidate",
            "mode": "exact",
            "loops": 2,
            "model": {"gguf": str(args.q8_gguf), "sha256": sha256(args.q8_gguf)},
            "command": info["cmd"],
            "perplexity": info["ppl"],
            "seconds": info["duration_s"],
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    result = score_files(reference, candidate)
    payload = result.as_dict()
    scores = args.out_dir / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    (scores / "q8_0_exact_loop2_vs_bf16_exact_loop2.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(result.summary(), flush=True)
    if args.delete_candidates:
        candidate.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    return payload


def gap_report(args) -> dict:
    """Score every decode-mode reference against its exact one: the last_loop floor.

    Each decode reference is scored against the exact reference with the SAME
    loop count - that is the model the candidate is trying to reproduce - so the
    floor is reported per loop count and per batching.
    """
    out = {}
    scores = args.out_dir / "scores"
    scores.mkdir(parents=True, exist_ok=True)
    for name, (mode, loops, n_batch) in REFERENCES.items():
        if mode != "decode":
            continue
        exact = args.out_dir / f"bf16_exact_loop{loops}.kld"
        path = args.out_dir / f"{name}.kld"
        if not exact.is_file():
            print(f"[gap] {exact.name} missing, skipping {name}", flush=True)
            continue
        if not path.is_file():
            print(f"[gap] {name} missing, skipping", flush=True)
            continue
        result = score_files(exact, path)
        payload = result.as_dict()
        payload["note"] = (
            "exact mode with the same loop count is the reference here: this number is "
            "the floor any last_loop candidate can reach against a per_loop reference"
        )
        payload["loops"] = loops
        payload["n_batch"] = n_batch
        payload["exact_reference"] = str(exact)
        out[name] = payload
        print(
            f"[gap] {name}: mean KLD {result.mean_kld:.6f}, "
            f"top-1 {100 * result.top1_agreement:.3f}% vs {exact.name}",
            flush=True,
        )
    if out:
        (scores / "decode_gap.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the BF16 reference log-probability artifacts for Dory.",
    )
    parser.add_argument("--llama-perplexity", type=Path, required=True)
    parser.add_argument("--bf16-gguf", type=Path, required=True)
    parser.add_argument("--q8-gguf", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--eval-text", type=Path, default=FIXTURE_DIR / TEXT_FILE)
    parser.add_argument("--ctx", type=int, default=CTX)
    parser.add_argument("--chunks", type=int, default=CHUNKS)
    parser.add_argument("--only", nargs="*", default=None, help="subset of reference names")
    parser.add_argument("--force", action="store_true", help="rebuild artifacts that exist")
    parser.add_argument("--keep-log", action="store_true")
    parser.add_argument("--skip-q8", action="store_true", help="do not run the Q8_0 proof")
    parser.add_argument("--delete-candidates", action="store_true")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_set = load_eval_set(FIXTURE_DIR)
    gguf_sha = sha256(args.bf16_gguf)
    names = args.only or list(REFERENCES)

    for name in names:
        if name not in REFERENCES:
            raise SystemExit(f"unknown reference {name!r}; choose from {sorted(REFERENCES)}")
        build_reference(name, REFERENCES[name], args, eval_set, gguf_sha)

    if not args.skip_q8 and args.q8_gguf is not None and args.only is None:
        q8_proof(args, eval_set, gguf_sha)
    if args.only is None:
        gap_report(args)

    total = sum(p.stat().st_size for p in args.out_dir.glob("*.kld"))
    print(f"[done] {len(list(args.out_dir.glob('*.kld')))} artifacts, {human(total)} in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
