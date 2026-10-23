#!/usr/bin/env python3
"""One command: Dory1.0-4B BF16 safetensors -> the five Core ML chunks of spec D6.

    python scripts/convert_dory.py --model <bf16-dir> --output <dir>

What it does, in order:

1. prints the plan and the size estimate of every component and refuses the run
   (before converting anything) if a component would exceed the ~2 GB per-file
   limit the Neural Engine imposes, or if the disk cannot take it;
2. converts embeddings, input (blocks 0-15), recurrent (16-55), output (56-71)
   and the LM head (split into even parts), each as one multifunction file with
   infer + prefill sharing one deduplicated copy of the weights;
3. checks every function of every file against Core ML's compute plan and fails
   if any op is not on the Neural Engine;
4. compiles the packages to .mlmodelc and writes meta.yaml, config.json, the
   tokenizer, the chat template and conversion_report.json.

The converter itself lives in anemll/ane_converter/dory_converter.py; this
script only supplies the defaults the ticket uses and prints the environment
facts (uptime, free disk) that make a timing believable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HF_REPO = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
HF_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/"
    "snapshots/*/")


def default_weights() -> str:
    """The BF16 snapshot in the Hugging Face cache, if it is there."""
    found = sorted(p for p in glob.glob(HF_GLOB) if Path(p, "model.safetensors").is_file())
    return found[-1] if found else ""


def environment() -> dict:
    facts = {"cwd": os.getcwd()}
    try:
        facts["uptime"] = subprocess.run(["uptime"], capture_output=True,
                                         text=True).stdout.strip()
    except Exception:
        pass
    try:
        usage = shutil.disk_usage(str(ROOT))
        facts["disk_free_gb"] = round(usage.free / 1024 ** 3, 1)
    except Exception:
        pass
    return facts


def main(argv=None) -> int:
    from anemll.ane_converter import dory_converter as dcv

    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=None,
                        help="BF16 model directory (default: the HF cache snapshot)")
    parser.add_argument("--output", default=None,
                        help="output directory (default: "
                             "~/dev/dory-ane-artifacts/DANE-5/<ctx>ctx)")
    parser.add_argument("--context-length", type=int, default=1024,
                        help="tracer-bullet context; 24576 is the spec's target")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lut-blocks", default="6", help="LUT bits for the block chunks")
    parser.add_argument("--lut-lm-head", default="6")
    parser.add_argument("--lut-embeddings", default="none", help="spec D8: none or 8")
    parser.add_argument("--lm-head-splits", type=int, default=8)
    parser.add_argument("--recurrent-splits", type=int, default=1,
                        help="2 when the recurrent block exceeds 2 GB")
    parser.add_argument("--prefix", default="dory")
    parser.add_argument("--only", default=None, help="comma-separated components")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--no-placement-check", action="store_true")
    args = parser.parse_args(argv)

    model = args.model or default_weights()
    if not model:
        parser.error("no --model given and the BF16 snapshot is not in the HF cache")
    if not Path(model).is_dir():
        parser.error(f"--model {model} is not a directory")
    output = args.output or os.path.expanduser(
        f"~/dev/dory-ane-artifacts/DANE-5/{args.context_length}ctx")

    facts = environment()
    print(f"weights : {model}")
    print(f"output  : {output}")
    print(f"machine : {facts.get('uptime', '?')}")
    print(f"disk    : {facts.get('disk_free_gb', '?')} GB free")

    config = dcv.build_config(model, args.context_length, args.batch_size)
    converter = dcv.DoryConverter(
        model_path=model, output_dir=output, config=config, prefix=args.prefix,
        lut_blocks=args.lut_blocks, lut_lm_head=args.lut_lm_head,
        lut_embeddings=args.lut_embeddings, lm_head_splits=args.lm_head_splits,
        recurrent_splits=args.recurrent_splits,
        check_placement=not args.no_placement_check)
    try:
        result = converter.run(only=args.only, dry_run=args.dry_run,
                               do_compile=not args.no_compile)
    except dcv.ModelTooLargeError as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 2
    except dcv.PlacementError as exc:
        print(f"placement gate failed: {exc}", file=sys.stderr)
        return 3
    if args.dry_run:
        print(json.dumps(result["estimates"], indent=2))
        return 0
    print(json.dumps({"components_mb": {name: round(size / 1024 ** 2)
                                        for name, size in result["components"].items()},
                      "seconds": round(result["seconds"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
