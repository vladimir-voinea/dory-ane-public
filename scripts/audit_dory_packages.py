"""Audit the converted Dory packages: size, functions, and recorded names.

The size check compares each package against estimate_component_bytes() from the
converter.  That estimate is calibrated on a real package: coremltools
palettizes over a group of 16 weights, so LUT6 costs 0.759 bytes per parameter
(6.07 bits) rather than the 1.05 a one-scale-per-8-weights model predicts.

On top of the size, each package must hold exactly one function (Dory ships one
package per function -- welded multifunction packages are rejected by macOS) and
the function recorded in its metadata must match the function its file name
claims.

usage:
    python scripts/audit_dory_packages.py <model-dir> --model-path <hf-snapshot>
        [--context 4096] [--ub 64] [--lut-blocks 6] [--lut-lm-head 6]
        [--lut-embeddings none] [--component input ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from anemll.ane_converter import dory_converter as dcv

TOLERANCE = 0.15
HARD_LIMIT = 2 * 1024 ** 3


def directory_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def spec_functions(package: Path):
    """The functions the package declares; a plain mlprogram has one, "main"."""
    try:
        from coremltools.models.utils import load_spec
        declared = list(load_spec(str(package)).description.functions)
    except Exception:
        return []
    return [f.name for f in declared] if declared else ["main"]


def metadata_functions(package: Path):
    try:
        from coremltools.models.utils import load_spec
        value = load_spec(str(package)).description.metadata.userDefined.get(
            "com.anemll.functions", "")
    except Exception:
        return []
    return [part for part in str(value).split(",") if part]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir")
    parser.add_argument("--model-path", required=True,
                        help="the converted checkpoint (for config.json)")
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--ub", type=int, default=64)
    parser.add_argument("--lut-blocks", default="6")
    parser.add_argument("--lut-lm-head", default="6")
    parser.add_argument("--lut-embeddings", default="none")
    parser.add_argument("--lm-head-splits", type=int, default=8)
    parser.add_argument("--component", action="append", default=None)
    args = parser.parse_args()
    model_dir = Path(args.model_dir)

    config = dcv.build_config(args.model_path, args.context, args.ub)
    converter = dcv.DoryConverter(model_path=args.model_path, output_dir=model_dir,
                                  config=config, lut_blocks=args.lut_blocks,
                                  lut_lm_head=args.lut_lm_head,
                                  lut_embeddings=args.lut_embeddings,
                                  lm_head_splits=args.lm_head_splits,
                                  check_placement=False, verbose=False)

    print(f"{'package':30s} {'MiB':>7s} {'predicted':>9s} {'diff':>7s}  checks")
    failures = []
    for spec in converter.specs:
        if args.component and spec.name not in args.component:
            continue
        bits = converter.lut_bits.get(spec.lut_role)
        predicted = dcv.estimate_component_bytes(spec, config, bits)
        for function, _ in converter._functions(spec):
            package = model_dir / converter.file_name(spec, function)
            label = f"{spec.name}/{function}"
            if not package.is_dir():
                failures.append(f"{label}: missing {package.name}")
                print(f"{label:30s} {'-':>7s} {predicted / 1024**2:9.0f} {'-':>7s}  MISSING")
                continue
            size = directory_bytes(package)
            diff = (size - predicted) / predicted
            functions = spec_functions(package)
            recorded = metadata_functions(package)
            problems = []
            if abs(diff) > TOLERANCE:
                problems.append(f"{diff * 100:+.0f}% off the prediction")
            if size > HARD_LIMIT:
                problems.append("over the 2 GB guard")
            if len(functions) != 1:
                problems.append(f"{len(functions)} functions, expected 1 ({functions})")
            if recorded != [function]:
                problems.append(f"metadata says {recorded or 'nothing'}")
            status = "ok" if not problems else "; ".join(problems)
            print(f"{label:30s} {size / 1024**2:7.0f} {predicted / 1024**2:9.0f} "
                  f"{diff * 100:+6.1f}%  {status}")
            if problems:
                failures.append(f"{label}: {status}")
    if failures:
        print("\nAUDIT: FAILED")
        for line in failures:
            print("  -", line)
        return 1
    print("\nAUDIT: PASSED (each package within 15 % of the calibrated prediction, "
          "exactly one function, recorded function matches the file name)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
