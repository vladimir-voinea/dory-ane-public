"""Placement probe: do the 24,576-row caches stay on the ANE inside a chunk?

Spec D7's open question.  Converting all five components at 24K takes an hour of
palettization; this probe converts a *seven-block* chunk (blocks 0-6: three
sliding, one full, three MLP) at the target context, unquantized -- quantization
does not change placement -- and prints Core ML's compute plan per op for both
functions.  The answer decides one 24,576-row cache against the 16,384 + 8,192
tile fallback.

    python scripts/dory_24k_placement_probe.py --model <bf16 dir> [--full-tile-rows 16384,8192]

The chunk carries the same state declaration as the real one, because every
component declares the union of caches: 9 full-attention caches and 27 sliding.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import torch

__all__ = ["probe", "main"]

BLOCKS = (0, 1, 2, 3, 4, 5, 6)          # 0,2,4 sliding; 6 full; 1,3,5 MLP


def probe(model_path, context_length=24576, batch_size=64, tile_rows=None,
          verbose=True) -> dict:
    import coremltools as ct
    import numpy as np

    from anemll.ane_converter.dory_converter import (
        _ComponentModel, assert_ane_only, placement_report,
    )
    from anemll.models import dory_chunks as dch
    from anemll.models.dory_ane_model import DoryANEConfig

    config = DoryANEConfig(context_length=int(context_length),
                           batch_size=int(batch_size),
                           full_cache_tiles=tile_rows)
    model = _ComponentModel(config, list(BLOCKS), load_lm_head=False)
    model.load_pretrained_weights(str(model_path), blocks=list(BLOCKS))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.model.reset_caches()

    if config.tiled:
        wrapper = dch.TiledBlockChunk(model.model, 0, len(BLOCKS))
    else:
        wrapper = dch.BlockChunk(model.model, 0, len(BLOCKS))
    wrapper.eval()

    results = {}
    out_dir = Path(tempfile.mkdtemp(prefix="dory24k-probe-"))
    for function, seq in (("infer", 1), ("prefill", int(batch_size))):
        dtype = config.dtype
        hidden = torch.zeros(1, seq, config.hidden_size, dtype=dtype)
        mask = torch.zeros(1, 1, seq, config.mask_len, dtype=dtype)
        pos = torch.zeros(1, dtype=torch.int32)
        cache_pos = torch.zeros(1, dtype=torch.int32)
        args = [hidden, mask, pos, cache_pos]
        if config.tiled:
            for _ in range(2):
                args.append(torch.zeros(1, dtype=torch.int32))
                args.append(torch.zeros(1, dtype=dtype))

        inputs = [
            ct.TensorType(name="hidden_states", shape=tuple(hidden.shape), dtype=np.float16),
            ct.TensorType(name="causal_mask", shape=tuple(mask.shape), dtype=np.float16),
            ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
            ct.TensorType(name="cache_pos", shape=(1,), dtype=np.int32),
        ]
        if config.tiled:
            for tile in (1, 2):
                inputs.append(ct.TensorType(name=f"write_index_{tile}", shape=(1,),
                                            dtype=np.int32))
                inputs.append(ct.TensorType(name=f"write_gate_{tile}", shape=(1,),
                                            dtype=np.float16))
        states = []
        for attr in dch.state_attrs(config):
            buffer = getattr(model.model, attr)
            states.append(ct.StateType(name=f"model.{attr}",
                                       wrapped_type=ct.TensorType(
                                           shape=tuple(int(d) for d in buffer.shape),
                                           dtype=np.float16)))
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, tuple(args), strict=False)
        started = time.time()
        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
            states=states,
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        path = out_dir / f"probe_{function}.mlpackage"
        mlmodel.save(str(path))
        report = placement_report(path)
        entry = next(iter(report.values()))            # one function per package
        results[function] = {"functions": list(report), **entry,
                            "seconds": round(time.time() - started, 1)}
        if verbose:
            print(f"[{function}] ane {entry['ane']} gpu {entry['gpu']} cpu {entry['cpu']} "
                  f"none {entry['none']} of {entry['ops']} ops, "
                  f"{results[function]['seconds']}s")
            for label in entry["fallbacks"][:12]:
                print(f"    fallback {label}")
        try:
            assert_ane_only(report, f"probe/{function}")
            results[function]["gate"] = "pass"
        except Exception as error:                    # noqa: BLE001 - the point of the probe
            results[function]["gate"] = "fail"
            results[function]["gate_error"] = str(error)
            if verbose:
                print(f"[{function}] NOT ALL ON ANE: {error}")
    results["path"] = str(out_dir)
    results["context_length"] = int(context_length)
    results["tile_rows"] = list(tile_rows) if tile_rows else [int(context_length)]
    results["state_mib"] = round(
        sum(int(torch.tensor(shape, dtype=torch.int64).prod()) for shape in
            [getattr(model.model, attr).shape for attr in dch.state_attrs(config)]) * 2 / 2 ** 20, 1)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="24K ANE placement probe")
    parser.add_argument("--model", required=True, help="BF16 safetensors directory")
    parser.add_argument("--context-length", type=int, default=24576)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--full-tile-rows", default=None,
                        help="e.g. 16384,8192 to probe the tile fallback")
    parser.add_argument("--out", default=None, help="write the report JSON here")
    args = parser.parse_args(argv)
    tiles = None
    if args.full_tile_rows:
        tiles = tuple(int(p) for p in args.full_tile_rows.replace(" ", "").split(",") if p)
    report = probe(args.model, args.context_length, args.batch_size, tiles)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str) + chr(10))
    failed = [fn for fn in ("infer", "prefill") if "gate_error" in (report.get(fn) or {})]
    print(json.dumps({"passed": not failed, "failed_functions": failed,
                      "state_mib": report["state_mib"],
                      "tile_rows": report["tile_rows"]}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())

