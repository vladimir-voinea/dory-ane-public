"""Integration tests: the real reference artifacts built from the BF16 GGUF.

These are skipped unless the artifacts directory exists (they are 1.07 GB each
and live outside the repository). They run the real scorer against real
llama.cpp dumps - no mocking anywhere.

Rebuild them with:

    python scripts/make_dory_references.py --llama-perplexity <bin/llama-perplexity> \
        --bf16-gguf <Dory1.0-4B-BF16.gguf> --q8-gguf <Dory1.0-4B-Q8_0.gguf>
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from anemll.eval.eval_set import FIXTURE_DIR, TOKENS_FILE, load_eval_set
from anemll.eval.kld import read_llama_cpp_kld, score_files

ARTIFACTS = Path(
    os.environ.get("DORY_ARTIFACTS", Path.home() / "dev" / "dory-ane-artifacts" / "refs")
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ARTIFACTS.is_dir(), reason=f"{ARTIFACTS} not built"),
]

EXACT = {
    "bf16_exact_loop1": 1,
    "bf16_exact_loop2": 2,
    "bf16_exact_loop4": 4,
}
DECODE = {
    "bf16_decode_loop2_batch1": 1,
    "bf16_decode_loop2_batch64": 64,
}


def _meta(name: str) -> dict:
    path = ARTIFACTS / f"{name}.kld"
    if not path.is_file():
        pytest.skip(f"{path} not built")
    return json.loads(path.with_suffix(".kld.meta.json").read_text())


@pytest.mark.parametrize("name,loops", sorted(EXACT.items()))
def test_exact_reference_is_the_documented_artifact(name, loops):
    meta = _meta(name)
    assert meta["mode"] == "exact"
    assert meta["kv_cache_mode"] == "per_loop"
    assert meta["loops"] == loops
    assert meta["chunk_size"] == 4096
    assert meta["n_chunks"] == 2
    assert meta["n_rows"] == load_eval_set(FIXTURE_DIR).n_rows
    assert "recurrent_kv_cache_mode=str:per_loop" in " ".join(meta["tool"]["command"])


@pytest.mark.parametrize("name,batch", sorted(DECODE.items()))
def test_decode_reference_is_the_documented_artifact(name, batch):
    meta = _meta(name)
    assert meta["mode"] == "decode"
    assert meta["kv_cache_mode"] == "last_loop"
    assert meta["n_batch"] == batch
    assert meta["n_ubatch"] == batch
    assert "recurrent_kv_cache_mode=str:last_loop" in " ".join(meta["tool"]["command"])


def test_every_reference_scores_the_committed_eval_set():
    tokens = load_eval_set(FIXTURE_DIR).token_ids
    for name in list(EXACT) + list(DECODE):
        path = ARTIFACTS / f"{name}.kld"
        if not path.is_file():
            pytest.skip(f"{path} not built")
        source = read_llama_cpp_kld(path)
        assert source.n_rows == 4094
        assert source.vocab_size == 131072
        assert np.array_equal(source.tokens, tokens), f"{name} is on different tokens"
        assert source.positions.tolist() == load_eval_set(FIXTURE_DIR).scored_positions().tolist()


def test_q8_0_exact_loop2_lands_near_the_published_figures():
    candidate = ARTIFACTS / "q8_0_exact_loop2.kld"
    reference = ARTIFACTS / "bf16_exact_loop2.kld"
    if not candidate.is_file():
        pytest.skip(f"{candidate} not built")
    result = score_files(reference, candidate)
    # Measured here: 0.000420 mean KLD / 98.534% top-1. The published Q8_0
    # figures are 0.002972 / 98.852% on 32k positions of NVIDIA post-training
    # data; on this prose set the quantisation error shows up smaller because
    # the reference distributions are much flatter (see docs/dory-eval.md).
    # The gate is therefore on top-1, plus "nowhere near Q4_K_M territory".
    assert result.mean_kld < 0.002, result.summary()
    assert result.top1_agreement > 0.98, result.summary()


def test_decode_mode_gap_is_the_last_loop_floor():
    # measured on this set: one token per step 0.006976 KLD / 95.994% top-1,
    # 64-token batches 0.002853 / 97.533%.  That is the floor a last_loop
    # candidate can reach - it cannot beat the batching it runs.
    reference = ARTIFACTS / "bf16_exact_loop2.kld"
    if not reference.is_file():
        pytest.skip(f"{reference} not built")
    gaps = {}
    for name in DECODE:
        path = ARTIFACTS / f"{name}.kld"
        if not path.is_file():
            pytest.skip(f"{path} not built")
        result = score_files(reference, path)
        gaps[name] = result.mean_kld
        assert result.mean_kld < 0.02, f"{name}: {result.summary()}"
        assert result.top1_agreement > 0.95, f"{name}: {result.summary()}"
    scores_dir = ARTIFACTS / "scores"
    report = scores_dir / "decode_gap.json"
    if report.is_file():
        recorded = json.loads(report.read_text())
        for name, value in gaps.items():
            assert recorded[name]["mean_kld"] == pytest.approx(value, rel=1e-6)
