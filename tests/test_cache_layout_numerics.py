"""Numerics equality: per-(layer, head) cache layout must reproduce the aggregated one.

Reference is the AGGREGATED layout, because that is what DANE-4's form gates
validated: the test asserts that splitting the cache by (layer, head) changes
nothing but the storage, not that both layouts are plausible.

Tolerance, justified rather than picked: the split changes how K/V are stored and
read back, but every arithmetic operation is unchanged (same matmul, same
softmax, same dtypes, same order within a head), so the expected difference is
exactly zero.  A tolerance of 0.0 would fail on any reordering the compiler is
free to introduce, so the test allows a relative 1e-3 on logits -- three orders
of magnitude below the 0.05 KLD gate's resolution -- and asserts the ARGMAX is
identical at every scored position, which is the property the KLD gate and the
recall gate actually consume.

Runs at 4,096 (the shipped default, must be untouched) and at 24,576 (the design
target).  SKIPPED until DoryANEChunk consumes per-(layer, head) buffers: the
aggregated path has no split variant to compare against yet.
"""
from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skip(
    reason="awaits the chunk refactor: DoryANEChunk still addresses one "
           "aggregated cache per component, so there is no per-(layer, head) "
           "path to compare against.  The test is written and will run the "
           "moment the chunk reads split buffers.")


def _seeded_inputs(hidden_size: int, seq: int, seed: int = 1234):
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(1, seq, hidden_size, generator=generator, dtype=torch.float16)
    return hidden


@pytest.mark.parametrize("context_length", [4096, 24576])
def test_split_cache_matches_aggregated(context_length):
    """Logits from the split layout equal the aggregated reference."""
    from anemll.models import dory_ane_model as dam

    config = dam.DoryANEConfig.variant(context_length=context_length)
    reference = dam.DoryANE(config).eval()
    split = dam.DoryANE(config, heads_per_state=1, slots_per_sliding_carrier=3).eval()
    split.load_state_dict(reference.state_dict())

    hidden = _seeded_inputs(config.hidden_size, seq=2)
    with torch.no_grad():
        expected = reference.forward(hidden, start=0)
        actual = split.forward(hidden, start=0)

    assert actual.shape == expected.shape
    max_abs = (actual.float() - expected.float()).abs().max().item()
    scale = expected.float().abs().max().item() or 1.0
    assert max_abs <= 1e-3 * scale, f"logits diverged: max abs {max_abs:.3e} vs scale {scale:.3e}"
    assert torch.equal(actual.float().argmax(-1), expected.float().argmax(-1)), \
        "argmax differs at one or more positions"


def test_default_state_list_is_unchanged():
    """The shipped default must not move: two aggregated buffers, same shapes."""
    from anemll.models import dory_chunks as dch
    from anemll.utils import dory_caches as dc

    class Config:
        context_length = 4096
        num_key_value_heads = 4
        head_dim = 256
        sliding_window = 1280
        batch_size = 64
        full_tiles = (1,)

    layout = dch.state_layout(Config())
    assert [name for name, _ in layout] == ["kv_cache_full", "kv_cache_sliding"]
    expected_full, expected_sliding = dc.cache_shapes(
        Config.context_length, Config.num_key_value_heads, Config.head_dim,
        Config.sliding_window, Config.batch_size)
    assert tuple(layout[0][1]) == tuple(expected_full)
    assert tuple(layout[1][1]) == tuple(expected_sliding)
