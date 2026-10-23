"""Tests for the Dory ANE-safe form and its `last_loop` caches (DANE-4).

The cache-layout half of this file needs no weights.  The model half runs NVIDIA's
real BF16 safetensors -- there are no mocked numbers anywhere.
"""

from __future__ import annotations

import torch

from anemll.utils import dory_caches as dc


# ---------------------------------------------------------------------------
# Cache inventory and rows (math-dory.md G2 / G3)
# ---------------------------------------------------------------------------

def test_cache_inventory_is_27_sliding_and_9_full():
    assert len(dc.sliding_attention_blocks()) == 27
    assert len(dc.full_attention_blocks()) == 9
    assert len([b for b in range(72) if dc.block_is_mlp(b)]) == 36
    assert dc.full_attention_blocks() == (6, 14, 22, 30, 38, 46, 54, 62, 70)


def test_cache_inventory_per_zone():
    def counts(start, end):
        s = sum(dc.block_is_sliding_attention(b) for b in range(start, end))
        f = sum(dc.block_is_full_attention(b) for b in range(start, end))
        m = sum(dc.block_is_mlp(b) for b in range(start, end))
        return s, f, m

    assert counts(0, 16) == (6, 2, 8)      # input zone
    assert counts(16, 56) == (15, 5, 20)   # recurrent zone (shared across loops)
    assert counts(56, 72) == (6, 2, 8)     # output zone


def test_slots_are_dense_and_unique():
    full, sliding = dc.cache_slots()
    assert sorted(full.values()) == list(range(9))
    assert sorted(sliding.values()) == list(range(27))
    assert set(full) | set(sliding) == {b for b in range(72) if dc.block_is_attention(b)}


def test_sliding_cache_rows_follows_llama_cpp_sizing():
    # PAD(min(ctx, window + ubatch), 256) -- 1280 at the shipped 24K context.
    assert dc.sliding_cache_rows(24576, 1024, 64) == 1280
    assert dc.sliding_cache_rows(24576, 1024, 512) == 1536
    assert dc.sliding_cache_rows(512, 1024, 64) == 512
    assert dc.sliding_cache_rows(1000, 1024, 64) == 1024 == dc._pad_to(min(1000, 1088), 256)
    assert dc.sliding_cache_rows(24576, 1024, 64) == 1280 > 1024  # the G3 correction


def test_cache_shapes():
    full, sliding = dc.cache_shapes(512, 4, 256, 1024, 64)
    assert full == (18, 4, 512, 256)     # 2 * 9 slots, K and V interleaved
    assert sliding == (54, 4, 512, 256)  # 2 * 27 slots


# ---------------------------------------------------------------------------
# Cache writes: single-row / n-row slices only (spec D5)
# ---------------------------------------------------------------------------

def _empty_cache(rows=16, heads=2, dim=4, slots=3):
    return torch.zeros(2 * slots, heads, rows, dim)


def test_decode_write_touches_only_one_row():
    cache = _empty_cache()
    k = torch.full((1, 2, 1, 4), 7.0)
    v = torch.full((1, 2, 1, 4), 9.0)
    before = cache.clone()
    dc.write_cache_rows(cache, slot=1, key=k, value=v, cache_pos=5, num_slots=3)
    changed = (cache != before).any(dim=(1, 3))          # (2*slots, rows)
    untouched = [0, 2, 3, 5]                            # slot 1 = K, slot 4 = V
    assert not changed[untouched].any()
    assert changed[1, 5] and changed[1].sum() == 1       # K row 5 only
    assert changed[4, 5] and changed[4].sum() == 1       # V row 5 only (slot + num_slots)
    assert torch.equal(cache[1, :, 5, :], k[0, :, 0, :])
    assert torch.equal(cache[4, :, 5, :], v[0, :, 0, :])


def test_prefill_write_is_one_contiguous_slice():
    cache = _empty_cache(rows=32, slots=3)
    k = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(1, 2, 8, 4)
    v = k + 100
    dc.write_cache_rows(cache, slot=2, key=k, value=v, cache_pos=8, num_slots=3)
    assert torch.equal(cache[2, :, 8:16, :], k[0])
    assert torch.equal(cache[5, :, 8:16, :], v[0])
    # nothing outside the 8 written rows moved
    assert cache[2, :, :8, :].abs().sum() == 0
    assert cache[2, :, 16:, :].abs().sum() == 0


def test_write_accepts_a_tensor_cache_pos():
    """The traced form reads cache_pos from an int32 graph input."""
    cache = _empty_cache()
    k = torch.ones(1, 2, 1, 4)
    dc.write_cache_rows(cache, slot=0, key=k, value=k, cache_pos=torch.tensor([3]), num_slots=3)
    assert cache[0, :, 3, :].sum() == 8.0


def test_ring_positions_never_straddle_the_seam():
    rows = dc.sliding_cache_rows(24576, 1024, 64)        # 1280
    assert rows == 1280
    for start in range(0, 24576, 64):
        first = dc.cache_positions(start, 64, rows)      # raises if it straddles
        assert 0 <= first < rows and first % 64 == 0
    # the last aligned batch of a ring: 1216 + 64 == 1280, exactly at the seam
    assert dc.cache_positions(1216, 64, 1280) == 1216
    assert dc.cache_positions(1280, 64, 1280) == 0


def test_ring_write_that_would_wrap_is_refused():
    try:
        dc.cache_positions(1248, 64, 1280)
    except ValueError as exc:
        assert "straddle" in str(exc)
    else:
        raise AssertionError("a prefill write crossing the ring seam must raise")


def test_reset_caches_zeroes_them():
    cache = torch.ones(4, 2, 8, 4)
    dc.reset_caches(cache)
    assert cache.abs().sum() == 0


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def test_full_causal_mask():
    m = dc.full_causal_mask(start=3, q_len=2, rows=8)
    assert m.shape == (1, 1, 2, 8)
    allowed = (m[0, 0] == 0)
    assert allowed[0].tolist() == [True, True, True, True, False, False, False, False]
    assert allowed[1].tolist() == [True, True, True, True, True, False, False, False]


def test_sliding_mask_during_fill_sees_only_the_past():
    rows, window = 512, 1024
    m = dc.sliding_ring_mask(start=0, q_len=3, rows=rows, sliding_window=window)
    allowed = (m[0, 0] == 0)
    # query 0 sees slot 0 only, query 2 sees slots 0..2, and the batch's own
    # newest written slot (2) is visible to earlier queries? No: causal.
    assert allowed[0, :4].tolist() == [True, False, False, False]
    assert allowed[2, :4].tolist() == [True, True, True, False]
    assert allowed.sum(dim=1).tolist() == [1, 2, 3]


def test_sliding_mask_in_steady_state_is_exactly_the_window():
    rows, window = 1280, 1024
    for pos in (1279, 1280, 1400, 5000, 24575):
        m = dc.sliding_ring_mask(start=pos, q_len=1, rows=rows, sliding_window=window)
        allowed = (m[0, 0, 0] == 0)
        assert int(allowed.sum()) == window, pos
        # the allowed slots are exactly the tokens at positions pos-1023..pos
        last = pos
        for j in range(rows):
            tok_pos = last - ((last - j) % rows)
            expected = (tok_pos >= 0) and (0 <= pos - tok_pos <= window - 1)
            assert bool(allowed[j]) == expected
        # the newest token is visible, one token past the window is not
        assert allowed[pos % rows]
        assert not allowed[(pos - window) % rows]


def test_mask_input_is_the_concatenation():
    ctx, rows = 32, 512
    m = dc.build_causal_mask(start=4, q_len=2, context_length=ctx, sliding_rows=rows)
    assert m.shape == (1, 1, 2, ctx + rows)
    assert torch.equal(m[..., :ctx], dc.full_causal_mask(4, 2, ctx))
    assert torch.equal(m[..., ctx:], dc.sliding_ring_mask(4, 2, rows))
