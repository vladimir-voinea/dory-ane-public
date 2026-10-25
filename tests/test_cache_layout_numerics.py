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
target).

How the model is driven
-----------------------
Not with a 72-block model: the claim under test is the arithmetic of the cache
layout, and a five-block slice exercises every part of it -- sliding slots 0, 1
and 3 (slots 0 and 1 share one sliding carrier, slot 3 starts the next) and full
slots 0 and 1, so both the shared-carrier indexing and the per-slot separation are
covered -- for a fraction of the time and memory the full model would need.  The
logits are a seeded 1x1 readout over the hidden states, which is the same kind of
layer the shipped LM head is (one 1x1 convolution per vocabulary part), with a
4,096-wide vocabulary instead of 131,072.

Four calls per model, with the caches carried between them: a prefill of 64 rows
(the traced batch size), then single-token steps at 64, 1,250 and 4,000.  The
early steps catch the trap this refactor walks past -- the ring's cache_pos and
the tiled write_index only mean anything if the current rows were written
*before* the buffers were read, so a version that read first would pass a
single-token test and be wrong from the second token on.  The late steps are the
other half: near position 0 the causal mask leaves one or two keys visible, the
softmax is one-hot, and the attention result stops depending on the queries at
all, so a head reading another head's K and V would pass unnoticed.  Deep in the
context the softmax is genuinely soft and every head's contribution has to come
from its own buffer.  That only holds if the logits have a real spread, which is
why _seed_weights gives the sq scale its production magnitude (~256, see
DoryANEAttention.compute_sq) instead of a small random value.
"""
from __future__ import annotations

import gc

import pytest
import torch
import torch.nn as nn

from anemll.models import dory_ane_model as dam
from anemll.models import dory_chunks as dch
from anemll.utils import dory_caches as dc

#: one sliding pair inside a carrier, one sliding layer at the start of the next
#: carrier, and two full-attention layers: five blocks, four attention slots.
BLOCKS = (0, 2, 6, 8, 14)
READOUT_VOCAB = 4096
#: prefill, then three single-token steps.  The first two catch a read-before-write
#: (nothing is in the cache yet, so a graph that read first would attend to zeros);
#: the last two start deep in the context, where the causal mask leaves more than a
#: thousand keys visible -- there the softmax is genuinely soft, and a head that
#: read another head's K and V cannot come out the same by accident.
STEPS = ((0, 64), (64, 1), (1250, 1), (4000, 1))


def _config(context_length: int, **overrides):
    return dam.DoryANEConfig(context_length=context_length, compute_mode="matmul",
                             **overrides)


def _seeded_hidden(hidden_size: int, seq: int, start: int):
    generator = torch.Generator().manual_seed(1000 + start)
    return torch.randn(1, seq, hidden_size, generator=generator, dtype=torch.float16)


def _seed_weights(model) -> None:
    torch.manual_seed(11)
    with torch.no_grad():
        for _, parameter in model.named_parameters():
            parameter.normal_(0.0, 0.05)
        for name, buffer in model.named_buffers():
            if name.startswith("kv_cache"):
                # the caches stay empty: the two layouts declare a different
                # number of buffers, so seeding them would hand the aggregated
                # model and the split model different contents in the rows
                # neither has written yet, and the late steps would compare two
                # different caches instead of two layouts of one cache
                continue
            if "cos_cached" in name or "sin_cached" in name or not buffer.numel():
                continue
            if name.endswith(".sq"):
                # the real sq is O(256) -- it is (sqk * sqrt(head_dim) /
                # init_std) squared -- and that magnitude is what makes the
                # attention logits large enough for the softmax to be sharp.
                # Left at 0.1 every logit lands around 1e-4, the softmax is
                # uniform, and the attention output stops depending on the
                # queries at all: a head reading another head's K and V would
                # then pass unnoticed.
                buffer.copy_(torch.rand(buffer.shape, dtype=torch.float32) * 100.0 + 200.0)
                continue
            # positive, so the L2 scales and the residual alpha stay in range
            buffer.normal_(0.0, 0.05).abs_().add_(0.1)


class _Readout(nn.Module):
    """A seeded 1x1 convolution standing in for the split LM head."""

    def __init__(self, hidden_size: int, vocab: int, seed: int = 23):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(hidden_size, vocab, kernel_size=1, bias=False,
                             dtype=torch.float16)
        with torch.no_grad():
            self.conv.weight.normal_(0.0, 0.02)

    def forward(self, hidden_states):
        x = hidden_states.permute(0, 2, 1).unsqueeze(2)
        return self.conv(x).squeeze(2).permute(0, 2, 1).float()


def _logits(model, readout, config, steps=STEPS):
    """Logits of every step, the caches carried from one call to the next."""
    out = []
    with torch.no_grad():
        for start, q_len in steps:
            hidden = _seeded_hidden(config.hidden_size, q_len, start)
            mask = dc.build_causal_mask(start, q_len, config.context_length,
                                        config.sliding_rows, config.sliding_window,
                                        dtype=config.dtype)
            cache_pos = start % config.sliding_rows
            h = model.process_blocks(hidden, mask, start, cache_pos)
            out.append(readout(h))
    return out


def _mib(shape) -> float:
    elements = 1
    for dim in shape:
        elements *= int(dim)
    return elements * 2 / 1024 ** 2


@pytest.mark.parametrize("context_length", [4096, 24576])
def test_split_cache_matches_aggregated(context_length):
    """Logits from the split layout equal the aggregated reference."""
    config_ref = _config(context_length)
    reference = dam.DoryANEModel(config_ref, block_indices=BLOCKS).eval()
    _seed_weights(reference)
    readout = _Readout(config_ref.hidden_size, READOUT_VOCAB).eval()
    expected = _logits(reference, readout, config_ref)

    # the reference must be the shipped aggregated pair, or this compares a
    # layout against itself
    assert reference.state_buffer_names == ("kv_cache_full", "kv_cache_sliding")

    weights = {name: tensor.detach().clone()
               for name, tensor in reference.state_dict().items()}
    del reference
    gc.collect()

    config_split = _config(context_length, heads_per_state=1,
                           slots_per_sliding_carrier=3)
    layout = dch.state_layout(config_split)
    full = [shape for name, shape in layout if name.startswith("kv_cache_full")]
    sliding = [shape for name, shape in layout if name.startswith("kv_cache_sliding")]
    assert len(layout) == 45, "36 full states (one per layer-slot and head) + 9 carriers"
    assert len(full) == 36 and len(sliding) == 9
    assert all(int(shape[1]) == 1 for shape in full), "one head per full state"
    assert max(_mib(shape) for _, shape in layout) <= 24.0 + 1e-6, "over the 24 MiB ceiling"
    if context_length == 24576:
        assert pytest.approx(max(_mib(shape) for _, shape in layout), abs=0.1) == 24.0

    split = dam.DoryANEModel(config_split, block_indices=BLOCKS).eval()
    split.load_state_dict(weights)
    assert len(split.state_buffers) == 45, "the model registered the split layout"
    actual = _logits(split, readout, config_split)

    assert len(actual) == len(expected) == len(STEPS)
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert got.shape == want.shape
        max_abs = (got - want).abs().max().item()
        scale = want.abs().max().item() or 1.0
        assert max_abs <= 1e-3 * scale, (
            f"step {index} at {context_length}: logits diverged, max abs {max_abs:.3e} "
            f"against scale {scale:.3e}")
        assert torch.equal(got.argmax(-1), want.argmax(-1)), (
            f"step {index} at {context_length}: argmax differs at one or more positions")


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


# -- the conversion path: a declared state must be a buffer read ------------

@pytest.mark.parametrize("overrides, expected_states", [
    ({}, 2),
    ({"heads_per_state": 1, "slots_per_sliding_carrier": 3}, 45),
    ({"heads_per_state": 2, "slots_per_sliding_carrier": 9}, 21),
])
def test_the_traced_chunk_reads_every_state_it_declares(overrides, expected_states):
    """Every declared state must show up as a buffer read in the traced graph.

    Core ML maps a declared state onto a buffer read in the traced graph; a buffer
    the tracer inlined as a constant, or one reached through a Python list built
    once at construction, would convert into a model with no cache at all.  That
    cannot be caught by comparing two PyTorch models -- both would still compute
    the right answer -- so it is asserted on the graph itself.  Tracing needs no
    Core ML, which is why this check can live here.
    """
    config = dam.DoryANEConfig(hidden_size=64, intermediate_size=32,
                               num_attention_heads=16, num_key_value_heads=4,
                               head_dim=256, vocab_size=64, context_length=128,
                               batch_size=2, compute_mode="matmul", **overrides)
    model = dam.DoryANEModel(config, block_indices=tuple(range(7))).eval()
    chunk = dch.BlockChunk(model, 0, 7).eval()
    assert len(model.state_buffer_names) == expected_states

    q_len = 2
    hidden = torch.randn(1, q_len, config.hidden_size, dtype=torch.float16)
    mask = torch.zeros(1, 1, q_len, config.mask_len, dtype=torch.float16)
    pos = torch.tensor([0], dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(chunk, (hidden, mask, pos, pos), strict=False)
    graph = str(traced.inlined_graph)
    missing = [name for name in model.state_buffer_names if f'name="{name}"' not in graph]
    assert not missing, f"states the traced graph never reads: {missing[:6]}"
