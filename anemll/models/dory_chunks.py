"""The traceable pieces a Dory Core ML chunk is made of.

Spec D6 (five components), D5 (ANE-safe form), D7 (caches).

The math lives in :mod:anemll.models.dory_ane_model (ticket DANE-4) and is not
repeated here.  This module only *wraps* that math into the shapes Core ML
traces: fixed input/output names, a batch dimension of 1 (decode) or
batch_size (prefill), and the union-of-states rule.

Union of states
---------------
The runtime creates **one** MLState from one component and passes it to all
five (ANEMLL tests/chat.py create_unified_state).  Core ML silently falls back
to per-chunk states when the state declarations differ, which for Dory would
stop the input block's cache writes from reaching the recurrent block.  Every
wrapper here therefore carries the *same* two buffers -- the 9 full-attention
caches and the 27 sliding ones -- and adds '0 * state[0,0,0,0]' into its output
so Core ML does not prune an unused one.  That is the Gemma 3 trick
(gemma3_converter.py:341-371, :1474-1481), extended to two buffers.

Component I/O
-------------
============ ================================ =============================
component    inputs                            output
============ ================================ =============================
embeddings   input_ids (1,seq) int32           hidden_states (1,seq,2560)
input        hidden_states, causal_mask,       output_hidden_states
             current_pos, cache_pos
recurrent    ... plus h_in                     output_hidden_states
output       same as input                     output_hidden_states
lm_head      hidden_states                     logits_1..N
============ ================================ =============================

seq is 1 for the infer function and batch_size for prefill.  current_pos is the
position of the first row (it selects the RoPE rows), cache_pos is where those
rows are written in the ring caches.  Both are int32 (1,) inputs, exactly like
ANEMLL's current_pos.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anemll.models.dory_ane_model import (
    DoryANEChunk,
    DoryANEModel,
)

__all__ = [
    "STATE_ATTRS",
    "state_attrs",
    "state_names",
    "block_input_names",
    "BlockChunk",
    "TiledBlockChunk",
    "RopeBlockChunk",
    "TiledRopeBlockChunk",
    "traced_block_chunk_class",
    "EmbeddingChunk",
    "LMHeadChunk",
    "empty_state_carrier",
    "lm_head_split_sizes",
]

#: The two state buffers of an untiled model, in declaration order.
STATE_ATTRS = ("kv_cache_full", "kv_cache_sliding")


def _caches():
    """dory_caches, imported lazily.

    This module deliberately imports nothing at module scope (the extracted
    build loads it before its siblings exist), so the shape helpers are fetched
    on first use instead of bound at import time.
    """
    from anemll.utils import dory_caches

    return dory_caches


def state_layout(config=None) -> tuple:
    """Every state buffer as (name, shape), in declaration order.

    Three axes decide the list, and only the first exists by default:

      tiles  the tiled full-attention fallback (spec D7) splits the full cache
             by context rows, one buffer per tile (DANE-6's path).
      heads  the measured single-state ceiling on this host -- 24 MiB loads,
             32 MiB fails, 3/3 attempts each -- splits each cache by key/value
             head, so one buffer holds (2 slots, heads_per_state, rows, head_dim).
      slots  the aggregate buffer interleaves K and V along dim 0: K at slot,
             V at slot + num_slots, as dory_caches.write_cache_rows expects.

    Default: no tiles, no head split -> exactly the two buffers this module
    declared before, byte for byte, so nothing that exists today changes shape.
    Tiles and head-splitting are mutually exclusive until the tiled path learns
    the head axis: both at once would produce a list neither caller understands,
    so asking for it raises instead of guessing.
    """
    tiles = getattr(config, "full_tiles", (1,)) if config is not None else (1,)
    heads_per_state = getattr(config, "heads_per_state", None) if config is not None else None
    if heads_per_state is None and config is not None:
        heads_per_state = getattr(config, "cache_heads_per_state", None)
    if heads_per_state is None:
        if len(tiles) == 1:
            full = [("kv_cache_full", _caches().cache_shapes(
                config.context_length, config.num_key_value_heads, config.head_dim,
                config.sliding_window, config.batch_size)[0])]
        else:
            shapes = _caches().full_cache_shapes(
                config.context_length, config.num_key_value_heads, config.head_dim,
                tiles, num_slots=len(_caches().cache_slots()[0]))
            full = [(f"kv_cache_full_{index}", shape)
                    for index, shape in enumerate(shapes, start=1)]
        _, sliding_shape = _caches().cache_shapes(
            config.context_length, config.num_key_value_heads, config.head_dim,
            config.sliding_window, config.batch_size)
        return tuple(full + [("kv_cache_sliding", sliding_shape)])

    if len(tiles) > 1:
        raise ValueError(
            "the tiled full cache and the per-head cache split cannot be combined yet: "
            "tiles split the full cache by context rows, heads split every cache by key "
            "head, and a model needs one of them at a time until the tiled path carries "
            "the head axis too.")

    heads = int(heads_per_state)
    num_heads = int(config.num_key_value_heads)
    if heads < 1 or num_heads % heads:
        raise ValueError(f"cache_heads_per_state must divide {num_heads} heads, got {heads}")
    full_slots, sliding_slots = _caches().cache_slots()
    full_rows = _caches().full_cache_rows(config.context_length)
    sliding_rows = _caches().sliding_cache_rows(config.context_length, config.sliding_window,
                                         config.batch_size)
    head_dim = int(config.head_dim)
    layout = []
    for slot in range(len(full_slots)):
        for head in range(0, num_heads, heads):
            layout.append((f"kv_cache_full_{slot}_{head}",
                           (2, heads, full_rows, head_dim)))
    # The sliding cache is windowed, so its carriers are grouped by SLOT, not by
    # head: one carrier holds a whole layer's K and V for every head (4 heads x
    # 1,280 rows = 5.0 MiB per slot at 24K), and the group size is the largest
    # divisor of the slot count whose carrier still fits under the measured
    # per-state ceiling -- 27 slots become 9 carriers of 15.0 MiB rather than 27
    # carriers of 5.0 MiB, because the state COUNT has its own ceiling (56 loads,
    # 63 fails) and 36 full + 27 sliding = 63 would sit exactly on the failure.
    slots_total = len(sliding_slots)
    group = int(getattr(config, "slots_per_sliding_carrier", 3))
    if group < 1 or slots_total % group:
        raise ValueError(
            f"slots_per_sliding_carrier must be a positive divisor of {slots_total} "
            f"(the number of sliding layer-slots), got {group}.  Each carrier holds "
            f"K and V of {group} layer(s); the per-state ceiling (24 MiB loads, 32 MiB "
            f"fails) caps a carrier at "
            f"{int(24.0 // (2 * sliding_rows * head_dim * num_heads * 2 / 1024 ** 2))} "
            f"layer(s) at this context length.")
    for start in range(0, slots_total, group):
        count = min(group, slots_total - start)
        layout.append((f"kv_cache_sliding_{start}_{start + count - 1}",
                       (2 * count, num_heads, sliding_rows, head_dim)))
    return tuple(layout)


def state_attrs(config=None) -> tuple:
    """Every state buffer a component declares, in declaration order.

    The union is derived from the config rather than hard-coded: the converter,
    the runtime's single MLState and every component must agree on the list or
    Core ML refuses the shared state.  See state_layout for the axes.
    """
    return tuple(name for name, _ in state_layout(config))


def state_summary(config=None) -> str:
    """One line for the conversion report and the meta: sizes and the ceiling.

    Anyone who later sees a -14 on this model should find these numbers next to
    the artifact rather than re-deriving them from shapes.
    """
    layout = state_layout(config)
    sizes = []
    for _, shape in layout:
        elements = 1
        for dim in shape:
            elements *= int(dim)
        sizes.append(elements * 2 / 1024 ** 2)
    total = sum(sizes)
    largest = max(sizes) if sizes else 0.0
    smallest = min(sizes) if sizes else 0.0
    verdict = "OK" if largest <= 24.0 and len(layout) <= 56 else "OVER THE MEASURED LIMIT"
    return ("%d states, %.1f-%.1f MiB each, %.1f MiB total, max single state "
            "%.1f MiB / %d states (limits measured at 24 MiB / 56 states OK, "
            "32 MiB / 63 states FAIL): %s"
            % (len(layout), smallest, largest, total, largest, len(layout), verdict))


def state_names(config=None, prefix: str = "model") -> tuple:
    """Qualified names of the states, as Core ML sees them.

    coremltools 8.3 renames dots inside a state name (model.kv_cache_full ->
    model_kv_cache_full) but consistently, so the union works as long as every
    component is built by this module.
    """
    return tuple(f"{prefix}.{attr}" for attr in state_attrs(config))


def block_input_names(tiled: bool, loop_entry: bool, rope_tables: bool = False) -> list:
    """The traced input names of a block component, in signature order.

    The converter, the traced wrapper and the runtime's call must agree on this
    order exactly: Core ML maps the declared inputs positionally onto the traced
    graph.  A tiled full cache needs the per-tile write index and gate, because
    a traced graph cannot branch on the position (see BlockChunk).
    """
    names = ["hidden_states", "causal_mask", "current_pos", "cache_pos"]
    if tiled:
        names += ["write_index_1", "write_gate_1", "write_index_2", "write_gate_2"]
    if rope_tables:
        # the four cos/sin tensors the host slices from its own tables (24K)
        names += ["cos_sliding", "sin_sliding", "cos_full", "sin_full"]
    if loop_entry:
        names.append("h_in")
    return names


class _StateTouching(nn.Module):
    """Base class for wrappers that keep the full state union alive.

    Subclasses set self.model to the DoryANEModel that owns the cache buffers
    (an empty one for the embedding/LM-head components).  The buffers must be
    the *module's own* registered buffers, not copies: Core ML matches states by
    the attribute path from the traced root.
    """

    model: DoryANEModel

    def touch_states(self, out: torch.Tensor) -> torch.Tensor:
        """Add 0 * every state, so none is pruned as unused.

        The list comes from the model's own buffers rather than a hard-coded
        pair: the split layout declares 45 of them, and one left out is a state
        Core ML would drop from the union the runtime shares between components.
        """
        buffers = self.model.state_buffers
        dummy = buffers[0][0, 0, 0, 0] * 0.0
        for cache in buffers[1:]:
            dummy = dummy + cache[0, 0, 0, 0] * 0.0
        return out + dummy.view(1, 1, 1)


def empty_state_carrier(config) -> DoryANEModel:
    """A DoryANEModel with no blocks -- states only.

    The embedding and LM-head components compute nothing on a cache, but they
    must still declare both to stay in the union.
    """
    return DoryANEModel(config, block_indices=[])


class BlockChunk(_StateTouching):
    """A contiguous range of physical blocks (input / recurrent / output).

    loop_entry / loop_exit are the recurrent block's entry and exit norms
    (h <- L2(h + h_in) and the trailing L2).  They are compile-time flags, so
    splitting the recurrent block across two files keeps exactly one entry and
    one exit.
    """

    def __init__(self, model, start_block: int, end_block: int,
                 loop_entry: bool = False, loop_exit: bool = False):
        super().__init__()
        self.model = model
        self.chunk = DoryANEChunk(model, start_block, end_block,
                                  loop_entry=loop_entry, loop_exit=loop_exit)
        self.start_block = int(start_block)
        self.end_block = int(end_block)
        self.loop_entry = bool(loop_entry)
        self.loop_exit = bool(loop_exit)

    def run(self, hidden_states, causal_mask, current_pos, cache_pos, h_in=None,
            write_index_1=None, write_gate_1=None, write_index_2=None, write_gate_2=None,
            cos_sliding=None, sin_sliding=None, cos_full=None, sin_full=None):
        out = self.chunk(hidden_states, causal_mask, current_pos, cache_pos, h_in,
                         write_index_1, write_gate_1, write_index_2, write_gate_2,
                         cos_sliding=cos_sliding, sin_sliding=sin_sliding,
                         cos_full=cos_full, sin_full=sin_full)
        return self.touch_states(out)

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos, h_in=None):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos, h_in)


class TiledBlockChunk(BlockChunk):
    """The same chunk, traced with the per-tile write index and gate.

    A separate class rather than an optional argument: the traced signature has
    to place every tensor input first (torch.jit.trace cannot be handed a None
    placeholder), while the eager callers keep h_in as the fifth positional
    argument.  block_input_names() is the single source of truth for the order.
    """

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos,
                write_index_1, write_gate_1, write_index_2, write_gate_2, h_in=None):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos, h_in,
                        write_index_1, write_gate_1, write_index_2, write_gate_2)


class RopeBlockChunk(BlockChunk):
    """Traced with the RoPE cos/sin rows as inputs (the 24K shape); h_in last.

    A traced signature has to put every tensor input first, because
    torch.jit.trace cannot be handed a None placeholder, so each present group of
    inputs gets a class whose parameter order is exactly block_input_names().
    h_in stays the fifth positional argument of the plain BlockChunk, which is
    where the eager callers and the block tests pass it.
    """

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos,
                cos_sliding, sin_sliding, cos_full, sin_full, h_in=None):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos, h_in,
                        cos_sliding=cos_sliding, sin_sliding=sin_sliding,
                        cos_full=cos_full, sin_full=sin_full)


class TiledRopeBlockChunk(BlockChunk):
    """Both: the tile write inputs and the RoPE rows, h_in last."""

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos,
                write_index_1, write_gate_1, write_index_2, write_gate_2,
                cos_sliding, sin_sliding, cos_full, sin_full, h_in=None):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos, h_in,
                        write_index_1, write_gate_1, write_index_2, write_gate_2,
                        cos_sliding=cos_sliding, sin_sliding=sin_sliding,
                        cos_full=cos_full, sin_full=sin_full)


def traced_block_chunk_class(tiled: bool, rope_tables: bool):
    """The wrapper whose signature is exactly block_input_names(tiled, _, rope)."""
    if tiled and rope_tables:
        return TiledRopeBlockChunk
    if tiled:
        return TiledBlockChunk
    if rope_tables:
        return RopeBlockChunk
    return BlockChunk


class EmbeddingChunk(_StateTouching):
    """Token ids -> hidden states, with the state union declared."""

    def __init__(self, config, embed_tokens: nn.Embedding, model: DoryANEModel):
        super().__init__()
        self.model = model
        self.embed_tokens = embed_tokens

    def forward(self, input_ids):
        return self.touch_states(self.embed_tokens(input_ids))


def lm_head_split_sizes(vocab_size: int, splits: int) -> list:
    """Even vocab_size // splits row slices, or a clear error.

    ANEMLL's LM-head split is even by construction (llama_model.py:1212-1221)
    and Dory's 131,072 is divisible by 8 and 16.  An uneven split is refused
    rather than silently padded: the runtime rebuilds the vocabulary from
    lm_head_chunk_sizes in this order.
    """
    vocab_size = int(vocab_size)
    splits = int(splits)
    if splits < 2:
        raise ValueError(
            f"the LM head must be split into at least 2 parts (got {splits}): "
            "a 131,072-wide conv output exceeds the 65,536 ANE axis limit")
    if vocab_size % splits:
        raise ValueError(f"vocab_size {vocab_size} is not divisible by {splits} "
                         "LM-head splits; pick a divisor of the vocabulary")
    size = vocab_size // splits
    return [size] * splits


class LMHeadChunk(_StateTouching):
    """The LM head, split into N independent 1x1 convolutions.

    Each part is vocab//N output channels (16,384 for N=8), which keeps a single
    ANE axis inside its documented range.  The parts are separate graph outputs,
    so one file holds the whole head and the runtime concatenates logits_1..N in
    order.

    argmax replaces the logits with argmax_idx_i (int32, a **local** index
    inside part i) plus argmax_val_i (fp16); the global token id is the local
    index plus that part's offset, computed on the host (ANEMLL
    llama_converter.py:661-671).
    """

    def __init__(self, config, weight: torch.Tensor, splits: int, model: DoryANEModel,
                 argmax: bool = False):
        super().__init__()
        self.model = model
        self.splits = int(splits)
        self.argmax = bool(argmax)
        sizes = lm_head_split_sizes(config.vocab_size, self.splits)
        self.chunk_sizes = sizes
        weight = weight.detach()
        if weight.shape[0] != config.vocab_size:
            raise ValueError(f"lm_head weight has {weight.shape[0]} rows, "
                             f"config says vocab_size {config.vocab_size}")
        parts = []
        start = 0
        for size in sizes:
            conv = nn.Conv2d(config.hidden_size, size, kernel_size=1, bias=False,
                             dtype=weight.dtype)
            conv.weight.data.copy_(
                weight[start:start + size].view(size, config.hidden_size, 1, 1))
            parts.append(conv)
            start += size
        self.parts = nn.ModuleList(parts)

    def forward(self, hidden_states):
        x = hidden_states.permute(0, 2, 1).unsqueeze(2)
        outs = []
        for conv in self.parts:
            logits = conv(x).squeeze(2).permute(0, 2, 1)
            if self.argmax:
                val, idx = torch.max(logits, dim=-1)
                outs.append(idx.to(torch.int32))
                outs.append(val)
            else:
                outs.append(logits)
        # keep both states alive on the first output only
        outs[0] = self.touch_states(outs[0])
        return tuple(outs)

    @property
    def output_names(self) -> list:
        if self.argmax:
            names = []
            for i in range(1, self.splits + 1):
                names.extend([f"argmax_idx_{i}", f"argmax_val_{i}"])
            return names
        return [f"logits_{i}" for i in range(1, self.splits + 1)]
