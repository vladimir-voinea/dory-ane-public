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
    "state_names",
    "BlockChunk",
    "EmbeddingChunk",
    "LMHeadChunk",
    "empty_state_carrier",
    "lm_head_split_sizes",
]

#: The two state buffers every component declares, in this order.
STATE_ATTRS = ("kv_cache_full", "kv_cache_sliding")


def state_names(prefix: str = "model") -> tuple:
    """Qualified names of the two states, as Core ML sees them.

    coremltools 8.3 renames dots inside a state name (model.kv_cache_full ->
    model_kv_cache_full) but consistently, so the union works as long as every
    component is built by this module.
    """
    return tuple(f"{prefix}.{attr}" for attr in STATE_ATTRS)


class _StateTouching(nn.Module):
    """Base class for wrappers that keep the full state union alive.

    Subclasses set self.model to the DoryANEModel that owns the cache buffers
    (an empty one for the embedding/LM-head components).  The buffers must be
    the *module's own* registered buffers, not copies: Core ML matches states by
    the attribute path from the traced root.
    """

    model: DoryANEModel

    def touch_states(self, out: torch.Tensor) -> torch.Tensor:
        """Add 0 * both caches, so neither is pruned as unused."""
        carrier = self.model
        dummy = carrier.kv_cache_full[0, 0, 0, 0] * 0.0
        dummy = dummy + carrier.kv_cache_sliding[0, 0, 0, 0] * 0.0
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

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos, h_in=None):
        if self.loop_entry:
            if h_in is None:
                raise ValueError("the recurrent chunk needs h_in (the loop entry state)")
            out = self.chunk(hidden_states, causal_mask, current_pos, cache_pos, h_in)
        else:
            out = self.chunk(hidden_states, causal_mask, current_pos, cache_pos)
        return self.touch_states(out)


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
