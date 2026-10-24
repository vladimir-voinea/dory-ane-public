"""The seam between the OpenAI server and whatever generates tokens.

The server never imports Core ML, torch or transformers itself: it talks to an
`Engine` (metadata, prompt preparation, token decoding) and to a `Session`
(the thing that owns a KV cache).  `DoryEngine` in `dory_engine.py` is the real
one -- spec D9's runtime, five Core ML components driven from
`anemll.dory_chat` -- and `FakeEngine` is the scripted test double.

Keeping the seam this narrow is what lets every part of the server be tested
without a converted model, and keeps the one place that knows about Dory's
recurrent loop count small enough to read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

__all__ = ["EngineInfo", "PreparedPrompt", "Session", "Engine"]


@dataclass(frozen=True)
class EngineInfo:
    """What the server needs to know about the model behind it."""

    model_id: str
    context_length: int
    loop_min: int
    loop_max: int
    loop_default: int
    vocab_size: int
    display_name: str = ""
    kv_mode: str = "last_loop"
    reasoning_default: bool = True

    @property
    def title(self) -> str:
        return self.display_name or self.model_id


@dataclass(frozen=True)
class PreparedPrompt:
    """A rendered, tokenised conversation, plus where the model starts writing.

    `thinking_open` is True when the prompt left the assistant turn inside an
    open thinking block, which tells the stream splitter that the first tokens
    are reasoning rather than answer.
    """

    token_ids: Tuple[int, ...]
    thinking_open: bool = False


class Session(Protocol):
    """One conversation's cache, and the only place model state lives.

    `prefill` takes the tokens *not yet seen* and the position the cache has
    reached: the server tracks that so a longer conversation reuses what is
    already cached instead of re-running the prompt.  `set_loops` returns True
    when it had to reset -- spec D4: older rows keep the final loop's K/V, so a
    loop change cannot keep the cache.
    """

    @property
    def loops(self) -> int: ...

    def prefill(self, token_ids: Sequence[int], start: int) -> np.ndarray: ...

    def decode(self, token_id: int, pos: int) -> np.ndarray: ...

    def set_loops(self, loops: int) -> bool: ...

    def reset(self) -> None: ...


class Engine(Protocol):
    """A Dory model, as the server sees it."""

    info: EngineInfo

    @property
    def eos_ids(self) -> frozenset: ...

    def prepare(self, messages: Sequence[Mapping[str, Any]], *, reasoning: bool,
                tools: Optional[Sequence[Mapping[str, Any]]] = None) -> PreparedPrompt: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...

    def new_session(self, loops: int) -> Session: ...

    def close(self) -> None: ...
