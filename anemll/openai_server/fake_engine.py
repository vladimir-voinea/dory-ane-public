"""A scripted engine: the server's test double, and its `--fake` mode.

Token ids are UTF-8 bytes and id 0 is EOS, so a scripted reply is exactly the
text a tokenizer would hand the server, one character per token.  That is
enough to exercise every part of the server that matters -- markers split
across token boundaries, cache reuse, loop changes, stop sequences -- without
Core ML, weights, or a machine that can hold a 5 GB model.

The reply is chosen by looking for a keyword in the prompt, so a test says
`FakeEngine(replies={"price": "It costs $3. "})` and then asks about a price.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from anemll.openai_server.engine import EngineInfo, PreparedPrompt

__all__ = ["FakeEngine", "FakeSession", "EOS_TOKEN_ID"]

#: The fake vocabulary is one byte per token, and byte 0 is the end of a turn.
EOS_TOKEN_ID = 0
_VOCAB = 256
_LOW = -1e4


class FakeEngine:
    """A deterministic engine with no model behind it."""

    def __init__(self, *, replies: Optional[Mapping[str, str]] = None,
                 default_reply: str = "", context_length: int = 24576,
                 loop_min: int = 1, loop_max: int = 4, loop_default: int = 2,
                 model_id: str = "dory-ane", reasoning_default: bool = True):
        self.info = EngineInfo(model_id=model_id, context_length=int(context_length),
                               loop_min=int(loop_min), loop_max=int(loop_max),
                               loop_default=int(loop_default), vocab_size=_VOCAB,
                               reasoning_default=bool(reasoning_default))
        self.replies: Dict[str, str] = dict(replies or {})
        self.default_reply = default_reply
        # Counters a test can assert on: what the server actually asked the model
        # to do, which is how cache reuse and cache resets are proved.
        self.prefill_calls = 0
        self.prefilled_tokens = 0
        self.decoded_tokens = 0
        self.reset_count = 0
        self.reset_loops: List[int] = []

    # -- engine ----------------------------------------------------
    @property
    def eos_ids(self) -> frozenset:
        return frozenset({EOS_TOKEN_ID})

    def prepare(self, messages: Sequence[Mapping[str, Any]], *, reasoning: bool,
                tools: Optional[Sequence[Mapping[str, Any]]] = None) -> PreparedPrompt:
        chunks: List[str] = []
        for message in messages:
            chunks.append(f"<{message['role']}>{message.get('content') or ''}"
                          f"</{message['role']}>")
            if message.get("tool_calls"):
                chunks.append("<tool_calls>"
                              + json.dumps(message["tool_calls"]) + "</tool_calls>")
        if tools:
            chunks.append("<tools>" + json.dumps(list(tools)) + "</tools>")
        text = "\n".join(chunks) + "\n<assistant>"
        return PreparedPrompt(token_ids=tuple(text.encode("utf-8")),
                              thinking_open=bool(reasoning))

    def decode(self, token_ids: Sequence[int]) -> str:
        return bytes(int(token) for token in token_ids).decode("utf-8", "replace")

    def reply_for(self, prompt: str) -> str:
        for keyword, reply in self.replies.items():
            if keyword in prompt:
                return reply
        return self.default_reply

    def new_session(self, loops: int) -> "FakeSession":
        return FakeSession(self, loops)

    def close(self) -> None:
        return None


class FakeSession:
    """A cache that remembers what it was fed, so reuse can be observed."""

    def __init__(self, engine: FakeEngine, loops: int):
        self.engine = engine
        self._loops = int(loops)
        self.fed: List[int] = []
        self._script: List[int] = []
        self._step = 0

    @property
    def loops(self) -> int:
        return self._loops

    def prefill(self, token_ids: Sequence[int], start: int) -> np.ndarray:
        self.engine.prefill_calls += 1
        self.engine.prefilled_tokens += len(token_ids)
        self.fed.extend(int(token) for token in token_ids)
        prompt = bytes(self.fed).decode("utf-8", "replace")
        self._script = list(self.engine.reply_for(prompt).encode("utf-8"))
        self._step = 0
        rows = np.full((len(token_ids), _VOCAB), _LOW, dtype=np.float32)
        rows[:, self._next_id()] = 0.0
        return rows

    def decode(self, token_id: int, pos: int) -> np.ndarray:
        self.engine.decoded_tokens += 1
        self.fed.append(int(token_id))
        self._step += 1
        row = np.full((_VOCAB,), _LOW, dtype=np.float32)
        row[self._next_id()] = 0.0
        return row

    def _next_id(self) -> int:
        return self._script[self._step] if self._step < len(self._script) else EOS_TOKEN_ID

    def set_loops(self, loops: int) -> bool:
        loops = int(loops)
        if loops == self._loops:
            return False
        self._loops = loops
        self.reset()
        return True

    def reset(self) -> None:
        self.fed.clear()
        self._script = []
        self._step = 0
        self.engine.reset_count += 1
        self.engine.reset_loops.append(self._loops)
