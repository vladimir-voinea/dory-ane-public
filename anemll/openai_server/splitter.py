"""Split streamed text into reasoning, content and tool calls.

Dory's chat template opens a thinking block in the prompt with reasoning on
(`<|im_start|>assistant\n thinking\n`), so the first tokens the model generates
are its reasoning, and it closes that block itself with `</think>`.  An OpenAI
client wants the two streams apart, and it wants them apart *while streaming*:
no chunk may leak half a marker into the text, and no fragment of a
`<tool_call>` block may reach the client as content.

So this is a small text state machine, kept free of tokenizer and engine code so
it can be tested on the exact sequences a tokenizer produces:

    reasoning -> (on </think>) content -> (on <tool_call>) tool -> content

`[TOOL_CALLS]`, the token in Dory's vocabulary for the older Hermes-style
payload, buffers the rest of the message and parses it when generation ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from anemll.openai_server.toolcalls import (
    CALLS_OPEN,
    TOOL_CALL_CLOSE,
    TOOL_CALL_OPEN,
    ParsedToolCall,
    parse_tool_calls,
)

__all__ = ["Delta", "DeltaSplitter", "REASONING_CLOSE", "REASONING_OPEN"]

#: The token that closes Dory's thinking block: id 13 in the BF16 vocabulary.
REASONING_OPEN = "<" + "think" + ">"
REASONING_CLOSE = "<" + "/" + "think" + ">"


def _partial_marker(text: str, marker: str) -> int:
    """Length of the longest proper prefix of `marker` that `text` ends with."""
    limit = min(len(marker) - 1, len(text))
    for size in range(limit, 0, -1):
        if marker.startswith(text[-size:]):
            return size
    return 0


_REASONING = "reasoning"
_CONTENT = "content"
_TOOL = "tool"
_CALLS_JSON = "calls_json"


@dataclass
class Delta:
    """One piece of an assistant message, in the OpenAI delta vocabulary."""

    reasoning: str = ""
    content: str = ""
    tool_calls: Tuple[ParsedToolCall, ...] = ()

    def is_empty(self) -> bool:
        return not (self.reasoning or self.content or self.tool_calls)


class DeltaSplitter:
    """Consume text as it arrives, emit only finished pieces.

    `thinking_open` is True when the prompt ended inside an open thinking block,
    which is what Dory's template does with reasoning on.  `tool_calls` is True
    only when the request carried tools: without them the model's markup is
    ordinary text and is passed through unchanged.
    """

    def __init__(self, *, thinking_open: bool = False, tool_calls: bool = False):
        self.reasoning_open = bool(thinking_open)
        self.parse_tool_calls = bool(tool_calls)
        self._state = _REASONING if self.reasoning_open else _CONTENT
        self._pending = ""
        self._buffered = ""

    # -- markers -----------------------------------------------------------
    def _markers(self) -> Tuple[Tuple[str, str], ...]:
        """(marker, the state it moves us to) for the two streaming states."""
        if self._state == _REASONING:
            markers: List[Tuple[str, str]] = [(REASONING_CLOSE, _CONTENT)]
            if self.parse_tool_calls:
                markers.append((TOOL_CALL_OPEN, _TOOL))
                markers.append((CALLS_OPEN, _CALLS_JSON))
            return tuple(markers)
        if self._state == _CONTENT and self.parse_tool_calls:
            return ((TOOL_CALL_OPEN, _TOOL), (CALLS_OPEN, _CALLS_JSON))
        return ()

    def _holdback(self, text: str) -> int:
        """Trailing characters that may still turn out to be a marker."""
        return max((_partial_marker(text, marker) for marker, _ in self._markers()),
                   default=0)

    def _find_marker(self) -> Optional[Tuple[int, str, str]]:
        best: Optional[Tuple[int, str, str]] = None
        for marker, target in self._markers():
            index = self._pending.find(marker)
            if index >= 0 and (best is None or index < best[0]):
                best = (index, marker, target)
        return best

    def _emit(self, text: str) -> List[Delta]:
        if not text:
            return []
        if self._state == _REASONING:
            return [Delta(reasoning=text)]
        return [Delta(content=text)]

    # -- feeding -----------------------------------------------------------
    def feed(self, text: str) -> List[Delta]:
        """Add text; return the deltas that are now safe to send."""
        if not text:
            return []
        self._pending += text
        out: List[Delta] = []
        while True:
            if self._state == _CALLS_JSON:
                self._buffered += self._pending
                self._pending = ""
                break
            if self._state == _TOOL:
                end = self._pending.find(TOOL_CALL_CLOSE)
                if end < 0:
                    # Buffer the block, but keep a partial </tool_call> marker
                    # in hand: the close arrives in pieces like everything else.
                    keep = _partial_marker(self._pending, TOOL_CALL_CLOSE)
                    cut = len(self._pending) - keep
                    self._buffered += self._pending[:cut]
                    self._pending = self._pending[cut:]
                    break
                self._buffered += self._pending[:end]
                self._pending = self._pending[end + len(TOOL_CALL_CLOSE):]
                block = TOOL_CALL_OPEN + self._buffered + TOOL_CALL_CLOSE
                self._buffered = ""
                self._state = _CONTENT
                calls = tuple(parse_tool_calls(block))
                if calls:
                    out.append(Delta(tool_calls=calls))
                continue
            found = self._find_marker()
            if found is None:
                break
            index, marker, target = found
            if index > 0:
                out.extend(self._emit(self._pending[:index]))
            self._pending = self._pending[index + len(marker):]
            self._state = target
            if target == _TOOL:
                self._buffered = ""

        if self._state in (_REASONING, _CONTENT):
            keep = self._holdback(self._pending)
            if len(self._pending) > keep:
                cut = len(self._pending) - keep
                out.extend(self._emit(self._pending[:cut]))
                self._pending = self._pending[cut:]
        return out

    def finish(self) -> List[Delta]:
        """Flush what is left when generation stops mid-stream."""
        if self._state == _CALLS_JSON:
            calls = tuple(parse_tool_calls(CALLS_OPEN + self._buffered))
            self._buffered = ""
            self._pending = ""
            self._state = _CONTENT
            return [Delta(tool_calls=calls)] if calls else []
        if self._state == _TOOL:
            # An unterminated <tool_call>: keep the text as content, because a
            # truncated call is not a call.
            text = TOOL_CALL_OPEN + self._buffered + self._pending
            self._buffered = ""
            self._pending = ""
            self._state = _CONTENT
            return self._emit(text)
        out = self._emit(self._pending)
        self._pending = ""
        return out
