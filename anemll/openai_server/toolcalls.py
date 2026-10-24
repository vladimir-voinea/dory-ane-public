"""Dory's native tool-call format, parsed into the OpenAI shape.

Dory's template (the checkpoint's `chat_template.jinja`, and NVIDIA's model card)
teaches the model to answer a call like this, and nothing else:

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Santa Clara
    </parameter>
    </function>
    </tool_call>

An OpenAI client expects `tool_calls: [{id, type: "function", function: {name,
arguments}}]` with `arguments` a JSON *string*, so this module turns one into
the other.  A parameter value is JSON when it parses as JSON and a plain string
otherwise -- the template writes non-scalar values with `tojson` and scalars
as-is, so this round-trips both.

The older Hermes-style `[TOOL_CALLS][{...}]` payload (the `[TOOL_CALLS]`
token is in Dory's vocabulary) is accepted as a fallback, in case a fine-tune or
a prompt pushes the model towards it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = [
    "TOOL_CALL_OPEN",
    "TOOL_CALL_CLOSE",
    "CALLS_OPEN",
    "ParsedToolCall",
    "parse_tool_calls",
    "split_tool_calls",
    "arguments_json",
]

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"
CALLS_OPEN = "[TOOL_CALLS]"

_BLOCK = re.compile(re.escape(TOOL_CALL_OPEN) + r"(.*?)" + re.escape(TOOL_CALL_CLOSE), re.S)
_FUNCTION = re.compile(r"<function=([^>]*?)>(.*?)</function>", re.S)
_PARAMETER = re.compile(r"<parameter=([^>]*?)>(.*?)</parameter>", re.S)


@dataclass(frozen=True)
class ParsedToolCall:
    """One call: a function name and its arguments, already typed."""

    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def arguments_json(self) -> str:
        """The `arguments` field OpenAI clients receive: a JSON object string."""
        return json.dumps(self.arguments, ensure_ascii=False)


def arguments_json(arguments: Dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False)


def _strip_newlines(text: str) -> str:
    """The template puts each value on its own lines: `<parameter=x>\nvalue\n`."""
    if text.startswith("\r\n"):
        text = text[2:]
    elif text.startswith("\n"):
        text = text[1:]
    if text.endswith("\r\n"):
        text = text[:-2]
    elif text.endswith("\n"):
        text = text[:-1]
    return text


def _typed(value: str) -> Any:
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _parse_function_block(block: str) -> List[ParsedToolCall]:
    """`<function=name><parameter=k>v</parameter>...</function>` -> ParsedToolCall."""
    calls = []
    for match in _FUNCTION.finditer(block):
        name = match.group(1).strip()
        if not name:
            continue
        arguments: Dict[str, Any] = {}
        for parameter in _PARAMETER.finditer(match.group(2)):
            key = parameter.group(1).strip()
            if not key:
                continue
            arguments[key] = _typed(_strip_newlines(parameter.group(2)))
        calls.append(ParsedToolCall(name=name, arguments=arguments))
    return calls


def _parse_json_payload(payload: str) -> List[ParsedToolCall]:
    """`[TOOL_CALLS][{"name": ..., "arguments": {...}}]` -> ParsedToolCall."""
    payload = payload.strip()
    if not payload:
        return []
    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError):
        return []
    if isinstance(decoded, dict):
        decoded = [decoded]
    if not isinstance(decoded, list):
        return []
    calls = []
    for item in decoded:
        if not isinstance(item, dict):
            continue
        function = item.get("function") if isinstance(item.get("function"), dict) else item
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except (TypeError, ValueError):
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(ParsedToolCall(name=name, arguments=arguments))
    return calls


def _parse_calls_block(block: str) -> List[ParsedToolCall]:
    """One `<tool_call>...</tool_call>` body."""
    calls = _parse_function_block(block)
    if calls:
        return calls
    return _parse_json_payload(block)


def parse_tool_calls(text: str) -> List[ParsedToolCall]:
    """Every tool call in `text`, in the order the model emitted them.

    Only text the model marked up is read: no `<tool_call>` (or
    `[TOOL_CALLS]`) marker means no calls, so ordinary prose that happens to
    mention a function is left alone.
    """
    if not text:
        return []
    calls: List[ParsedToolCall] = []
    for match in _BLOCK.finditer(text):
        calls.extend(_parse_calls_block(match.group(1)))
    if calls:
        return calls
    if CALLS_OPEN in text:
        tail = text.split(CALLS_OPEN, 1)[1]
        calls = _parse_json_payload(tail.split("</tool_call>", 1)[0])
        if calls:
            return calls
        calls = _parse_function_block(tail)
        if calls:
            return calls
    return []


def split_tool_calls(text: str) -> Tuple[str, List[ParsedToolCall]]:
    """`text` without its tool-call blocks, plus the calls those blocks held."""
    calls = parse_tool_calls(text)
    if not calls:
        return text, []
    remainder = _BLOCK.sub("", text)
    if CALLS_OPEN in text and not _BLOCK.search(text):
        remainder = text.split(CALLS_OPEN, 1)[0]
    return remainder.strip("\n"), calls
