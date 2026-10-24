"""OpenAI chat messages in, Dory template messages out.

The two speak almost the same language, with three differences that matter:

* an assistant message's `tool_calls[].function.arguments` is a JSON *string* in
  OpenAI's schema and a *mapping* in Dory's template (which loops over it with
  `|items`), so it is parsed here;
* `content` may be a list of parts instead of a string;
* an assistant turn may carry `reasoning_content`, which the template renders
  back into a ` thinking` block so the model sees its own reasoning in history.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence

__all__ = ["to_template_messages", "flatten_content"]

_ROLES = {"system", "user", "assistant", "tool"}


def flatten_content(content: Any) -> str:
    """A message's text, whether it is a string or a list of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        pieces: List[str] = []
        for part in content:
            if isinstance(part, str):
                pieces.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    pieces.append(text)
                elif part.get("type") not in (None, "text"):
                    raise ValueError(
                        f"content parts of type {part.get('type')!r} are not supported; "
                        "send text")
        return "".join(pieces)
    raise ValueError(f"message content must be a string or a list of parts, got {content!r}")


def _tool_arguments(raw: Any, index: int) -> Dict[str, Any]:
    """`arguments` as the template needs it: a mapping."""
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            decoded = json.loads(text)
        except (TypeError, ValueError):
            raise ValueError(
                f"tool_calls[{index}].function.arguments is not valid JSON: {raw!r}")
        if not isinstance(decoded, Mapping):
            raise ValueError(
                f"tool_calls[{index}].function.arguments must be a JSON object, got "
                f"{type(decoded).__name__}")
        return dict(decoded)
    raise ValueError(f"tool_calls[{index}].function.arguments must be a JSON object")


def to_template_messages(messages: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Convert an OpenAI `messages` array into template messages.

    Unknown roles are passed through: Dory's template has a catch-all branch, and
    refusing a role the model can render would be this server's opinion rather
    than the template's.
    """
    if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
        raise ValueError("messages must be a list")
    if not messages:
        raise ValueError("messages must not be empty")
    out: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"messages[{index}] must be an object")
        role = message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"messages[{index}].role is required")
        converted: Dict[str, Any] = {"role": role,
                                     "content": flatten_content(message.get("content"))}
        if role == "assistant":
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                if not isinstance(tool_calls, Sequence) or isinstance(tool_calls, (str, bytes)):
                    raise ValueError(f"messages[{index}].tool_calls must be a list")
                converted["tool_calls"] = []
                for call_index, call in enumerate(tool_calls):
                    if not isinstance(call, Mapping):
                        raise ValueError(
                            f"messages[{index}].tool_calls[{call_index}] must be an object")
                    nested = call.get("function")
                    function = nested if isinstance(nested, Mapping) else call
                    name = function.get("name")
                    if not isinstance(name, str) or not name:
                        raise ValueError(
                            f"messages[{index}].tool_calls[{call_index}].function.name is required")
                    converted["tool_calls"].append({
                        "name": name,
                        "arguments": _tool_arguments(function.get("arguments"), call_index),
                    })
            reasoning = message.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning.strip():
                converted["reasoning_content"] = reasoning
        out.append(converted)
    return out
