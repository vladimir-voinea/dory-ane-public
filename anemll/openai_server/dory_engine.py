"""The real engine: spec D9's runtime behind the server's seam.

`DoryRuntime` in `anemll/dory_chat.py` already does the model work -- the five
Core ML components in run order, the recurrent component called once per loop,
one `MLState` shared by all of them.  What this module adds is only what the
server needs and the CLI does not:

* a prompt prepared from OpenAI messages, including tools, through the model's
  own chat template;
* the reasoning marker the stream splitter needs (`thinking_open`), read off the
  rendered prompt rather than guessed;
* a session object the server can drive, whose loop-count range comes from the
  meta file.

`coremltools` is imported here, not in `engine.py`, so the server's other
modules stay importable -- and testable -- without a Core ML install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from anemll.openai_server.engine import EngineInfo, PreparedPrompt
from anemll.openai_server.splitter import REASONING_CLOSE, REASONING_OPEN

__all__ = ["DoryEngine", "DorySession", "thinking_open"]

_ASSISTANT = "<|im_start|>assistant"


def thinking_open(text: str) -> bool:
    """Does the rendered prompt leave the assistant inside an open thinking block?

    Dory's template opens one with reasoning on
    (`<|im_start|>assistant\n thinking\n`) and closes it itself when reasoning is
    off (`<|im_start|>assistant\n</think>`), so the question is whether the last
    assistant turn has an opening marker with no `</think>` after it.
    """
    tail = text[text.rfind(_ASSISTANT):] if _ASSISTANT in text else text
    return REASONING_OPEN in tail and REASONING_CLOSE not in tail


class DorySession:
    """One cache, driven the way the server drives it."""

    def __init__(self, runtime):
        self.runtime = runtime

    @property
    def loops(self) -> int:
        return int(self.runtime.loops)

    def prefill(self, token_ids: Sequence[int], start: int):
        return self.runtime.prefill([int(token) for token in token_ids], int(start))

    def decode(self, token_id: int, pos: int):
        return self.runtime.decode(int(token_id), int(pos))

    def set_loops(self, loops: int) -> bool:
        return bool(self.runtime.set_loops(int(loops)))

    def reset(self) -> None:
        self.runtime.reset_state()


class DoryEngine:
    """A converted model directory, loaded once, served many times."""

    def __init__(self, model_dir, *, model_id: str = "dory-ane",
                 compute_units=None, verbose: bool = False):
        from anemll.dory_chat import ChatTokenizer, DoryRuntime

        self.model_dir = Path(model_dir)
        self.runtime = DoryRuntime(self.model_dir, compute_units=compute_units,
                                   verbose=verbose)
        self.meta = self.runtime.meta
        self.tokenizer = ChatTokenizer(self.model_dir, self.meta)
        params = self.runtime.params
        loops = self.meta["loops"]
        tokenizer_meta = self.meta.get("tokenizer") or {}
        self.info = EngineInfo(
            model_id=model_id,
            context_length=int(params["context_length"]),
            loop_min=int(loops["min"]),
            loop_max=int(loops["max"]),
            loop_default=int(loops["default"]),
            vocab_size=int(params["vocab_size"]),
            display_name=str((self.meta.get("model_info") or {}).get("name") or model_id),
            kv_mode=str((self.meta.get("kv") or {}).get("mode") or "last_loop"),
            reasoning_default=bool(tokenizer_meta.get("reasoning_default", True)),
        )

    # -- engine ----------------------------------------------------
    @property
    def eos_ids(self) -> frozenset:
        return frozenset(int(token) for token in self.tokenizer.eos_ids)

    def prepare(self, messages: Sequence[Mapping[str, Any]], *, reasoning: bool,
                tools: Optional[Sequence[Mapping[str, Any]]] = None) -> PreparedPrompt:
        text = self.tokenizer.apply_chat_template(messages, reasoning=reasoning,
                                                  tools=list(tools) if tools else None)
        token_ids = self.tokenizer.encode(text)
        return PreparedPrompt(token_ids=tuple(int(token) for token in token_ids),
                              thinking_open=thinking_open(text))

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def new_session(self, loops: int) -> DorySession:
        session = DorySession(self.runtime)
        session.set_loops(int(loops))
        return session

    def close(self) -> None:
        """Drop the models and the state; the next load pays the compile again."""
        self.runtime.models.clear()
        self.runtime._state = None

    def stats(self) -> dict:
        return self.runtime.stats()
