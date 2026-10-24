"""Model ids and the loop-count aliases (spec D10 / us #21, ticket DANE-7).

The loop count is the one thing about this model a client may want to choose
without sending a custom body field, so every loop count has a model id of its
own: `dory-ane` (the default) plus `dory-ane-1loop` ... `dory-ane-4loops`.
The models list reports each one's loop count, the context length and the
supported range, so a client can size a prompt without reading the docs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from anemll.openai_server.engine import EngineInfo

__all__ = ["ModelEntry", "ModelCatalog", "UnknownModelError", "loop_alias"]

_ALIAS = re.compile(r"^(?P<base>.+?)-(?P<count>\d+)(?:loop|loops)$")


class UnknownModelError(ValueError):
    """The request named a model id this server does not serve."""


def loop_alias(base_id: str, loops: int) -> str:
    """`dory-ane` + 2 -> `dory-ane-2loops` (and `1loop`, singular)."""
    return f"{base_id}-{int(loops)}loop" if int(loops) == 1 else f"{base_id}-{int(loops)}loops"


@dataclass(frozen=True)
class ModelEntry:
    """One id in the models list."""

    id: str
    loops: int
    is_alias: bool


class ModelCatalog:
    """Every id this server answers to, and the loop count each one means."""

    def __init__(self, info: EngineInfo, created: Optional[int] = None):
        self.info = info
        self.created = created
        self.base_id = info.model_id
        self._entries: List[ModelEntry] = [ModelEntry(info.model_id, info.loop_default, False)]
        for loops in range(info.loop_min, info.loop_max + 1):
            self._entries.append(ModelEntry(loop_alias(info.model_id, loops), loops, True))
        self._by_id: Dict[str, ModelEntry] = {entry.id: entry for entry in self._entries}

    @property
    def entries(self) -> List[ModelEntry]:
        return list(self._entries)

    @property
    def ids(self) -> List[str]:
        return [entry.id for entry in self._entries]

    def entry(self, model_id: Optional[str]) -> ModelEntry:
        """The entry a request names, or UnknownModelError with the valid ids."""
        if not model_id:
            return self._by_id[self.base_id]
        if model_id in self._by_id:
            return self._by_id[model_id]
        raise UnknownModelError(
            f"the model {model_id!r} is not served here; the models are "
            + ", ".join(self.ids))

    def loops_for(self, model_id: Optional[str]) -> Optional[int]:
        """The loop count an id implies, or None for the plain model id."""
        entry = self.entry(model_id)
        return entry.loops if entry.is_alias else None

    def payload(self) -> dict:
        """The `GET /v1/models` body.

        `context_length` and `max_model_len` carry the same number on purpose:
        the first is what ANEMLL's own tooling reads, the second is what OpenAI
        compatible clients and vLLM expect.  The `dory` object is the part that
        is specific to this port -- the loop count, the range it may take, and
        the cache behaviour a client should know about.
        """
        info = self.info
        data = []
        for entry in self._entries:
            data.append({
                "id": entry.id,
                "object": "model",
                "created": self.created or 0,
                "owned_by": "anemll",
                "root": info.model_id,
                "parent": None if not entry.is_alias else info.model_id,
                "permission": [],
                "context_length": info.context_length,
                "max_model_len": info.context_length,
                "dory": {
                    "model": info.model_id,
                    "loops": entry.loops,
                    "loop_range": [info.loop_min, info.loop_max],
                    "default_loops": info.loop_default,
                    "kv_mode": info.kv_mode,
                    "loop_change_invalidates_cache": True,
                },
            })
        return {"object": "list", "data": data}
