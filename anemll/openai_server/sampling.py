"""Sampling for the Dory OpenAI server (spec D10, ticket DANE-7).

NVIDIA's recommendation for Dory1.0-4B is temperature 1.0, top_p 0.95, top_k 0
and min_p 0 (the model card and `generation_config.json`); the spec adds greedy
decoding at temperature 0 so a test can reproduce a run exactly.  Those are the
defaults a request inherits, and every one of them can be overridden per
request.

The filter order is the one llama.cpp's sampler chain uses -- temperature, then
top_k, top_p and min_p on the same distribution, then a draw -- so a request
made against the GPU build means the same thing here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_TOP_K",
    "DEFAULT_MIN_P",
    "DEFAULT_MAX_TOKENS",
    "SamplingParams",
    "Sampler",
    "probabilities",
]

#: NVIDIA's recommended defaults for Dory1.0-4B (model card, generation_config).
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = 0
DEFAULT_MIN_P = 0.0

#: The server's default `max_tokens` when a request does not ask for one.
DEFAULT_MAX_TOKENS = 512


def _as_float(body: Mapping[str, Any], key: str, default: float) -> float:
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a number, got a boolean")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be a number, got {value!r}") from None


def _as_int(body: Mapping[str, Any], key: str, default: int) -> int:
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer, got a boolean")
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{key} must be an integer, got {value!r}") from None
    if parsed != float(value):
        raise ValueError(f"{key} must be an integer, got {value!r}")
    return parsed


@dataclass(frozen=True)
class SamplingParams:
    """One request's sampling settings, with the defaults already applied."""

    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    top_k: int = DEFAULT_TOP_K
    min_p: float = DEFAULT_MIN_P
    seed: Optional[int] = None
    max_tokens: int = DEFAULT_MAX_TOKENS
    stop: Tuple[str, ...] = ()

    @property
    def greedy(self) -> bool:
        """Temperature 0 is argmax -- reproducible, no draw at all (spec us #18)."""
        return self.temperature <= 0.0

    @classmethod
    def from_body(cls, body: Mapping[str, Any]) -> "SamplingParams":
        """Read the request body, applying NVIDIA's defaults to every absent field.

        `top_k` and `min_p` are not OpenAI fields; llama.cpp's server accepts
        them in the body and Studio may forward them, so they are read straight
        from the body rather than from a nested extension object.
        """
        if not isinstance(body, Mapping):
            raise ValueError("the request body must be a JSON object")
        temperature = _as_float(body, "temperature", DEFAULT_TEMPERATURE)
        top_p = _as_float(body, "top_p", DEFAULT_TOP_P)
        top_k = _as_int(body, "top_k", DEFAULT_TOP_K)
        min_p = _as_float(body, "min_p", DEFAULT_MIN_P)
        seed = body.get("seed")
        if seed is not None:
            seed = _as_int(body, "seed", 0)
        max_tokens = _as_int(body, "max_tokens", DEFAULT_MAX_TOKENS)
        stop = _stop_sequences(body.get("stop"))

        if temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {temperature}")
        if not 0.0 < top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")
        if top_k < 0:
            raise ValueError(f"top_k must be >= 0 (0 disables it), got {top_k}")
        if not 0.0 <= min_p <= 1.0:
            raise ValueError(f"min_p must be in [0, 1], got {min_p}")
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        return cls(temperature=temperature, top_p=top_p, top_k=top_k, min_p=min_p,
                   seed=seed, max_tokens=max_tokens, stop=stop)


def _stop_sequences(value: Any) -> Tuple[str, ...]:
    """`stop` is either one string or a list of strings in the OpenAI schema."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError(f"stop must be a string or a list of strings, got {item!r}")
            if item:
                out.append(item)
        return tuple(out)
    raise ValueError(f"stop must be a string or a list of strings, got {value!r}")


def probabilities(logits: Sequence[float], params: SamplingParams) -> np.ndarray:
    """The distribution a draw is taken from: float64, filtered, renormalised."""
    scores = np.asarray(logits, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        raise ValueError("the model returned no logits")
    finite = np.isfinite(scores)
    if not finite.all():
        # A single inf/NaN would poison every sum below; drop those tokens
        # instead of letting them win.
        if not finite.any():
            raise ValueError("the model returned no finite logit")
        scores = np.where(finite, scores, -np.inf)

    scaled = scores - scores.max()
    if params.greedy:
        out = np.zeros_like(scaled)
        out[int(np.argmax(scores))] = 1.0
        return out
    scaled /= params.temperature
    weights = np.exp(scaled)
    total = weights.sum()
    if not np.isfinite(total) or total <= 0.0:
        out = np.zeros_like(scaled)
        out[int(np.argmax(scores))] = 1.0
        return out
    probs = weights / total

    if params.top_k > 0 and params.top_k < probs.size:
        cutoff = np.argpartition(-probs, params.top_k - 1)[: params.top_k]
        keep = np.zeros_like(probs, dtype=bool)
        keep[cutoff] = True
        probs = np.where(keep, probs, 0.0)

    if 0.0 < params.top_p < 1.0:
        order = np.argsort(-probs, kind="stable")
        cumulative = np.cumsum(probs[order])
        # Keep the smallest prefix whose mass reaches top_p (llama.cpp's rule).
        keep_count = int(np.searchsorted(cumulative, params.top_p) + 1)
        keep = np.zeros_like(probs, dtype=bool)
        keep[order[:keep_count]] = True
        probs = np.where(keep, probs, 0.0)

    if params.min_p > 0.0:
        peak = probs.max()
        if peak > 0.0:
            probs = np.where(probs >= params.min_p * peak, probs, 0.0)

    total = probs.sum()
    if total <= 0.0:                     # filters removed everything: fall back
        out = np.zeros_like(probs)
        out[int(np.argmax(scores))] = 1.0
        return out
    return probs / total


class Sampler:
    """Turns logits rows into token ids, one draw per step.

    A seed makes a sampling run reproducible; without one the generator is
    seeded from the OS, because two requests that share a process-wide stream
    would surprise the second caller.
    """

    def __init__(self, params: SamplingParams, rng: Optional[np.random.Generator] = None):
        self.params = params
        self.rng = rng if rng is not None else np.random.default_rng(params.seed)

    def sample(self, logits: Sequence[float]) -> int:
        if self.params.greedy:
            scores = np.asarray(logits, dtype=np.float64).reshape(-1)
            if scores.size == 0:
                raise ValueError("the model returned no logits")
            return int(np.argmax(scores))
        probs = probabilities(logits, self.params)
        return int(self.rng.choice(probs.shape[0], p=probs))
