"""The chat-completions service: one model, one cache, one request at a time.

Everything between "a JSON body arrived" and "deltas come out" lives here, and
none of it knows about HTTP.  That split is what lets the whole generation path
-- prompt preparation, cache reuse, sampling, reasoning/tool-call splitting,
stop sequences, the loop-count choice -- be tested against a scripted engine
without a Core ML model, and driven by `aiohttp` in production.

Two behaviours are worth calling out because the ticket asks for them:

* **the loop count comes from the alias or from the body field**, is validated
  against the model's range, is logged per request and is reported in every
  response's `dory` metadata;
* **a loop change resets the cache** -- spec D4: older rows keep the previous
  loop count's K/V, so keeping them would answer from a cache built at another
  depth.  The reset, its reason and the new cache epoch are logged and reported.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from anemll.openai_server.engine import Engine, PreparedPrompt, Session
from anemll.openai_server.messages import to_template_messages
from anemll.openai_server.models import ModelCatalog, UnknownModelError
from anemll.openai_server.sampling import Sampler, SamplingParams
from anemll.openai_server.splitter import Delta, DeltaSplitter
from anemll.openai_server.toolcalls import ParsedToolCall

__all__ = [
    "ChatError",
    "ChatService",
    "Completion",
    "RequestPlan",
    "Started",
    "common_prefix",
]

LOGGER = logging.getLogger("anemll.openai_server")

_EPS = 1e-9


class ChatError(Exception):
    """A request the server refuses, with the status an OpenAI client expects."""

    def __init__(self, message: str, *, status: int = 400,
                 code: Optional[str] = None, kind: str = "invalid_request_error"):
        super().__init__(message)
        self.message = message
        self.status = status
        self.code = code
        self.kind = kind


@dataclass(frozen=True)
class RequestPlan:
    """A validated request, before any model work has happened."""

    request_id: str
    model_id: str
    prepared: PreparedPrompt
    loops: int
    loop_source: str
    sampling: SamplingParams
    reasoning: bool
    parse_tool_calls: bool
    stream: bool


@dataclass(frozen=True)
class Started:
    """Sent once, when the model has the prompt and is about to generate."""

    request_id: str
    model_id: str
    loops: int
    loop_source: str
    cache_reset: bool
    cache_epoch: int
    prompt_tokens: int
    cached_tokens: int
    max_tokens: int
    waited_seconds: float


@dataclass(frozen=True)
class Completion:
    """The end of a generation: why it stopped, and what came out."""

    request_id: str
    model_id: str
    loops: int
    loop_source: str
    cache_reset: bool
    cache_epoch: int
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning: str
    content: str
    tool_calls: Tuple[ParsedToolCall, ...]
    seconds: float

    @property
    def decode_tokens_per_second(self) -> float:
        return self.completion_tokens / self.seconds if self.seconds > 0 else 0.0


StreamItem = Union[Started, Delta, Completion]


def common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    """How many leading tokens two sequences share."""
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


def _find_stop(buffer: str, stops: Sequence[str]) -> int:
    """Index of the earliest stop sequence in `buffer`, or -1."""
    earliest = -1
    for stop in stops:
        index = buffer.find(stop)
        if index >= 0 and (earliest < 0 or index < earliest):
            earliest = index
    return earliest


def _int_field(body: Mapping[str, Any],
               names: Sequence[str]) -> Tuple[Optional[int], Optional[str]]:
    """The first of `names` present in the body, validated as an int."""
    for name in names:
        if name in body and body[name] is not None:
            value = body[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ChatError(f"{name} must be an integer, got {value!r}",
                                code="invalid_value")
            if int(value) != value:
                raise ChatError(f"{name} must be an integer, got {value!r}",
                                code="invalid_value")
            return int(value), name
    return None, None


def _enable_thinking(body: Mapping[str, Any]) -> Optional[bool]:
    """The reasoning switch: vLLM's `chat_template_kwargs.enable_thinking` first."""
    kwargs = body.get("chat_template_kwargs")
    if isinstance(kwargs, Mapping) and "enable_thinking" in kwargs:
        return bool(kwargs["enable_thinking"])
    if "enable_thinking" in body:
        return bool(body["enable_thinking"])
    return None


class ChatService:
    """Owns the engine, the cache and the lock that serialises requests."""

    def __init__(self, engine: Engine, catalog: Optional[ModelCatalog] = None,
                 *, logger: Optional[logging.Logger] = None):
        self.engine = engine
        self.catalog = catalog or ModelCatalog(engine.info)
        self.log = logger or LOGGER
        self._lock = threading.Lock()
        self._session: Optional[Session] = None
        self._cached: List[int] = []
        self._epoch = 0
        self._requests = 0

    # -- planning ----------------------------------------------------------
    def plan(self, body: Mapping[str, Any]) -> RequestPlan:
        """Validate a request body and prepare its prompt; no model work yet."""
        if not isinstance(body, Mapping):
            raise ChatError("the request body must be a JSON object")
        if "messages" not in body:
            raise ChatError("messages is required", code="missing_messages")
        try:
            messages = to_template_messages(body["messages"])
        except ValueError as error:
            raise ChatError(str(error)) from None

        info = self.engine.info
        model_id = body.get("model") or self.catalog.base_id
        if not isinstance(model_id, str):
            raise ChatError(f"model must be a string, got {model_id!r}")
        try:
            self.catalog.entry(model_id)
        except UnknownModelError as error:
            raise ChatError(str(error), status=404, code="model_not_found") from None

        alias_loops = self.catalog.loops_for(model_id)
        body_loops, body_field = _int_field(body, ("dory_loops", "loops"))
        if alias_loops is not None and body_loops is not None and alias_loops != body_loops:
            raise ChatError(
                f"the model {model_id} fixes the loop count at {alias_loops}, but "
                f"{body_field}={body_loops}; use one or the other",
                code="conflicting_loops")
        loops = alias_loops if alias_loops is not None else (
            body_loops if body_loops is not None else info.loop_default)
        source = "alias" if alias_loops is not None else (
            "body" if body_loops is not None else "default")
        if not info.loop_min <= loops <= info.loop_max:
            raise ChatError(
                f"{body_field or 'loops'} must be between {info.loop_min} and "
                f"{info.loop_max}, got {loops}", code="invalid_loops")

        try:
            sampling = SamplingParams.from_body(body)
        except ValueError as error:
            raise ChatError(str(error), code="invalid_sampling") from None

        tools = body.get("tools") or []
        if tools and not isinstance(tools, Sequence):
            raise ChatError("tools must be a list", code="invalid_tools")
        reasoning = _enable_thinking(body)
        if reasoning is None:
            reasoning = info.reasoning_default
        try:
            prepared = self.engine.prepare(messages, reasoning=bool(reasoning),
                                           tools=tools or None)
        except ValueError as error:
            raise ChatError(str(error), code="invalid_prompt") from None

        stream = bool(body.get("stream", False))
        self._requests += 1
        request_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        return RequestPlan(request_id=request_id, model_id=model_id, prepared=prepared,
                           loops=loops, loop_source=source, sampling=sampling,
                           reasoning=bool(reasoning), parse_tool_calls=bool(tools),
                           stream=stream)

    # -- cache -------------------------------------------------------------
    def _get_session(self) -> Session:
        if self._session is None:
            self._session = self.engine.new_session(self.engine.info.loop_default)
            self.log.info("session opened: loops=%d epoch=%d",
                          self._session.loops, self._epoch)
        return self._session

    def _acquire(self, request_id: str) -> float:
        """Take the model lock; return how long the request had to wait."""
        started = time.time()
        if not self._lock.acquire(blocking=False):
            self.log.info("request %s is waiting: another request holds the model",
                          request_id)
            self._lock.acquire()
        return time.time() - started

    def _reset(self, session: Session, reason: str) -> None:
        session.reset()
        self._cached = []
        self._epoch += 1
        self.log.info("cache reset (%s): new epoch %d", reason, self._epoch)

    # -- generation --------------------------------------------------------
    def stream(self, plan: RequestPlan) -> Iterator[StreamItem]:
        """Run one request: yields Started, then Deltas, then a Completion."""
        waited = self._acquire(plan.request_id)
        try:
            yield from self._stream_locked(plan, waited)
        finally:
            self._lock.release()

    def _stream_locked(self, plan: RequestPlan, waited: float) -> Iterator[StreamItem]:
        info = self.engine.info
        session = self._get_session()
        cache_reset = False

        if session.loops != plan.loops:
            # Spec D4: the previous loop count's K/V is still in the cache, so a
            # new loop count has to start from an empty one.
            previous = session.loops
            changed = session.set_loops(plan.loops)
            cache_reset = changed or cache_reset
            if changed:
                self._cached = []
                self._epoch += 1
                self.log.info(
                    "loop count %d -> %d (from %s): cache reset, epoch %d",
                    previous, session.loops, plan.loop_source, self._epoch)

        prompt = list(plan.prepared.token_ids)
        if not prompt:
            raise ChatError("the prompt is empty", code="empty_prompt")
        if len(prompt) >= info.context_length:
            raise ChatError(
                f"the prompt is {len(prompt)} tokens but the context is "
                f"{info.context_length}; shorten the conversation",
                code="context_length_exceeded")
        available = info.context_length - len(prompt) - 1
        limit = min(plan.sampling.max_tokens, available)
        if limit < 1:
            raise ChatError(
                f"only {available} tokens of context are left for the answer; "
                f"shorten the prompt", code="context_length_exceeded")

        common = common_prefix(prompt, self._cached)
        if common < len(self._cached):
            self._reset(session, "the prompt is not an extension of the cached one")
            cache_reset = True
            common = 0
        new_ids = prompt[common:]
        if not new_ids:
            # The cache already holds this exact prompt (or more): re-feed it,
            # because the next token needs a forward pass over the last token.
            self._reset(session, "the prompt is already in the cache")
            cache_reset = True
            common = 0
            new_ids = prompt

        started = time.time()
        rows = np.asarray(session.prefill(new_ids, common))
        if rows.ndim != 2 or rows.shape[0] != len(new_ids):
            raise ChatError(
                f"the engine fed {rows.shape[0] if rows.ndim else 0} rows for "
                f"{len(new_ids)} tokens", status=500, code="engine_error",
                kind="server_error")
        logits = np.asarray(rows[-1], dtype=np.float64).reshape(-1)
        self._cached = list(prompt)
        prefill_seconds = time.time() - started

        yield Started(request_id=plan.request_id, model_id=plan.model_id,
                      loops=session.loops, loop_source=plan.loop_source,
                      cache_reset=cache_reset, cache_epoch=self._epoch,
                      prompt_tokens=len(prompt), cached_tokens=common,
                      max_tokens=limit, waited_seconds=waited)
        self.log.info(
            "request %s model=%s loops=%d source=%s cache=%s epoch=%d "
            "prompt_tokens=%d cached=%d prefill=%.2fs max_tokens=%d stream=%s "
            "temperature=%s top_p=%s top_k=%d min_p=%s reasoning=%s tools=%s",
            plan.request_id, plan.model_id, session.loops, plan.loop_source,
            "reset" if cache_reset else ("reused" if common else "fresh"),
            self._epoch, len(prompt), common,
            prefill_seconds, limit, plan.stream, plan.sampling.temperature,
            plan.sampling.top_p, plan.sampling.top_k, plan.sampling.min_p,
            plan.reasoning, plan.parse_tool_calls)

        sampler = Sampler(plan.sampling)
        splitter = DeltaSplitter(thinking_open=plan.prepared.thinking_open,
                                 tool_calls=plan.parse_tool_calls)
        eos = self.engine.eos_ids
        stops = plan.sampling.stop
        holdback = max((len(stop) for stop in stops), default=1) - 1
        buffer = ""
        generated: List[int] = []
        emitted = 0
        reasoning_parts: List[str] = []
        content_parts: List[str] = []
        call_parts: List[ParsedToolCall] = []
        position = len(prompt)
        finish_reason = "length"
        generation_started = time.time()

        while True:
            if len(generated) >= limit:
                finish_reason = "length"
                break
            token = sampler.sample(logits)
            if token in eos:
                finish_reason = "stop"
                break
            generated.append(token)
            text = self.engine.decode(generated)
            if len(text) > emitted:
                buffer += text[emitted:]
                emitted = len(text)

            stopped = False
            if stops:
                found = _find_stop(buffer, stops)
                if found >= 0:
                    buffer = buffer[:found]
                    stopped = True
            visible = buffer
            if not stopped and holdback > 0:
                keep = min(holdback, len(buffer))
                visible = buffer[: len(buffer) - keep]
                buffer = buffer[len(buffer) - keep:]
            else:
                buffer = ""

            for delta in splitter.feed(visible):
                if delta.reasoning:
                    reasoning_parts.append(delta.reasoning)
                if delta.content:
                    content_parts.append(delta.content)
                if delta.tool_calls:
                    call_parts.extend(delta.tool_calls)
                yield delta
            if stopped:
                finish_reason = "stop"
                break
            if len(generated) >= limit:
                finish_reason = "length"
                break
            logits = np.asarray(session.decode(token, position), dtype=np.float64).reshape(-1)
            position += 1
            self._cached.append(token)

        for delta in splitter.finish():
            if delta.reasoning:
                reasoning_parts.append(delta.reasoning)
            if delta.content:
                content_parts.append(delta.content)
            if delta.tool_calls:
                call_parts.extend(delta.tool_calls)
            yield delta

        if call_parts:
            finish_reason = "tool_calls"
        completion = Completion(
            request_id=plan.request_id, model_id=plan.model_id, loops=session.loops,
            loop_source=plan.loop_source, cache_reset=cache_reset,
            cache_epoch=self._epoch, finish_reason=finish_reason,
            prompt_tokens=len(prompt), completion_tokens=len(generated),
            cached_tokens=common, reasoning="".join(reasoning_parts),
            content="".join(content_parts), tool_calls=tuple(call_parts),
            seconds=max(time.time() - generation_started, _EPS))
        self.log.info(
            "request %s finished: reason=%s loops=%d completion_tokens=%d "
            "reasoning_chars=%d content_chars=%d tool_calls=%d %.2fs (%.1f tok/s)",
            plan.request_id, completion.finish_reason, completion.loops,
            completion.completion_tokens, len(completion.reasoning),
            len(completion.content), len(completion.tool_calls),
            completion.seconds, completion.decode_tokens_per_second)
        yield completion

    def complete(self, plan: RequestPlan) -> Tuple[Started, Completion]:
        """Run a request to the end and hand back its two boundary events."""
        started: Optional[Started] = None
        completion: Optional[Completion] = None
        for item in self.stream(plan):
            if isinstance(item, Started):
                started = item
            elif isinstance(item, Completion):
                completion = item
        if started is None or completion is None:
            raise ChatError("the generation ended without a result", status=500,
                            code="engine_error", kind="server_error")
        return started, completion

    def close(self) -> None:
        self.engine.close()
