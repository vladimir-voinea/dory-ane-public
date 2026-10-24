"""The OpenAI-compatible HTTP surface (spec D10, ticket DANE-7).

Two endpoints are required -- `GET /v1/models` and `POST /v1/chat/completions`
-- plus `GET /health` for scripts.  Chat completions stream by default in
OpenAI's SSE dialect: `data: {chunk}\n\n` per piece, `data: [DONE]\n\n` at the
end, reasoning in `delta.reasoning_content`, tool calls in `delta.tool_calls`.

Generation is synchronous -- one Core ML prediction at a time -- so it runs in a
worker thread and its events are handed to the event loop, which keeps the
server answering `/v1/models` and `/health` while a 24K prompt prefills.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, AsyncIterator, Dict, Optional

from aiohttp import web

from anemll.openai_server.models import ModelCatalog
from anemll.openai_server.service import (
    ChatError,
    ChatService,
    Completion,
    Delta,
    RequestPlan,
    Started,
)

__all__ = ["create_app", "ThreadedStream", "chunk_body", "completion_body"]

LOGGER = logging.getLogger("anemll.openai_server")

SERVICE = web.AppKey("service", ChatService)
CATALOG = web.AppKey("catalog", ModelCatalog)

_DONE = object()


class ThreadedStream:
    """Run a blocking generator in a worker thread, deliver its items to asyncio.

    Items arrive in order through an `asyncio.Queue`.  Closing the async side
    (a client hung up) sets a flag the worker checks between items, so a
    disconnected request stops using the model at the next token instead of
    running to completion.
    """

    def __init__(self, factory):
        self._factory = factory
        self._loop = asyncio.get_running_loop()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dory-generation",
                                        daemon=True)

    def _run(self) -> None:
        try:
            generator = self._factory()
            try:
                for item in generator:
                    if self._stop.is_set():
                        break
                    self._loop.call_soon_threadsafe(self._queue.put_nowait, item)
            finally:
                generator.close()
        except BaseException as error:            # forwarded to the event loop
            self._loop.call_soon_threadsafe(self._queue.put_nowait, error)
        finally:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, _DONE)

    async def drain(self) -> AsyncIterator[Any]:
        self._thread.start()
        try:
            while True:
                item = await self._queue.get()
                if item is _DONE:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self._stop.set()
            if self._thread.is_alive():
                await asyncio.to_thread(self._thread.join, 30)


def _headers() -> Dict[str, str]:
    return {
        "Content-Type": "text/event-stream; charset=utf-8",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


def _fingerprint(model_id: str, loops: int) -> str:
    return f"{model_id}-loops{loops}"


def _dory_block(*, loops: int, source: str, cache_reset: Optional[bool] = None,
                cache_epoch: Optional[int] = None,
                prompt_tokens: Optional[int] = None,
                cached_tokens: Optional[int] = None) -> Dict[str, Any]:
    """The metadata that makes this port's choices visible to a client."""
    block: Dict[str, Any] = {"loops": loops, "loops_source": source}
    if cache_reset is not None:
        block["cache_reset"] = cache_reset
    if cache_epoch is not None:
        block["cache_epoch"] = cache_epoch
    if prompt_tokens is not None:
        block["prompt_tokens"] = prompt_tokens
    if cached_tokens is not None:
        block["cached_tokens"] = cached_tokens
    return block


def chunk_body(*, completion_id: str, created: int, model: str, loops: int,
               delta: Dict[str, Any], finish_reason: Optional[str] = None,
               dory: Optional[Dict[str, Any]] = None,
               usage: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """One `chat.completion.chunk` object."""
    body: Dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": _fingerprint(model, loops),
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason,
                     "logprobs": None}],
    }
    if usage is not None:
        body["usage"] = usage
    if dory is not None:
        body["dory"] = dory
    return body


def _usage(completion: Completion) -> Dict[str, int]:
    return {
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "total_tokens": completion.prompt_tokens + completion.completion_tokens,
    }


def _tool_call(index: int, call) -> Dict[str, Any]:
    return {
        "index": index,
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": call.name, "arguments": call.arguments_json()},
    }


def completion_body(*, completion_id: str, created: int, model: str,
                    plan: RequestPlan, started: Started,
                    completion: Completion) -> Dict[str, Any]:
    """The non-streaming `chat.completion` object."""
    message: Dict[str, Any] = {"role": "assistant"}
    if completion.tool_calls and not completion.content:
        message["content"] = None
    else:
        message["content"] = completion.content
    if completion.reasoning:
        message["reasoning_content"] = completion.reasoning
    if completion.tool_calls:
        message["tool_calls"] = [_tool_call(index, call)
                                 for index, call in enumerate(completion.tool_calls)]
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": _fingerprint(model, completion.loops),
        "choices": [{"index": 0, "message": message,
                     "finish_reason": completion.finish_reason, "logprobs": None}],
        "usage": _usage(completion),
        "dory": _dory_block(loops=completion.loops, source=completion.loop_source,
                            cache_reset=completion.cache_reset,
                            cache_epoch=completion.cache_epoch,
                            prompt_tokens=completion.prompt_tokens,
                            cached_tokens=completion.cached_tokens),
    }


def _error_body(message: str, code: Optional[str], kind: str) -> Dict[str, Any]:
    return {"error": {"message": message, "type": kind, "param": None, "code": code}}


async def _send(response: web.StreamResponse, payload: Any) -> None:
    if isinstance(payload, str):
        await response.write(f"data: {payload}\n\n".encode("utf-8"))
    else:
        await response.write(
            ("data: " + json.dumps(payload, ensure_ascii=False) + "\n\n").encode("utf-8"))


# -- handlers --------------------------------------------------------------

async def handle_models(request: web.Request) -> web.Response:
    return web.json_response(request.app[CATALOG].payload())


async def handle_health(request: web.Request) -> web.Response:
    catalog = request.app[CATALOG]
    info = catalog.info
    return web.json_response({
        "status": "ok",
        "model": info.model_id,
        "context_length": info.context_length,
        "loops": {"min": info.loop_min, "max": info.loop_max,
                  "default": info.loop_default},
        "models": catalog.ids,
    })


async def handle_options(request: web.Request) -> web.Response:
    return web.Response(status=204)


async def handle_chat(request: web.Request) -> web.Response:
    service = request.app[SERVICE]
    try:
        body = await request.json()
    except json.JSONDecodeError as error:
        return web.json_response(
            _error_body(f"the request body is not JSON: {error}", "invalid_json",
                        "invalid_request_error"), status=400)
    if not isinstance(body, dict):
        return web.json_response(
            _error_body("the request body must be a JSON object", "invalid_json",
                        "invalid_request_error"), status=400)
    try:
        plan = service.plan(body)
    except ChatError as error:
        return web.json_response(_error_body(error.message, error.code, error.kind),
                                 status=error.status)
    if plan.stream:
        return await _stream(request, service, plan)
    started, completion = await asyncio.to_thread(service.complete, plan)
    created = int(time.time())
    return web.json_response(completion_body(
        completion_id=plan.request_id, created=created, model=plan.model_id,
        plan=plan, started=started, completion=completion))


async def _stream(request: web.Request, service: ChatService,
                  plan: RequestPlan) -> web.StreamResponse:
    response = web.StreamResponse(status=200, headers=_headers())
    await response.prepare(request)
    created = int(time.time())
    common = dict(completion_id=plan.request_id, created=created, model=plan.model_id,
                  loops=plan.loops)
    await _send(response, chunk_body(
        delta={"role": "assistant", "content": ""},
        dory=_dory_block(loops=plan.loops, source=plan.loop_source), **common))
    relay = ThreadedStream(lambda: service.stream(plan))
    tool_index = 0
    try:
        async for item in relay.drain():
            if isinstance(item, Started):
                continue
            if isinstance(item, Delta):
                if item.reasoning:
                    await _send(response, chunk_body(
                        delta={"reasoning_content": item.reasoning}, **common))
                if item.content:
                    await _send(response, chunk_body(
                        delta={"content": item.content}, **common))
                for call in item.tool_calls:
                    delta = {"tool_calls": [_tool_call(tool_index, call)]}
                    await _send(response, chunk_body(delta=delta, **common))
                    tool_index += 1
                continue
            if isinstance(item, Completion):
                await _send(response, chunk_body(
                    delta={}, finish_reason=item.finish_reason,
                    dory=_dory_block(loops=item.loops, source=item.loop_source,
                                     cache_reset=item.cache_reset,
                                     cache_epoch=item.cache_epoch,
                                     prompt_tokens=item.prompt_tokens,
                                     cached_tokens=item.cached_tokens),
                    usage=_usage(item), **common))
        await _send(response, "[DONE]")
    except ChatError as error:
        LOGGER.warning("request %s failed mid-stream: %s", plan.request_id, error.message)
        await _send(response, _error_body(error.message, error.code, error.kind))
        await _send(response, "[DONE]")
    except (ConnectionResetError, asyncio.CancelledError):
        LOGGER.info("request %s: the client went away", plan.request_id)
        raise
    except Exception as error:                    # noqa: BLE001 - reported to the client
        LOGGER.exception("request %s failed mid-stream", plan.request_id)
        await _send(response, _error_body(str(error), "engine_error", "server_error"))
        await _send(response, "[DONE]")
    return response


# -- app -------------------------------------------------------------------

def _add_cors(headers) -> None:
    headers.setdefault("Access-Control-Allow-Origin", "*")
    headers.setdefault("Access-Control-Allow-Headers", "*")
    headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")


@web.middleware
async def _cors_and_log(request: web.Request, handler) -> web.StreamResponse:
    started = time.time()
    response = await handler(request)
    _add_cors(response.headers)
    LOGGER.info("%s %s -> %s (%.3fs)", request.method, request.path, response.status,
                time.time() - started)
    return response


def create_app(service: ChatService, *, catalog: Optional[ModelCatalog] = None) -> web.Application:
    """The aiohttp application: two OpenAI endpoints, health, CORS for a local UI."""
    application = web.Application(client_max_size=32 * 1024 * 1024,
                                  middlewares=[_cors_and_log])
    application[SERVICE] = service
    application[CATALOG] = catalog or service.catalog
    application.router.add_get("/v1/models", handle_models)
    application.router.add_post("/v1/chat/completions", handle_chat)
    application.router.add_get("/health", handle_health)
    application.router.add_route("OPTIONS", "/{tail:.*}", handle_options)
    return application
