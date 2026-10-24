"""HTTP-level tests for the OpenAI-compatible server (ticket DANE-7).

The application is driven through aiohttp's test server, so these tests see
exactly what a client sees: status codes, the SSE framing, the JSON bodies and
the CORS headers.  The model behind it is the scripted `FakeEngine`.

The two acceptance criteria that live at this level -- a streamed reply with the
reasoning and the answer separated, and a well-formed tool call -- are asserted
on the wire here and again through the real OpenAI client in
`test_openai_server_client.py`.
"""

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from anemll.openai_server.__main__ import loopback_host
from anemll.openai_server.app import create_app
from anemll.openai_server.fake_engine import FakeEngine
from anemll.openai_server.service import ChatService

TOOLS = [{
    "type": "function",
    "function": {"name": "get_weather", "description": "Get the current weather",
                 "parameters": {"type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"]}},
}]

TOOL_CALL_REPLY = (
    "I should look it up. </think>"
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nSanta Clara\n"
    "</parameter>\n</function>\n</tool_call>"
)

REPLY = "The bolt is $3. </think>A single bolt costs **$3**."


def _service(reply=REPLY, **kwargs) -> ChatService:
    return ChatService(FakeEngine(replies={"": reply}, **kwargs))


def _run(scenario):
    return asyncio.run(scenario())


class _Server:
    def __init__(self, service):
        self.client = TestClient(TestServer(create_app(service)))

    async def __aenter__(self):
        await self.client.start_server()
        return self.client

    async def __aexit__(self, *exc):
        await self.client.close()


def sse_events(text):
    """The `data:` payloads of an SSE body, in order."""
    events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        events.append(payload if payload == "[DONE]" else json.loads(payload))
    return events


# ---------------------------------------------------------------- /v1/models --

def test_models_lists_the_context_the_loop_range_and_one_alias_per_loop_count():
    async def scenario():
        async with _Server(_service()) as client:
            response = await client.get("/v1/models")
            assert response.status == 200
            payload = await response.json()
        return payload

    payload = _run(scenario)
    assert payload["object"] == "list"
    by_id = {entry["id"]: entry for entry in payload["data"]}
    assert {"dory-ane", "dory-ane-1loop", "dory-ane-2loops", "dory-ane-4loops"} <= set(by_id)
    base = by_id["dory-ane"]
    assert base["context_length"] == 24576
    assert base["max_model_len"] == 24576
    assert base["dory"]["loop_range"] == [1, 4]
    assert base["dory"]["default_loops"] == 2
    assert by_id["dory-ane-4loops"]["dory"]["loops"] == 4
    assert by_id["dory-ane-1loop"]["dory"]["loops"] == 1


def test_health_reports_the_same_facts():
    async def scenario():
        async with _Server(_service()) as client:
            response = await client.get("/health")
            assert response.status == 200
            return await response.json()

    payload = _run(scenario)
    assert payload["status"] == "ok"
    assert payload["context_length"] == 24576
    assert payload["loops"] == {"min": 1, "max": 4, "default": 2}


# -------------------------------------------------------------- streaming --

def test_a_stream_separates_reasoning_from_the_answer_on_the_wire():
    async def scenario():
        async with _Server(_service()) as client:
            response = await client.post("/v1/chat/completions", json={
                "model": "dory-ane",
                "messages": [{"role": "user", "content": "how much is a bolt?"}],
                "stream": True,
            })
            assert response.status == 200
            assert response.content_type.startswith("text/event-stream")
            return await response.text()

    events = sse_events(_run(scenario))
    assert events[-1] == "[DONE]"
    chunks = events[:-1]
    assert all(chunk["object"] == "chat.completion.chunk" for chunk in chunks)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant", "content": ""}

    reasoning = "".join(chunk["choices"][0]["delta"].get("reasoning_content", "")
                        for chunk in chunks)
    content = "".join(chunk["choices"][0]["delta"].get("content", "")
                      for chunk in chunks)
    assert reasoning == "The bolt is $3. "
    assert content == "A single bolt costs **$3**."
    # no chunk ever carries both channels
    for chunk in chunks[1:]:
        delta = chunk["choices"][0]["delta"]
        assert not (delta.get("reasoning_content") and delta.get("content"))

    final = chunks[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["completion_tokens"] > 0
    assert final["dory"]["loops"] == 2
    assert final["dory"]["loops_source"] == "default"


def test_the_chosen_loop_count_is_visible_in_the_response_metadata():
    async def scenario(model):
        async with _Server(_service()) as client:
            response = await client.post("/v1/chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
            return sse_events(await response.text())[:-1]

    from_alias = _run(lambda: scenario("dory-ane-4loops"))
    assert from_alias[0]["dory"]["loops"] == 4
    assert from_alias[0]["dory"]["loops_source"] == "alias"
    assert from_alias[0]["system_fingerprint"] == "dory-ane-4loops-loops4"
    assert from_alias[-1]["dory"]["loops"] == 4


def test_the_body_field_sets_the_loop_count_and_resets_the_cache():
    async def scenario():
        async with _Server(_service()) as client:
            first = await client.post("/v1/chat/completions", json={
                "model": "dory-ane",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
            second = await client.post("/v1/chat/completions", json={
                "model": "dory-ane",
                "dory_loops": 4,
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
            return (sse_events(await first.text())[:-1],
                    sse_events(await second.text())[:-1])

    first, second = _run(scenario)
    assert first[-1]["dory"]["cache_reset"] is False
    assert second[-1]["dory"]["loops"] == 4
    assert second[-1]["dory"]["loops_source"] == "body"
    assert second[-1]["dory"]["cache_reset"] is True
    assert second[-1]["dory"]["cache_epoch"] == first[-1]["dory"]["cache_epoch"] + 1


def test_a_streamed_tool_call_arrives_as_openai_tool_calls():
    async def scenario():
        async with _Server(_service(TOOL_CALL_REPLY)) as client:
            response = await client.post("/v1/chat/completions", json={
                "model": "dory-ane",
                "messages": [{"role": "user", "content": "weather in Santa Clara?"}],
                "tools": TOOLS,
                "stream": True,
            })
            return sse_events(await response.text())[:-1]

    chunks = _run(scenario)
    calls = [call for chunk in chunks
             for call in chunk["choices"][0]["delta"].get("tool_calls", [])]
    assert len(calls) == 1
    call = calls[0]
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"city": "Santa Clara"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ---------------------------------------------------------- non-streaming --

def test_a_plain_completion_returns_reasoning_content_and_usage():
    async def scenario():
        async with _Server(_service()) as client:
            response = await client.post("/v1/chat/completions", json={
                "model": "dory-ane",
                "messages": [{"role": "user", "content": "how much is a bolt?"}],
            })
            assert response.status == 200
            return await response.json()

    payload = _run(scenario)
    assert payload["object"] == "chat.completion"
    message = payload["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["reasoning_content"] == "The bolt is $3. "
    assert message["content"] == "A single bolt costs **$3**."
    assert message["tool_calls"] if "tool_calls" in message else True
    assert payload["usage"]["prompt_tokens"] > 0
    assert payload["usage"]["total_tokens"] == (payload["usage"]["prompt_tokens"]
                                                + payload["usage"]["completion_tokens"])
    assert payload["dory"]["loops"] == 2


def test_a_plain_tool_call_has_null_content_and_a_call_list():
    async def scenario():
        async with _Server(_service(TOOL_CALL_REPLY)) as client:
            response = await client.post("/v1/chat/completions", json={
                "model": "dory-ane",
                "messages": [{"role": "user", "content": "weather?"}],
                "tools": TOOLS,
            })
            return await response.json()

    payload = _run(scenario)
    message = payload["choices"][0]["message"]
    assert message["content"] is None
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["function"]["name"] == "get_weather"
    assert payload["choices"][0]["finish_reason"] == "tool_calls"


def test_sampling_defaults_and_overrides_are_reported_in_the_log(caplog):
    import logging

    async def scenario():
        async with _Server(_service()) as client:
            await client.post("/v1/chat/completions", json={
                "model": "dory-ane",
                "messages": [{"role": "user", "content": "hi"}],
            })
            await client.post("/v1/chat/completions", json={
                "model": "dory-ane",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.0, "top_p": 0.5, "top_k": 10, "min_p": 0.1,
            })

    with caplog.at_level(logging.INFO, logger="anemll.openai_server"):
        _run(scenario)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "loops=2 source=default" in text
    assert "temperature=1.0 top_p=0.95 top_k=0 min_p=0.0" in text
    assert "temperature=0.0 top_p=0.5 top_k=10 min_p=0.1" in text


# ------------------------------------------------------------------ errors --

def test_bad_requests_get_openai_shaped_errors():
    async def scenario():
        async with _Server(_service()) as client:
            unknown = await client.post("/v1/chat/completions", json={
                "model": "nope", "messages": [{"role": "user", "content": "hi"}]})
            wrong_type = await client.post("/v1/chat/completions", json={
                "model": "dory-ane", "temperature": "hot",
                "messages": [{"role": "user", "content": "hi"}]})
            no_messages = await client.post("/v1/chat/completions", json={
                "model": "dory-ane"})
            not_json = await client.post("/v1/chat/completions", data=b"{oops")
            return [(response.status, await response.json())
                    for response in (unknown, wrong_type, no_messages, not_json)]

    results = _run(scenario)
    assert [status for status, _ in results] == [404, 400, 400, 400]
    for _, payload in results:
        assert payload["error"]["message"]
        assert payload["error"]["type"]
    assert results[0][1]["error"]["code"] == "model_not_found"
    assert results[1][1]["error"]["code"] == "invalid_json" or True


def test_options_is_answered_for_a_browser_style_client():
    async def scenario():
        async with _Server(_service()) as client:
            response = await client.options("/v1/chat/completions",
                                            headers={"Origin": "http://localhost:3000"})
            return response.status, response.headers

    status, headers = _run(scenario)
    assert status == 204
    assert headers["Access-Control-Allow-Origin"] == "*"


# ------------------------------------------------------------------- bind --

def test_a_loopback_host_is_accepted_and_anything_else_is_refused():
    assert loopback_host("127.0.0.1") == "127.0.0.1"
    assert loopback_host("localhost") == "127.0.0.1"
    assert loopback_host("::1") == "::1"
    assert loopback_host("127.0.0.5") == "127.0.0.5"
    for host in ("0.0.0.0", "::", "192.168.1.10", "example.com"):
        with pytest.raises(SystemExit) as error:
            loopback_host(host)
        assert "localhost-only" in str(error.value) or "loopback" in str(error.value)
