"""The acceptance tests at the user-facing seam: the OpenAI Python client.

Spec "Testing Decisions": the seam is an OpenAI client and Studio, nothing
lower.  These tests therefore drive the real `openai` package against a real
`aiohttp` server on a real loopback socket -- only the model behind it is the
scripted `FakeEngine`, because the converted packages are a separate ticket.

They skip when the `openai` package is not installed; `scripts/dory_server_smoke.py`
runs the same checks against a live server with the real model.
"""

import asyncio
import json
import logging
import socket

import pytest

openai = pytest.importorskip("openai")

from aiohttp import web  # noqa: E402  (after the importorskip above)

from anemll.openai_server.app import create_app  # noqa: E402
from anemll.openai_server.fake_engine import FakeEngine  # noqa: E402
from anemll.openai_server.service import ChatService  # noqa: E402

TOOLS = [{
    "type": "function",
    "function": {"name": "get_weather", "description": "Get the current weather",
                 "parameters": {"type": "object",
                                "properties": {"city": {"type": "string"}},
                                "required": ["city"]}},
}]

REPLY = "The bolt is $3. </think>A single bolt costs **$3**."
TOOL_CALL_REPLY = (
    "I should look it up. </think>"
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nSanta Clara\n"
    "</parameter>\n</function>\n</tool_call>"
)


def _dory(obj):
    """This port's metadata block, from an OpenAI response object."""
    extra = getattr(obj, "model_extra", None) or {}
    return extra.get("dory") or getattr(obj, "dory", None) or {}


class _Live:
    """A real server on 127.0.0.1, reachable by the OpenAI client."""

    def __init__(self, service):
        self.service = service

    async def __aenter__(self):
        self.runner = web.AppRunner(create_app(self.service), access_log=None)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.port = self.runner.addresses[0][1]
        self.url = f"http://127.0.0.1:{self.port}/v1"
        return self

    async def __aexit__(self, *exc):
        await self.runner.cleanup()


def _ask(service, scenario):
    """Run a scenario against the live server with the synchronous OpenAI client.

    The client blocks, and the server is this very event loop, so the scenario
    has to run in a worker thread -- otherwise the request would wait for a
    handler that can never run.
    """

    async def run():
        async with _Live(service) as live:
            client = openai.OpenAI(base_url=live.url, api_key="local", max_retries=0)
            try:
                return await asyncio.to_thread(scenario, client, live)
            finally:
                client.close()

    return asyncio.run(run())


def test_the_client_lists_the_models():
    def scenario(client, live):
        return [model.id for model in client.models.list()]

    ids = _ask(ChatService(FakeEngine()), scenario)
    assert "dory-ane" in ids
    assert "dory-ane-2loops" in ids and "dory-ane-4loops" in ids


def test_the_client_streams_reasoning_and_content_separately():
    def scenario(client, live):
        stream = client.chat.completions.create(
            model="dory-ane",
            messages=[{"role": "user", "content": "how much is a bolt?"}],
            stream=True,
        )
        reasoning = []
        content = []
        order = []
        dory = {}
        for chunk in stream:
            delta = chunk.choices[0].delta
            if getattr(delta, "reasoning_content", None):
                reasoning.append(delta.reasoning_content)
                order.append("reasoning")
            if delta.content:
                content.append(delta.content)
                order.append("content")
            if _dory(chunk):
                dory = _dory(chunk)
        return "".join(reasoning), "".join(content), order, dory

    reasoning, content, order, dory = _ask(ChatService(FakeEngine(replies={"": REPLY})),
                                           scenario)
    assert reasoning == "The bolt is $3. "
    assert content == "A single bolt costs **$3**."
    assert set(order) == {"reasoning", "content"}
    assert order.index("reasoning") < len(order) - 1 - order[::-1].index("content")
    assert dory["loops"] == 2


def test_the_client_gets_a_well_formed_tool_call():
    def scenario(client, live):
        response = client.chat.completions.create(
            model="dory-ane",
            messages=[{"role": "user", "content": "What is the weather in Santa Clara?"}],
            tools=TOOLS,
        )
        message = response.choices[0].message
        return (response.choices[0].finish_reason, message.content, message.tool_calls,
                message.reasoning_content)

    finish_reason, content, tool_calls, reasoning = _ask(
        ChatService(FakeEngine(replies={"": TOOL_CALL_REPLY})), scenario)
    assert finish_reason == "tool_calls"
    assert content is None
    assert reasoning == "I should look it up. "
    assert len(tool_calls) == 1
    call = tool_calls[0]
    assert call.type == "function"
    assert call.id.startswith("call_")
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Santa Clara"}


def test_the_client_sends_a_tool_result_back_and_gets_an_answer():
    def scenario(client, live):
        response = client.chat.completions.create(
            model="dory-ane",
            messages=[
                {"role": "user", "content": "What is the weather in Santa Clara?"},
                {"role": "assistant", "content": None, "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": "get_weather",
                                 "arguments": '{"city": "Santa Clara"}'}}]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 21 C"},
            ],
            tools=TOOLS,
        )
        return response.choices[0].message.content

    content = _ask(ChatService(FakeEngine(
        replies={"": "Reading the tool result. </think>It is sunny."})), scenario)
    assert content == "It is sunny."


def test_the_client_chooses_the_loop_count_with_an_extra_body_field():
    def scenario(client, live):
        response = client.chat.completions.create(
            model="dory-ane",
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"dory_loops": 4},
        )
        return _dory(response)

    dory = _ask(ChatService(FakeEngine(replies={"": REPLY})), scenario)
    assert dory["loops"] == 4
    assert dory["loops_source"] == "body"


def test_the_client_chooses_the_loop_count_with_a_model_alias():
    def scenario(client, live):
        response = client.chat.completions.create(
            model="dory-ane-4loops",
            messages=[{"role": "user", "content": "hi"}],
        )
        return response.model, _dory(response)

    model, dory = _ask(ChatService(FakeEngine(replies={"": REPLY})), scenario)
    assert model == "dory-ane-4loops"
    assert dory["loops"] == 4 and dory["loops_source"] == "alias"


def test_the_server_is_reachable_on_loopback_and_not_on_the_network():
    async def run():
        async with _Live(ChatService(FakeEngine(replies={"": REPLY}))) as live:
            host, port = live.runner.addresses[0][:2]
            assert host == "127.0.0.1"
            with socket.create_connection(("127.0.0.1", port), timeout=5):
                pass
            try:
                lan = socket.gethostbyname(socket.gethostname())
            except OSError:                       # pragma: no cover - no hostname
                lan = ""
            if lan and not lan.startswith("127."):
                with pytest.raises(OSError):
                    socket.create_connection((lan, port), timeout=2)

    asyncio.run(run())


def test_the_loop_choice_and_the_cache_reset_are_logged(caplog):
    def scenario(client, live):
        client.chat.completions.create(model="dory-ane",
                                       messages=[{"role": "user", "content": "hi"}])
        client.chat.completions.create(model="dory-ane",
                                       messages=[{"role": "user", "content": "hi"}],
                                       extra_body={"dory_loops": 4})

    with caplog.at_level(logging.INFO, logger="anemll.openai_server"):
        _ask(ChatService(FakeEngine(replies={"": REPLY})), scenario)
    text = "\n".join(record.getMessage() for record in caplog.records)
    assert "loops=2 source=default" in text
    assert "loops=4 source=body" in text
    assert "cache reset" in text
