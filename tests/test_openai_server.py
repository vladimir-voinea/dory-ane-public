"""Unit tests for the OpenAI-compatible server (ticket DANE-7, spec D10).

Everything here runs against the scripted `FakeEngine`: no Core ML, no weights,
no ANE.  The seam that makes that possible is `anemll.openai_server.engine`, and
the tests below are its contract -- the same calls `DoryEngine` makes on the real
model.
"""

import numpy as np
import pytest

from anemll.openai_server.engine import EngineInfo
from anemll.openai_server.fake_engine import FakeEngine
from anemll.openai_server.messages import flatten_content, to_template_messages
from anemll.openai_server.models import ModelCatalog, UnknownModelError, loop_alias
from anemll.openai_server.sampling import SamplingParams, Sampler, probabilities
from anemll.openai_server.service import ChatError, ChatService, common_prefix
from anemll.openai_server.splitter import DeltaSplitter
from anemll.openai_server.toolcalls import (
    ParsedToolCall,
    parse_tool_calls,
    split_tool_calls,
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]

TOOL_CALL_TEXT = (
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nSanta Clara\n"
    "</parameter>\n</function>\n</tool_call>"
)


# ---------------------------------------------------------------- sampling --

def test_nvidia_defaults_apply_to_an_empty_body():
    params = SamplingParams.from_body({})
    assert params.temperature == 1.0
    assert params.top_p == 0.95
    assert params.top_k == 0
    assert params.min_p == 0.0
    assert params.greedy is False


def test_a_request_overrides_every_default():
    params = SamplingParams.from_body({"temperature": 0.3, "top_p": 0.8, "top_k": 20,
                                       "min_p": 0.05, "max_tokens": 12, "seed": 7,
                                       "stop": "END"})
    assert (params.temperature, params.top_p, params.top_k, params.min_p) == (0.3, 0.8, 20, 0.05)
    assert params.max_tokens == 12
    assert params.seed == 7
    assert params.stop == ("END",)


def test_temperature_zero_is_greedy_and_deterministic():
    params = SamplingParams.from_body({"temperature": 0})
    sampler = Sampler(params)
    logits = np.array([0.1, 3.0, 2.9, -5.0])
    assert params.greedy
    assert [sampler.sample(logits) for _ in range(5)] == [1] * 5


def test_greedy_ignores_the_filters():
    # At temperature 0 the answer is the argmax whatever top_k/top_p/min_p say.
    params = SamplingParams.from_body({"temperature": 0, "top_k": 1, "top_p": 0.01,
                                       "min_p": 0.99})
    assert Sampler(params).sample([1.0, 5.0, 2.0]) == 1


def test_top_k_one_keeps_only_the_likeliest_token():
    params = SamplingParams.from_body({"temperature": 1.0, "top_k": 1, "seed": 1})
    sampler = Sampler(params)
    draws = {sampler.sample([0.0, 2.0, 1.0]) for _ in range(20)}
    assert draws == {1}


def test_top_p_keeps_the_smallest_prefix_that_reaches_it():
    logits = np.log(np.array([0.6, 0.3, 0.1]))
    params = SamplingParams.from_body({"temperature": 1.0, "top_p": 0.5})
    probs = probabilities(logits, params)
    assert probs[1] == 0.0 and probs[2] == 0.0
    assert probs[0] == pytest.approx(1.0)


def test_min_p_drops_tokens_below_its_fraction_of_the_peak():
    logits = np.log(np.array([0.70, 0.20, 0.10]))
    params = SamplingParams.from_body({"temperature": 1.0, "min_p": 0.5})
    probs = probabilities(logits, params)
    assert probs[1] == 0.0 and probs[2] == 0.0
    assert probs[0] == pytest.approx(1.0)


def test_a_seed_makes_a_sampling_run_reproducible():
    def draw(seed):
        sampler = Sampler(SamplingParams.from_body({"seed": seed}))
        return [sampler.sample([0.0, 0.0, 0.0, 0.0]) for _ in range(12)]

    assert draw(1234) == draw(1234)
    assert draw(1) != draw(2)


def test_probabilities_are_a_distribution_and_ignore_non_finite_logits():
    params = SamplingParams.from_body({"temperature": 1.0})
    probs = probabilities(np.array([1.0, -np.inf, 2.0]), params)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[1] == 0.0


@pytest.mark.parametrize("body", [
    {"temperature": -1},
    {"top_p": 0},
    {"top_p": 1.5},
    {"top_k": -1},
    {"min_p": 1.5},
    {"max_tokens": 0},
    {"temperature": "hot"},
    {"stop": [1]},
])
def test_bad_sampling_values_are_refused(body):
    with pytest.raises(ValueError):
        SamplingParams.from_body(body)


# --------------------------------------------------------------- tool calls --

def test_native_tool_call_block_becomes_an_openai_call():
    calls = parse_tool_calls(TOOL_CALL_TEXT)
    assert calls == [ParsedToolCall(name="get_weather", arguments={"city": "Santa Clara"})]
    assert calls[0].arguments_json() == '{"city": "Santa Clara"}'


def test_parameter_values_keep_their_type():
    text = ("<tool_call>\n<function=search>\n<parameter=query>\nbolts\n</parameter>\n"
            "<parameter=limit>\n3\n</parameter>\n<parameter=exact>\ntrue\n</parameter>\n"
            "<parameter=filter>\n{\"a\": 1}\n</parameter>\n"
            "<parameter=note>\nline one\nline two\n</parameter>\n"
            "</function>\n</tool_call>")
    (call,) = parse_tool_calls(text)
    assert call.name == "search"
    assert call.arguments == {"query": "bolts", "limit": 3, "exact": True,
                              "filter": {"a": 1}, "note": "line one\nline two"}


def test_two_calls_in_one_message_are_both_parsed():
    text = TOOL_CALL_TEXT + "\n" + TOOL_CALL_TEXT.replace("Santa Clara", "Oslo")
    calls = parse_tool_calls(text)
    assert [call.arguments["city"] for call in calls] == ["Santa Clara", "Oslo"]


def test_prose_without_markers_is_not_a_tool_call():
    assert parse_tool_calls("Call get_weather if you like; I will not.") == []


def test_malformed_blocks_are_ignored_rather_than_raised():
    assert parse_tool_calls("<tool_call></tool_call>") == []
    assert parse_tool_calls("<tool_call>garbage</tool_call>") == []
    assert parse_tool_calls("<tool_call><function=></function></tool_call>") == []


def test_the_hermes_tool_calls_payload_is_accepted_as_a_fallback():
    calls = parse_tool_calls('[TOOL_CALLS][{"name": "f", "arguments": {"a": 1}}]')
    assert calls == [ParsedToolCall(name="f", arguments={"a": 1})]


def test_split_tool_calls_keeps_the_text_around_the_block():
    text, calls = split_tool_calls("Sure. " + TOOL_CALL_TEXT + " Done.")
    assert calls and text == "Sure.  Done."


# ----------------------------------------------------------------- splitter --

def _collect(splitter, pieces):
    deltas = []
    for piece in pieces:
        deltas.extend(splitter.feed(piece))
    deltas.extend(splitter.finish())
    return ("".join(delta.reasoning for delta in deltas),
            "".join(delta.content for delta in deltas),
            [call for delta in deltas for call in delta.tool_calls])


def test_reasoning_and_content_are_split():
    text = "Two plus two is four. </think>It is **4**."
    assert _collect(DeltaSplitter(thinking_open=True), [text]) == (
        "Two plus two is four. ", "It is **4**.", [])


def test_the_split_is_identical_whether_text_arrives_by_character_or_at_once():
    text = "Think hard. </think>Done."
    by_char = _collect(DeltaSplitter(thinking_open=True), list(text))
    at_once = _collect(DeltaSplitter(thinking_open=True), [text])
    assert by_char == at_once == ("Think hard. ", "Done.", [])


def test_a_marker_split_across_two_tokens_is_not_leaked():
    splitter = DeltaSplitter(thinking_open=True)
    deltas = splitter.feed("reasoning</thi")
    assert "".join(delta.reasoning for delta in deltas) == "reasoning"
    assert all("</thi" not in delta.reasoning + delta.content for delta in deltas)
    deltas = splitter.feed("nk>answer")
    deltas.extend(splitter.finish())
    assert "".join(delta.content for delta in deltas) == "answer"


def test_without_an_open_thinking_block_everything_is_content():
    assert _collect(DeltaSplitter(thinking_open=False), ["Just an answer."]) == (
        "", "Just an answer.", [])


def test_a_tool_call_is_parsed_out_of_the_stream():
    text = "Checking. </think>" + TOOL_CALL_TEXT + "Done."
    reasoning, content, calls = _collect(DeltaSplitter(thinking_open=True, tool_calls=True),
                                         list(text))
    assert reasoning == "Checking. "
    assert content == "Done."
    assert [call.name for call in calls] == ["get_weather"]


def test_tool_markup_passes_through_when_the_request_carried_no_tools():
    text = "Checking. </think>" + TOOL_CALL_TEXT
    _, content, calls = _collect(DeltaSplitter(thinking_open=True, tool_calls=False), list(text))
    assert calls == []
    assert "<tool_call>" in content


def test_an_unterminated_tool_call_is_content_not_a_call():
    _, content, calls = _collect(
        DeltaSplitter(thinking_open=True, tool_calls=True),
        ["Answer. </think><tool_call>\n<function=f>\n<parameter=a>\n1"])
    assert calls == []
    assert content.startswith("<tool_call>")


def test_the_tool_calls_payload_is_parsed_at_the_end_of_the_stream():
    _, content, calls = _collect(
        DeltaSplitter(thinking_open=False, tool_calls=True),
        ['Hi [TOOL_CALLS][{"name": "f", "arguments": {"a": 1}}]'])
    assert content == "Hi "
    assert [call.arguments for call in calls] == [{"a": 1}]


# ----------------------------------------------------------------- messages --

def test_message_content_can_be_a_list_of_parts():
    messages = to_template_messages([{"role": "user", "content": [
        {"type": "text", "text": "hello "}, {"type": "text", "text": "world"}]}])
    assert messages == [{"role": "user", "content": "hello world"}]
    assert flatten_content("plain") == "plain"


def test_assistant_tool_call_arguments_are_parsed_for_the_template():
    messages = to_template_messages([{
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": "get_weather",
                                     "arguments": '{"city": "SF"}'}}],
        "reasoning_content": "I should call the tool.",
    }])
    assert messages[0]["tool_calls"] == [{"name": "get_weather",
                                          "arguments": {"city": "SF"}}]
    assert messages[0]["reasoning_content"] == "I should call the tool."


def test_broken_tool_call_arguments_are_a_request_error():
    with pytest.raises(ValueError):
        to_template_messages([{
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": "{not json"}}]}])
    with pytest.raises(ValueError):
        to_template_messages([{
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": "f", "arguments": "[1, 2]"}}]}])
    with pytest.raises(ValueError):
        to_template_messages([{
            "role": "assistant", "content": "",
            "tool_calls": [{"function": {"arguments": "{}"}}]}])


def test_empty_messages_are_refused():
    with pytest.raises(ValueError):
        to_template_messages([])
    with pytest.raises(ValueError):
        to_template_messages([{"content": "no role"}])


# ------------------------------------------------------------------- models --

def _catalog(context_length=24576, **kwargs):
    info = EngineInfo(model_id="dory-ane", context_length=context_length,
                      loop_min=1, loop_max=4, loop_default=2, vocab_size=256, **kwargs)
    return ModelCatalog(info, created=1700000000)


def test_the_models_list_reports_context_loops_and_aliases():
    payload = _catalog().payload()
    assert payload["object"] == "list"
    by_id = {entry["id"]: entry for entry in payload["data"]}
    base = by_id["dory-ane"]
    assert base["context_length"] == 24576
    assert base["max_model_len"] == 24576
    assert base["dory"]["loops"] == 2
    assert base["dory"]["loop_range"] == [1, 4]
    assert base["dory"]["default_loops"] == 2
    assert base["dory"]["kv_mode"] == "last_loop"
    for loops in (1, 2, 4):
        alias = by_id[loop_alias("dory-ane", loops)]
        assert alias["dory"]["loops"] == loops
        assert alias["parent"] == "dory-ane"
    assert loop_alias("dory-ane", 1) == "dory-ane-1loop"
    assert loop_alias("dory-ane", 2) == "dory-ane-2loops"
    assert loop_alias("dory-ane", 4) == "dory-ane-4loops"


def test_an_unknown_model_is_refused_with_the_valid_ids():
    catalog = _catalog()
    assert catalog.loops_for("dory-ane") is None
    assert catalog.loops_for("dory-ane-4loops") == 4
    with pytest.raises(UnknownModelError) as error:
        catalog.entry("gpt-4o")
    assert "dory-ane-1loop" in str(error.value)


# ------------------------------------------------------------------ service --

def _service(reply, **kwargs):
    engine = FakeEngine(replies={"": reply}, **kwargs)
    return ChatService(engine), engine


def _ask(service, content="hello", model="dory-ane", **body):
    request = {"model": model, "messages": [{"role": "user", "content": content}]}
    request.update(body)
    plan = service.plan(request)
    return plan, list(service.stream(plan))


def test_a_streamed_reply_separates_reasoning_from_the_answer():
    service, _ = _service("The bolt is $3. </think>A bolt costs **$3**.")
    plan, items = _ask(service)
    assert items[0].loops == 2 and items[0].loop_source == "default"
    completion = items[-1]
    assert completion.reasoning == "The bolt is $3. "
    assert completion.content == "A bolt costs **$3**."
    assert completion.finish_reason == "stop"
    assert completion.completion_tokens > 0
    # every delta carries exactly one channel
    for item in items[1:-1]:
        assert bool(item.reasoning) != bool(item.content)


def test_the_loop_count_can_come_from_the_body():
    service, _ = _service("ok. </think>done")
    for field in ("dory_loops", "loops"):
        plan, items = _ask(service, **{field: 4})
        assert plan.loops == 4 and plan.loop_source == "body"
        assert items[-1].loops == 4


def test_the_loop_count_can_come_from_the_alias():
    service, _ = _service("ok. </think>done")
    plan, items = _ask(service, model="dory-ane-4loops")
    assert plan.loops == 4 and plan.loop_source == "alias"
    assert items[-1].loops == 4
    assert items[-1].loop_source == "alias"


def test_an_alias_and_a_body_field_that_disagree_are_refused():
    service, _ = _service("ok")
    with pytest.raises(ChatError) as error:
        service.plan({"model": "dory-ane-2loops", "dory_loops": 4,
                      "messages": [{"role": "user", "content": "hi"}]})
    assert error.value.status == 400 and error.value.code == "conflicting_loops"


def test_an_out_of_range_loop_count_is_refused():
    service, _ = _service("ok")
    with pytest.raises(ChatError) as error:
        service.plan({"model": "dory-ane", "loops": 9,
                      "messages": [{"role": "user", "content": "hi"}]})
    assert error.value.code == "invalid_loops"


def test_a_loop_change_resets_the_cache():
    service, engine = _service("one. </think>answer")
    _, first = _ask(service)
    assert first[0].cache_reset is False             # first request: nothing cached
    epoch = first[0].cache_epoch

    _, second = _ask(service, loops=4)
    assert second[0].cache_reset is True             # loop change -> cache reset
    assert second[0].cache_epoch == epoch + 1
    assert second[0].loops == 4
    assert engine.reset_count == 1

    # The same loop count does not reset anything: this prompt is an extension
    # of what the 4-loop cache holds, so the tail alone is fed.
    _, third = _ask(service, loops=4, content="hello",
                    messages=[{"role": "user", "content": "hello"},
                              {"role": "assistant", "content": "one. </think>answer"},
                              {"role": "user", "content": "hello"}])
    assert third[0].cache_reset is False
    assert third[0].cache_epoch == epoch + 1
    assert engine.reset_count == 1
    assert third[0].cached_tokens > 0


def test_a_longer_conversation_reuses_the_cache():
    reply = "one. </think>answer"
    service, engine = _service(reply)
    plan = service.plan({"model": "dory-ane",
                         "messages": [{"role": "user", "content": "first question"}]})
    list(service.stream(plan))
    fed_first = engine.prefilled_tokens

    # A client that echoes the assistant turn back verbatim makes the new prompt
    # an extension of the cached one, so only the new tail is fed.
    longer = {"model": "dory-ane", "messages": [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": reply},
        {"role": "user", "content": "second question"}]}
    items = list(service.stream(service.plan(longer)))
    assert items[0].cache_reset is False
    assert items[0].cached_tokens > 0
    assert (engine.prefilled_tokens - fed_first
            == items[0].prompt_tokens - items[0].cached_tokens)


def test_an_unrelated_conversation_resets_the_cache():
    service, engine = _service("one. </think>answer")
    list(service.stream(service.plan(
        {"model": "dory-ane", "messages": [{"role": "user", "content": "about bolts"}]})))
    items = list(service.stream(service.plan(
        {"model": "dory-ane", "messages": [{"role": "user", "content": "about hammers"}]})))
    assert items[0].cache_reset is True
    assert items[0].cached_tokens == 0
    assert engine.reset_count == 1


def test_max_tokens_stops_the_reply_with_length():
    service, _ = _service("abcdefghij. </think>klmnopqrst")
    plan, items = _ask(service, max_tokens=4)
    completion = items[-1]
    assert completion.finish_reason == "length"
    assert completion.completion_tokens == 4
    assert plan.sampling.max_tokens == 4


def test_a_stop_sequence_ends_the_reply_and_is_not_emitted():
    service, _ = _service("The answer is 42. </think>It is 42. STOP trailing words")
    plan, items = _ask(service, stop=["STOP"])
    completion = items[-1]
    assert completion.finish_reason == "stop"
    assert "STOP" not in completion.content
    assert completion.content == "It is 42. "
    assert completion.reasoning == "The answer is 42. "


def test_a_tool_call_is_returned_in_the_openai_shape():
    service, _ = _service("Checking. </think>" + TOOL_CALL_TEXT)
    plan, items = _ask(service, tools=TOOLS)
    completion = items[-1]
    assert completion.finish_reason == "tool_calls"
    assert [call.name for call in completion.tool_calls] == ["get_weather"]
    assert completion.tool_calls[0].arguments == {"city": "Santa Clara"}
    assert plan.parse_tool_calls is True


def test_tool_markup_without_tools_in_the_request_stays_text():
    service, _ = _service("Checking. </think>" + TOOL_CALL_TEXT)
    _, items = _ask(service)
    completion = items[-1]
    assert completion.tool_calls == ()
    assert completion.finish_reason == "stop"
    assert "<tool_call>" in completion.content


def test_reasoning_can_be_switched_off():
    service, _ = _service("Just the answer.")
    plan, items = _ask(service, chat_template_kwargs={"enable_thinking": False})
    assert plan.reasoning is False
    completion = items[-1]
    assert completion.reasoning == ""
    assert completion.content == "Just the answer."


def test_a_prompt_longer_than_the_context_is_refused_with_an_error():
    service, _ = _service("ok. </think>done", context_length=48)
    with pytest.raises(ChatError) as error:
        _ask(service, content="x" * 200)
    assert error.value.code == "context_length_exceeded"
    assert error.value.status == 400


def test_max_tokens_is_clamped_to_the_context_that_is_left():
    service, _ = _service("abcdefghijklmnopqrstuvwxyz", context_length=64)
    plan, items = _ask(service, content="hello", max_tokens=10000)
    assert items[0].prompt_tokens < 64
    assert items[0].max_tokens == 64 - items[0].prompt_tokens - 1


def test_an_unknown_model_is_a_404():
    service, _ = _service("ok")
    with pytest.raises(ChatError) as error:
        service.plan({"model": "not-here", "messages": [{"role": "user", "content": "hi"}]})
    assert error.value.status == 404 and error.value.code == "model_not_found"


def test_common_prefix_counts_shared_leading_tokens():
    assert common_prefix([1, 2, 3], [1, 2, 4]) == 2
    assert common_prefix([1, 2], []) == 0
