# Serving a converted Dory model — the OpenAI-compatible endpoint

Spec D10, ticket DANE-7.  `anemll/openai_server/` puts an OpenAI-compatible
server in front of the runtime from `anemll/dory_chat.py`: a models list and
streaming chat completions, so any OpenAI client — and Unsloth Studio's
Connected provider — can talk to the neural engine model.  It is built on the
owner's `anemll-server` (same shape: aiohttp for SSE, a worker thread for the
model, an OpenAI client as the only consumer), with the parts that server did
not need: reasoning and answer separated, native tool calls, per-request loop
counts, and NVIDIA's sampling defaults.

## Running it

    # the real converted model (DANE-5's directory)
    python -m anemll.openai_server --model-dir ~/dev/dory-ane-artifacts/DANE-5/24576ctx
    # -> http://127.0.0.1:8000/v1

    # no model at all: scripted replies, for trying the API or a client
    python -m anemll.openai_server --fake

Options: `--model-id` (default `dory-ane`), `--host` (loopback only),
`--port` (8000), `--loops` (default loop count for requests that do not choose
one), `--log-level`.

The host is checked before the socket is opened: `0.0.0.0`, `::` and any
non-loopback address are refused with an explanation, because spec D10 binds
localhost only.  There is no auth for the same reason there is no network: there
is nothing to authenticate.

## Endpoints

| Endpoint | What it returns |
|---|---|
| `GET /v1/models` | the model, one alias per loop count, the context length and the loop range |
| `POST /v1/chat/completions` | streaming (SSE) or plain chat completions, OpenAI schema |
| `GET /health` | `{status, model, context_length, loops, models}` |
| `OPTIONS *` | CORS preflight (the GUI clients run in a browser context) |

### The models list

```json
{
  "object": "list",
  "data": [{
    "id": "dory-ane",
    "object": "model", "created": 0, "owned_by": "anemll",
    "root": "dory-ane", "parent": null, "permission": [],
    "context_length": 24576,
    "max_model_len": 24576,
    "dory": {
      "model": "dory-ane", "loops": 2, "loop_range": [1, 4],
      "default_loops": 2, "kv_mode": "last_loop",
      "loop_change_invalidates_cache": true
    }
  }]
}
```

One alias per loop count follows the base entry: `dory-ane-1loop`,
`dory-ane-2loops`, `dory-ane-3loops`, `dory-ane-4loops`, each with its loop
count in `dory.loops` and `parent` set to `dory-ane`.  `context_length` is what
ANEMLL's own tooling reads; `max_model_len` is the same number under the name
OpenAI clients use.

## Choosing the loop count

Three ways, and the server says which one it used:

1. **the model alias** — `"model": "dory-ane-4loops"`.  This is the one Studio
   can use, because it sends no custom fields.
2. **the body field** — `"dory_loops": 4` (or `"loops": 4`) on the request, e.g.
   `extra_body={"dory_loops": 4}` from the OpenAI client.
3. nothing, and the default applies (2, NVIDIA's HF config default; the community
   GGUF defaults to 4).

An alias and a body field that disagree are a 400, not a silent precedence: a
request should mean one thing.  A value outside the model's range is a 400 with
the range in the message.

**A loop change resets the cache.**  Spec D4: older rows keep the previous loop
count's K/V in `last_loop` mode, so keeping them would answer from a cache built
at another depth.  The reset, and the reason, appear in the log and in the
response:

```
INFO anemll.openai_server: loop count 2 -> 4 (from alias): cache reset, epoch 3
INFO anemll.openai_server: request chatcmpl-… model=dory-ane-4loops loops=4
     source=alias cache=reset epoch=3 prompt_tokens=61 cached=0 prefill=0.02s …
```

and every response carries the same facts in a `dory` block (on the last SSE
chunk and on the non-streaming body), plus `system_fingerprint` set to
`<model>-loops<N>`:

```json
"dory": {"loops": 4, "loops_source": "alias", "cache_reset": true,
         "cache_epoch": 3, "prompt_tokens": 61, "cached_tokens": 0}
```

Cache reuse is invisible but measurable: when a request's prompt extends what the
cache already holds, only the new tail is fed (`cached_tokens` counts the shared
prefix).  Anything else — a different conversation, or a repeated prompt whose
reply is already in the cache — resets it, because a Core ML state cannot be
rewound.

## Sampling

NVIDIA's recommendation is the default for every request: **temperature 1.0,
top_p 0.95, top_k 0, min_p 0** (`generation_config.json` and the model card; the
spec adds the two zeros).  Each can be overridden per request in the body —
`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `max_tokens`, `stop`.
`temperature: 0` is greedy: the argmax, no draw, so a test can reproduce a run
exactly.  The filter order is llama.cpp's (temperature, top_k, top_p, min_p, then
draw), so a request written for the GPU build means the same thing here.

## Reasoning

Reasoning is on by default, as Dory's template does
(`enable_thinking=True` opens a `<think>` block in the prompt).  The
model's reasoning streams as `delta.reasoning_content` and the answer as
`delta.content`; the two are never mixed in one chunk, and a marker split across
two tokens never leaks into either.  The non-streaming body carries the same
split (`message.reasoning_content`, `message.content`).

Turn it off per request with vLLM's spelling:

```python
client.chat.completions.create(..., extra_body={
    "chat_template_kwargs": {"enable_thinking": False}})
```

With reasoning off the template closes the block itself and everything the model
writes is content.

## Tool calls

Dory's native format is what its template teaches:

    <tool_call>
    <function=get_weather>
    <parameter=city>
    Santa Clara
    </parameter>
    </function>
    </tool_call>

The server parses that into OpenAI's shape: `message.tool_calls[].function.name`
and `.arguments` as a JSON string, `finish_reason: "tool_calls"`, `content`
`null` when there is no prose.  Values keep their type — a JSON object or list
stays one, a bare number becomes a number, anything else stays a string — which
is how the template writes them.  A tool result sent back as
`{"role": "tool", …}` renders as `<tool_response>` and the conversation carries
on.  Markup is only parsed when the request carried `tools`; without them the
model's text passes through unchanged.

## Example

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")

stream = client.chat.completions.create(
    model="dory-ane-2loops",
    messages=[{"role": "user", "content": "Why is the sky blue?"}],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.reasoning_content:
        print(delta.reasoning_content, end="", flush=True)   # thinking
    elif delta.content:
        print(delta.content, end="", flush=True)             # the answer
```

## Unsloth Studio (the manual step)

Studio's own code is not modified; the endpoint is added as a connection in its
providers dialog (`features/chat/chat-providers-dialog.tsx`).  With the server
running on `http://127.0.0.1:8000/v1`:

1. Studio → the chat providers/connections dialog → **Add connection**;
2. pick the custom / OpenAI-compatible provider type, and fill in
   * **Base URL**: `http://127.0.0.1:8000/v1` (Studio's own placeholder for a
     local server is `http://localhost:9931/v1`, so the `/v1` belongs in it);
   * **API key**: any non-empty value — this server has no auth, and Studio
     refuses to load models without one;
3. **Load available models** — the five ids appear (`dory-ane` and the
   `-1loop`/`-2loops`/`-3loops`/`-4loops` aliases) — enable the ones you want,
   then **Save connection**; **Check connection** should go green;
4. start a chat and pick `dory-ane` (or an alias).  Studio sends no custom body
   fields, so **the alias is how you choose the loop count from Studio**;
5. Studio stores the connection in its own backend (`/api/providers`), so the
   step survives a restart and needs no code change on either side.

Studio's GPU path is a separate `llama-server` (port 49561 here) with its own
`--override-kv dory.num_loops`; nothing in this port touches it, and both can be
registered side by side.

## Checking it

    # the acceptance checks through the OpenAI client
    python scripts/dory_server_smoke.py --base-url http://127.0.0.1:8000/v1

    # the server's own tests (no model needed: a scripted engine plays the model)
    python -m pytest tests/test_openai_server.py tests/test_openai_server_http.py \
        tests/test_openai_server_client.py -q

`scripts/dory_server_smoke.py` prints PASS/FAIL per check: the models list, a
streamed reply with the reasoning and the answer separated, a well-formed tool
call, the loop count from the alias and from the body field with the cache reset
visible, and a connection to a non-loopback address being refused.  The client
tests in `tests/test_openai_server_client.py` run the same checks against a real
loopback socket.
