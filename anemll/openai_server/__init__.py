"""An OpenAI-compatible server for a converted Dory1.0-4B model.

Spec D10 / ticket DANE-7.  The modules, in the order a request moves through
them:

| module | what it is |
|---|---|
| `engine.py` | the seam: `Engine`/`Session` protocols, no Core ML import |
| `dory_engine.py` | the real engine, `anemll.dory_chat` behind that seam |
| `fake_engine.py` | a scripted engine, so the server tests need no model |
| `messages.py` | OpenAI messages -> Dory template messages |
| `models.py` | model ids and the loop-count aliases |
| `sampling.py` | NVIDIA's defaults, greedy at temperature 0, per-request overrides |
| `toolcalls.py` | Dory's native `<tool_call>` format -> OpenAI tool calls |
| `splitter.py` | streamed text -> reasoning / content / tool calls |
| `service.py` | cache, loop choice, the generation loop |
| `app.py` | the HTTP surface: `/v1/models`, `/v1/chat/completions` (SSE), `/health` |

Run it with `python -m anemll.openai_server --model-dir DIR`.
"""

__all__ = ["app", "dory_engine", "engine", "fake_engine", "messages", "models",
           "sampling", "service", "splitter", "toolcalls"]
