"""Command line entry point: `python -m anemll.openai_server`.

    python -m anemll.openai_server --model-dir <converted-dir> --port 8000
    python -m anemll.openai_server --fake           # no model, scripted replies

The host is a loopback address and nothing else (spec D10, ticket DANE-7): the
model is not exposed on the network, and there is no auth because there is
nothing to authenticate from the outside.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import ipaddress
import logging
import signal
import sys

from aiohttp import web

from anemll.openai_server.app import create_app
from anemll.openai_server.models import ModelCatalog
from anemll.openai_server.service import ChatService

__all__ = ["build_parser", "main", "loopback_host", "serve"]

LOGGER = logging.getLogger("anemll.openai_server")

#: Replies for `--fake` so the endpoint can be tried without a converted model.
#: The first keyword found in the prompt wins, so a request that carries the
#: weather tool (the template prints its name) gets a tool call back, and every
#: other request gets a reasoning-plus-answer reply.
FAKE_TOOL_CALL = (
    "I should look it up. </think>"
    "<tool_call>\n<function=get_weather>\n<parameter=city>\nSanta Clara\n"
    "</parameter>\n</function>\n</tool_call>")
FAKE_REPLIES = {
    "get_weather": FAKE_TOOL_CALL,
    "": ("The bolt costs $3. </think>A single bolt costs **$3**, and a box of "
         "twenty costs $48."),
}


def loopback_host(host: str) -> str:
    """`host` if it is a loopback address, else SystemExit with the reason."""
    if host in ("localhost", ""):
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        raise SystemExit(
            f"--host must be a loopback address (localhost, 127.0.0.1, ::1); "
            f"{host!r} is not one") from None
    if not address.is_loopback:
        raise SystemExit(
            f"refusing to bind {host}: this server is localhost-only by design "
            f"(spec D10) and {host} would expose the model on the network")
    return host


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible server for Dory1.0-4B on the Apple Neural Engine")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", help="a converted model directory (with meta.yaml)")
    source.add_argument("--fake", action="store_true",
                        help="serve scripted replies, no model (for trying the API)")
    parser.add_argument("--model-id", default="dory-ane",
                        help="base model id; aliases are <id>-<n>loop(s)")
    parser.add_argument("--host", default="127.0.0.1", help="loopback address to bind")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--loops", type=int, default=None,
                        help="default loop count for requests that do not choose one")
    parser.add_argument("--log-level", default="INFO",
                        choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser


def _engine(args):
    if args.fake:
        from anemll.openai_server.fake_engine import FakeEngine
        return FakeEngine(replies=FAKE_REPLIES)
    from anemll.openai_server.dory_engine import DoryEngine
    return DoryEngine(args.model_dir, model_id=args.model_id)


async def _run(service: ChatService, host: str, port: int) -> None:
    application = create_app(service, catalog=service.catalog)
    runner = web.AppRunner(application, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    addresses = ", ".join(
        f"http://[{host}]:{port}" if ":" in host else f"http://{host}:{port}"
        for host, port in runner.addresses)
    info = service.engine.info
    LOGGER.info("%s (context %d, loops %d-%d, default %d) listening on %s -- "
                "localhost only, no auth", info.model_id, info.context_length,
                info.loop_min, info.loop_max, info.loop_default, addresses)
    LOGGER.info("models: %s", ", ".join(service.catalog.ids))
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_name in ("SIGINT", "SIGTERM"):
        try:
            loop.add_signal_handler(getattr(signal, signal_name), stopping.set)
        except NotImplementedError:            # pragma: no cover - not on macOS
            pass
    try:
        await stopping.wait()
    finally:
        await runner.cleanup()
        service.close()


def serve(service: ChatService, host: str, port: int) -> None:
    asyncio.run(_run(service, host, port))


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stderr)
    host = loopback_host(args.host)
    engine = _engine(args)
    if args.loops is not None:
        info = engine.info
        if not info.loop_min <= args.loops <= info.loop_max:
            raise SystemExit(f"--loops must be between {info.loop_min} and "
                             f"{info.loop_max}, got {args.loops}")
        engine.info = dataclasses.replace(info, loop_default=args.loops)
    service = ChatService(engine, ModelCatalog(engine.info))
    serve(service, host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
