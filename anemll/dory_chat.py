"""Talk to a converted Dory model on the Neural Engine.

Spec D9: embeddings -> input -> recurrent x N -> output -> LM head per decode
token, and per 64-token prefill batch, with coremltools predictions carrying one
shared MLState.  Spec D4: the loop count is chosen at run time (1-4) and changing
it invalidates the cache, because older rows keep the *final* loop's K/V.

Three entry points:

    python -m anemll.dory_chat chat  --model-dir DIR [--loops 2] [--prompt TEXT]
    python -m anemll.dory_chat bench --model-dir DIR --context 1024 --loops 1,2,4
    python -m anemll.dory_chat score --model-dir DIR --reference REF --out CAND

The chat loop streams tokens, stops on EOS, and takes commands mid-session:

    /loops 4     change the loop count (resets the KV cache, and says so)
    /reset       drop the conversation and the cache
    /quit        leave

Reasoning is on by default, through Dory's own chat template (enable_thinking),
as spec D9 and the ticket ask.  --no-think turns it off for a run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from anemll.ane_converter.dory_converter import MetaError, read_meta
from anemll.utils import dory_caches as dc

__all__ = ["DoryRuntime", "ChatTokenizer", "main", "build_parser"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _coreml():
    import coremltools as ct
    return ct


def _torch():
    import torch
    return torch


def resolve_model_path(model_dir: Path, entry: dict) -> Path:
    """The file the meta names, or its sibling if only the other form exists."""
    path = Path(model_dir) / entry["file"]
    if path.exists():
        return path
    for suffix in (".mlpackage", ".mlmodelc"):
        alternative = path.with_suffix(suffix)
        if alternative.exists():
            return alternative
    raise FileNotFoundError(f"neither {path.name} nor its .mlpackage/.mlmodelc sibling "
                            f"exists in {model_dir}")


def load_function(path: Path, function_name: str, compute_units=None):
    """Load one function of one component.

    A multifunction .mlmodelc cannot always be told which function to use, so a
    compiled package falls back to the .mlpackage beside it, which Core ML will
    compile on first load (that first load is what the compile-cache gate
    measures).  Mirrors ANEMLL tests/chat.py:382-470.
    """
    ct = _coreml()
    if compute_units is None:
        compute_units = ct.ComputeUnit.CPU_AND_NE
    path = Path(path)
    if path.suffix == ".mlmodelc":
        try:
            return ct.models.CompiledMLModel(str(path), compute_units,
                                             function_name=function_name)
        except Exception:
            package = path.with_suffix(".mlpackage")
            if not package.exists():
                raise
            return ct.models.MLModel(str(package), function_name=function_name)
    return ct.models.MLModel(str(path), function_name=function_name)


class DoryRuntime:
    """The five components, driven the way spec D6 says they are called."""

    def __init__(self, model_dir, compute_units=None, verbose=True):
        self.model_dir = Path(model_dir)
        self.meta = read_meta(self.model_dir / "meta.yaml")
        self.params = self.meta["parameters"]
        self.verbose = verbose
        self.compute_units = compute_units
        self.models = {}
        self.load_seconds = 0.0
        started = time.time()
        self._load_models()
        self.load_seconds = time.time() - started
        self._state = None
        self._loops = int(self.meta["loops"]["default"])
        self.reset_state()
        self.predict_seconds = 0.0
        self.predict_calls = 0

    # -- wiring ------------------------------------------------------------
    def _load_models(self):
        for name in self.meta["run_order"]:
            entry = self.meta["components"][name]
            path = resolve_model_path(self.model_dir, entry)
            self.models[name] = {
                fn: load_function(path, fn, self.compute_units)
                for fn in entry["functions"]
            }
            if self.verbose:
                print(f"loaded {name}: {path.name} ({', '.join(entry['functions'])})",
                      file=sys.stderr)

    @property
    def state_owner(self) -> str:
        return self.meta["run_order"][0]

    def reset_state(self) -> None:
        """A fresh MLState -- what a loop-count change must do (spec D4)."""
        self._state = self.models[self.state_owner]["infer"].make_state()

    @property
    def loops(self) -> int:
        return self._loops

    def set_loops(self, loops: int) -> bool:
        """Select the loop count; returns True when the cache was reset."""
        loops = int(loops)
        lo, hi = self.meta["loops"]["min"], self.meta["loops"]["max"]
        if not lo <= loops <= hi:
            raise ValueError(f"loop count must be {lo}..{hi}, got {loops}")
        changed = loops != self._loops
        if changed:
            self._loops = loops
            self.reset_state()
        return changed

    # -- tensors -----------------------------------------------------------
    def build_mask(self, start: int, q_len: int) -> np.ndarray:
        return dc.build_causal_mask(
            start, q_len, self.params["context_length"],
            self.params["sliding_rows"], self.params["sliding_window"],
            dtype=_torch().float16).numpy()

    def _predict(self, name: str, inputs: dict, function: str) -> dict:
        started = time.time()
        out = self.models[name][function].predict(inputs, state=self._state)
        self.predict_seconds += time.time() - started
        self.predict_calls += 1
        return out

    def _blocks(self, hidden, start, q_len):
        """Run every block component in run order, looping the recurrent ones."""
        mask = self.build_mask(start, q_len)
        pos = np.array([start], dtype=np.int32)
        cache_pos = np.array([start % self.params["sliding_rows"]], dtype=np.int32)
        function = "prefill" if q_len > 1 else "infer"
        order = [name for name in self.meta["run_order"]
                 if self.meta["components"][name]["kind"] == "blocks"]
        hidden = np.ascontiguousarray(hidden, dtype=np.float16)
        index = 0
        while index < len(order):
            entry = self.meta["components"][order[index]]
            if entry["calls_per_token"] == "loops":
                group = []
                while (index < len(order)
                       and self.meta["components"][order[index]]["calls_per_token"] == "loops"):
                    group.append(order[index])
                    index += 1
                h_in = hidden
                for _ in range(self._loops):
                    for name in group:
                        inputs = {"hidden_states": hidden, "causal_mask": mask,
                                  "current_pos": pos, "cache_pos": cache_pos}
                        if self.meta["components"][name]["loop_entry"]:
                            inputs["h_in"] = h_in
                        hidden = self._predict(name, inputs, function)["output_hidden_states"]
                continue
            inputs = {"hidden_states": hidden, "causal_mask": mask,
                      "current_pos": pos, "cache_pos": cache_pos}
            hidden = self._predict(order[index], inputs, function)["output_hidden_states"]
            index += 1
        return hidden

    def embeddings(self, input_ids) -> np.ndarray:
        ids = np.asarray(input_ids, dtype=np.int32).reshape(1, -1)
        function = "prefill" if ids.shape[1] > 1 else "infer"
        return self._predict("embeddings", {"input_ids": ids}, function)["hidden_states"]

    def lm_head(self, hidden) -> np.ndarray:
        entry = self.meta["components"]["lm_head"]
        function = "prefill" if hidden.shape[1] > 1 else "infer"
        out = self._predict("lm_head", {"hidden_states": hidden}, function)
        if self.params["argmax_in_model"]:
            best_val, best_id = None, None
            offset = 0
            for part, size in enumerate(self.params["lm_head_chunk_sizes"], start=1):
                value = out[f"argmax_val_{part}"]
                index = out[f"argmax_idx_{part}"] + offset
                if best_val is None:
                    best_val, best_id = value, index
                else:
                    take = value > best_val
                    best_val = np.where(take, value, best_val)
                    best_id = np.where(take, index, best_id)
                offset += size
            return best_id.astype(np.int64)
        parts = [out[name] for name in entry["outputs"]]
        return np.concatenate(parts, axis=-1)

    # -- the two call paths -------------------------------------------------
    def forward(self, input_ids, start: int):
        """One call of the whole chain; returns logits as (seq, vocab)."""
        hidden = self.embeddings(input_ids)
        hidden = self._blocks(hidden, start, len(input_ids))
        return self.lm_head(hidden)[0]

    def prefill(self, input_ids, start: int = 0):
        """Feed a prompt; returns the log-probability rows for every fed position."""
        ids = list(int(t) for t in np.asarray(input_ids).reshape(-1))
        batch = int(self.params["batch_size"])
        pos = int(start)
        rows = []
        while len(ids) - pos + start >= batch:
            piece = ids[pos - start: pos - start + batch]
            rows.append(self.forward(piece, pos))
            pos += batch
        while pos < start + len(ids):
            rows.append(self.forward([ids[pos - start]], pos))
            pos += 1
        if not rows:
            return np.zeros((0, self.params["vocab_size"]), dtype=np.float32)
        return np.concatenate(rows, axis=0)

    def decode(self, token_id: int, pos: int) -> np.ndarray:
        return self.forward([int(token_id)], pos)[0]

    # -- helpers -----------------------------------------------------------
    @property
    def eos_ids(self) -> list:
        ids = {int(self.meta["tokenizer"]["eos_token_id"])}
        return sorted(ids)

    def stats(self) -> dict:
        return {"load_seconds": self.load_seconds,
                "predict_seconds": self.predict_seconds,
                "predict_calls": self.predict_calls}


# ---------------------------------------------------------------------------
# Tokenizer and sampling
# ---------------------------------------------------------------------------

#: ChatML with an open reasoning block -- the documented fallback (see
#: FALLBACK_TEMPLATE in dory_converter): Dory's weights carry the special tokens
#: but ship no template, and NVIDIA's README runs llama.cpp with reasoning on.
CHATML_FALLBACK = (
    "{% if messages[0]['role'] == 'system' %}<|im_start|>system\n"
    "{{ messages[0]['content'] }}<|im_end|>\n{% endif %}"
    "{% for message in messages %}"
    "{% if not (loop.first and message['role'] == 'system') %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking %} thinking\n{% endif %}{% endif %}"
)


def render_template(template_text: str, messages, reasoning: bool = True) -> str:
    """Render a Jinja chat template with the variables a HF template expects."""
    from jinja2 import Environment, BaseLoader

    environment = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True,
                              keep_trailing_newline=True)
    template = environment.from_string(template_text)
    return template.render(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=bool(reasoning),
        bos_token="<s>",
        eos_token="<|im_end|>",
    )


class ChatTokenizer:
    """The BF16 repository's tokenizer and chat template (spec D9).

    Order of preference: a chat_template.jinja beside the models (written by the
    converter when the weights had one), then a chat_template inside
    tokenizer_config.json, then the documented ChatML fallback -- the ticket
    needs reasoning on by default either way, and the prompt format is recorded
    in the meta file rather than hidden in code.
    """

    def __init__(self, model_dir, meta):
        from transformers import AutoTokenizer

        self.meta = meta
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.template_text, self.template_source = self._resolve_template()

    def _resolve_template(self):
        name = (self.meta.get("tokenizer") or {}).get("chat_template")
        if name:
            path = self.model_dir / name
            if path.is_file():
                return path.read_text(encoding="utf-8"), str(path.name)
        config = getattr(self.tokenizer, "chat_template", None)
        if isinstance(config, str) and config.strip():
            return config, "tokenizer_config.json"
        fallback = (self.meta.get("tokenizer") or {}).get("chat_template_fallback")
        return CHATML_FALLBACK, fallback or "chatml-think"

    def apply_chat_template(self, messages, reasoning: bool = True) -> str:
        """Render the conversation with reasoning on by default."""
        return render_template(self.template_text, messages, reasoning=reasoning)

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tokenizer.encode(text, add_special_tokens=False),
                          dtype=np.int32)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(int(i) for i in ids), skip_special_tokens=True)

    @property
    def eos_ids(self) -> list:
        ids = set(int(t) for t in (self.meta["tokenizer"].get("eos_token_ids") or []))
        if not ids:
            value = self.tokenizer.eos_token_id
            if isinstance(value, (list, tuple)):
                ids.update(int(v) for v in value)
            elif value is not None:
                ids.add(int(value))
        ids.add(int(self.meta["tokenizer"]["eos_token_id"]))
        return sorted(ids)


def sample_token(logits, temperature: float = 0.0, top_p: float = 1.0, rng=None):
    """Greedy at temperature 0, otherwise temperature + nucleus sampling."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature is None or temperature <= 0:
        return int(np.argmax(logits))
    rng = rng or np.random.default_rng()
    scaled = logits / float(temperature)
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        order = np.argsort(-probs)
        cumulative = np.cumsum(probs[order])
        keep = order[: int(np.searchsorted(cumulative, float(top_p)) + 1)]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()
    return int(rng.choice(probs.shape[0], p=probs))


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def build_messages(prompt: str, system: str = None) -> list:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def generate(runtime, tokenizer, prompt, max_tokens=256, temperature=0.0, top_p=1.0,
             reasoning=True, system=None, stream=True, out=sys.stdout):
    """Prefill the prompt, then stream tokens until EOS or max_tokens."""
    text = tokenizer.apply_chat_template(build_messages(prompt, system), reasoning=reasoning)
    prompt_ids = tokenizer.encode(text)
    context = int(runtime.params["context_length"])
    if len(prompt_ids) >= context - 1:
        raise ValueError(f"the prompt is {len(prompt_ids)} tokens, the converted "
                         f"context is {context}")
    runtime.reset_state()
    started = time.time()
    logits = runtime.prefill(prompt_ids, 0)
    prefill_seconds = time.time() - started
    generated = []
    printing = 0
    pos = len(prompt_ids)
    eos = set(tokenizer.eos_ids)
    while len(generated) < max_tokens and pos < context - 1:
        next_id = sample_token(logits[-1], temperature, top_p)
        if next_id in eos:
            break
        generated.append(next_id)
        pos += 1
        if stream:
            piece = tokenizer.decode(generated)
            if len(piece) > printing:
                out.write(piece[printing:])
                out.flush()
                printing = len(piece)
        if len(generated) >= max_tokens or pos >= context - 1:
            break
        logits = runtime.decode(next_id, pos).reshape(1, -1)
    seconds = time.time() - started
    if stream:
        out.write("\n")
    return {
        "text": tokenizer.decode(generated),
        "prompt_tokens": int(len(prompt_ids)),
        "generated_tokens": int(len(generated)),
        "prefill_seconds": prefill_seconds,
        "seconds": seconds,
        "decode_tok_s": (len(generated) / max(seconds - prefill_seconds, 1e-9)),
        "loops": runtime.loops,
    }


def run_session(args) -> int:
    """One prompt, or an interactive session with /loops and /reset."""
    runtime = DoryRuntime(args.model_dir, verbose=args.verbose)
    tokenizer = ChatTokenizer(Path(args.model_dir), runtime.meta)
    runtime.set_loops(args.loops)
    print(f"loops {runtime.loops}, context {runtime.params['context_length']}, "
          f"reasoning {'on' if not args.no_think else 'off'}", file=sys.stderr)

    if args.prompt:
        result = generate(runtime, tokenizer, args.prompt, args.max_tokens,
                          args.temperature, args.top_p, not args.no_think,
                          args.system, args.stream)
        print(json.dumps({k: v for k, v in result.items() if k != "text"}, indent=2),
              file=sys.stderr)
        return 0

    print("commands: /loops N  /reset  /quit", file=sys.stderr)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/reset":
            runtime.reset_state()
            print("context and KV cache reset", file=sys.stderr)
            continue
        if line.startswith("/loops"):
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("usage: /loops 1|2|3|4", file=sys.stderr)
                continue
            wanted = int(parts[1])
            changed = runtime.set_loops(wanted)
            if changed:
                print(f"loop count {wanted}; KV cache reset -- older rows keep the "
                      f"previous loop count's K/V, so the conversation starts empty "
                      f"(spec D4)", file=sys.stderr)
            else:
                print(f"loop count already {wanted}; cache kept", file=sys.stderr)
            continue
        print("dory> ", end="", flush=True)
        generate(runtime, tokenizer, line, args.max_tokens, args.temperature,
                 args.top_p, not args.no_think, args.system, args.stream)


# ---------------------------------------------------------------------------
# Benchmarks and scoring
# ---------------------------------------------------------------------------

def _uptime() -> dict:
    try:
        return {"loadavg": list(open("/proc/loadavg").read().split()[:3])}
    except Exception:
        pass
    try:
        import subprocess
        text = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
        return {"uptime": text}
    except Exception:
        return {}


def benchmark(args) -> int:
    """Decode tok/s at a chosen context, for one or more loop counts."""
    runtime = DoryRuntime(args.model_dir, verbose=args.verbose)
    tokenizer = ChatTokenizer(Path(args.model_dir), runtime.meta)
    context = int(args.context)
    filler = (tokenizer.encode(args.filler_text) if args.filler_text
              else np.ones(context, dtype=np.int32))
    if len(filler) < context:
        filler = np.tile(filler, int(np.ceil(context / max(len(filler), 1))))[:context]
    prompt = filler[:context].astype(np.int32)
    results = []
    for loops in [int(v) for v in str(args.loops).split(",")]:
        runtime.set_loops(loops)
        runtime.reset_state()
        started = time.time()
        logits = runtime.prefill(prompt, 0)
        prefill_seconds = time.time() - started
        window = []
        positions = [context]
        token = int(np.argmax(logits[-1]))
        for step in range(int(args.tokens)):
            started = time.time()
            logits = runtime.decode(token, positions[-1])
            window.append(time.time() - started)
            token = int(np.argmax(logits))
            positions.append(positions[-1] + 1)
        warm = window[len(window) // 4:] or window
        mean = sum(warm) / len(warm)
        results.append({"loops": loops, "context": context,
                        "prefill_seconds": prefill_seconds,
                        "decode_tok_s": 1.0 / mean, "decode_ms": mean * 1e3,
                        "tokens": len(warm), "machine": _uptime()})
        print(json.dumps(results[-1]), file=sys.stderr)
    print(json.dumps(results, indent=2))
    return 0


def score(args) -> int:
    """Score the Core ML pipeline in decode mode against a reference (spec gates)."""
    from anemll.eval.kld import score_files, write_candidate

    runtime = DoryRuntime(args.model_dir, verbose=args.verbose)
    tokens_path = Path(args.tokens)
    ids = np.loadtxt(tokens_path, dtype=np.int64).reshape(-1).astype(np.int32)
    chunk = int(args.chunk_size)
    n_chunks = len(ids) // chunk
    if n_chunks < 1:
        raise ValueError(f"{len(ids)} tokens is shorter than one {chunk}-token window")
    runtime.set_loops(int(args.loops))
    logprobs, positions = [], []
    for index in range(n_chunks):
        start = index * chunk
        if args.max_windows and index >= int(args.max_windows):
            break
        runtime.reset_state()
        logits = runtime.prefill(ids[start:start + chunk], 0)
        lo, hi = chunk // 2, chunk - 1
        block = np.asarray(logits[lo:hi], dtype=np.float32)
        block -= block.max(axis=-1, keepdims=True)
        block -= np.log(np.exp(block).sum(axis=-1, keepdims=True))
        logprobs.append(block)
        positions.append(np.arange(start + lo, start + hi, dtype=np.int32))
        if args.verbose:
            print(f"window {index + 1}/{n_chunks}", file=sys.stderr)
    payload = {
        "model": "dory-coreml",
        "loops": int(args.loops),
        "kv_mode": "last_loop",
        "batch_size": int(runtime.params["batch_size"]),
        "n_ctx": chunk,
        "trace": "coreml",
        "machine": _uptime(),
    }
    candidate = write_candidate(args.out, np.concatenate(logprobs),
                                np.concatenate(positions), ids, payload)
    print(f"wrote {candidate}", file=sys.stderr)
    if args.reference:
        result = score_files(args.reference, candidate)
        print(result.summary())
        print(json.dumps(result.as_dict(), indent=2))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dory on the Neural Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="talk to a converted model")
    chat.add_argument("--model-dir", required=True)
    chat.add_argument("--prompt", default=None, help="one prompt, then exit")
    chat.add_argument("--loops", type=int, default=2, help="loop count at start (1-4)")
    chat.add_argument("--max-tokens", type=int, default=256)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.add_argument("--top-p", type=float, default=1.0)
    chat.add_argument("--system", default=None)
    chat.add_argument("--no-think", action="store_true", help="turn reasoning off")
    chat.add_argument("--no-stream", dest="stream", action="store_false")
    chat.add_argument("--verbose", action="store_true")
    chat.set_defaults(func=run_session)

    bench = sub.add_parser("bench", help="decode tok/s at a context and loop counts")
    bench.add_argument("--model-dir", required=True)
    bench.add_argument("--context", type=int, default=1024)
    bench.add_argument("--loops", default="1,2,4")
    bench.add_argument("--tokens", type=int, default=32)
    bench.add_argument("--filler-text", default=None)
    bench.add_argument("--verbose", action="store_true")
    bench.set_defaults(func=benchmark)

    sc = sub.add_parser("score", help="score this pipeline against a decode reference")
    sc.add_argument("--model-dir", required=True)
    sc.add_argument("--tokens", default="tests/fixtures/dory_eval/eval_tokens.txt")
    sc.add_argument("--chunk-size", type=int, default=1024)
    sc.add_argument("--max-windows", type=int, default=0)
    sc.add_argument("--loops", type=int, default=2)
    sc.add_argument("--out", required=True)
    sc.add_argument("--reference", default=None)
    sc.add_argument("--verbose", action="store_true")
    sc.set_defaults(func=score)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
