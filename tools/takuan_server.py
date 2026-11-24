#!/usr/bin/env python3
"""takuan_server.py — OpenAI-compatible HTTP backend server for the
TakuAn-MoE engine (zero external dependencies, Python 3 stdlib only).

It keeps ONE persistent `build/m64run <pack> --worker` subprocess alive
(line protocol: READY / GEN ... / TOK / DONE / ERR / QUIT) and exposes:

  GET  /health                      liveness + model id + ctx
  GET  /v1/models                   OpenAI model list
  POST /v1/chat/completions         chat completions (stream + non-stream)
  POST /v1/completions              legacy completions (stream + non-stream)

Tokenization/detokenization run in-process via tools/bpe_gguf.py (stdlib GGUF
metadata BPE), so token streaming decodes incrementally: multi-byte UTF-8
characters split across byte-level tokens never surface as replacement chars,
and `stop` strings are trimmed client-side with a holdback buffer.

Example:
  python3 tools/takuan_server.py --pack OUT.m64pack --gguf IN.gguf \
      --port 8081 --ctx 2048
  # bounded-RAM streaming variant:  --arena-gb 12 --io mtlio
"""
import argparse
import atexit
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bpe_gguf import BPE, StreamDecoder  # noqa: E402

# Inkling-2-Mini chat-template special tokens (GGUF CONTROL, token_type == 3).
INKLING_ROLE_TOKEN = {
    "system": "<|message_system|>",
    "user": "<|message_user|>",
    "assistant": "<|message_model|>",
    "tool": "<|message_tool|>",
}
INKLING_REQUIRED = ("<|message_system|>", "<|message_user|>", "<|message_model|>",
                    "<|content_text|>", "<|end_message|>",
                    "<|content_model_end_sampling|>")
INKLING_TOOL_TOKENS = ("<|content_xml|>", "<|content_invoke_tool_json|>")
# OpenAI reasoning_effort -> Inkling "Thinking effort level" in [0, 0.99].
REASONING_LEVELS = {"none": 0.0, "minimal": 0.1, "low": 0.2, "medium": 0.7,
                    "high": 0.9, "xhigh": 0.99, "max": 0.99}


def _thinking_level(reasoning_effort):
    if reasoning_effort is None:
        return 0.0
    key = str(reasoning_effort).lower()
    if key in REASONING_LEVELS:
        return REASONING_LEVELS[key]
    try:
        v = float(key)
    except ValueError:
        return 0.0
    return max(0.0, min(0.99, v))


def _fmt_level(v):
    return ("%g" % v) if v >= 0 else "0"


def _flatten_content(content):
    """OpenAI `content` may be a string or a list of parts
    ({"type": "text", "text": ...}); flatten to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts)
    return str(content)


def _parse_json_obj(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            v = json.loads(raw)
            if isinstance(v, dict):
                return v
        except ValueError:
            pass
    return {}


def _resolve_tool_name(msg, prior_calls):
    name = msg.get("name")
    if isinstance(name, str) and name:
        return name
    tcid = msg.get("tool_call_id")
    if isinstance(tcid, str) and tcid in prior_calls:
        return prior_calls[tcid]
    return ""


def _tool_declare_segment(tools):
    specs = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else t
        name = fn.get("name")
        if not isinstance(name, str) or not name:
            continue
        specs.append({
            "description": fn.get("description") or "",
            "name": name,
            "parameters": fn.get("parameters") or {},
            "type": t.get("type") or "function",
        })
    if not specs:
        return ""
    return ("<|message_system|>tool_declare<|content_xml|>"
            + json.dumps(specs, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False)
            + "<|end_message|>")


def has_inkling_template(bpe):
    return all(t in bpe.special_tokens for t in INKLING_REQUIRED)


def has_inkling_tools(bpe):
    return has_inkling_template(bpe) and all(
        t in bpe.special_tokens for t in INKLING_TOOL_TOKENS)


def is_structured_mode(bpe, tools, reasoning_effort):
    if not has_inkling_template(bpe):
        return False
    return bool(tools) or _thinking_level(reasoning_effort) > 0.0


def format_chat_prompt(messages, bpe, reasoning_effort="none", tools=None):
    """Render OpenAI `messages` (and optional `tools`) into the prompt string.

    Inkling-2-Mini GGUFs carry <|message_*|> CONTROL tokens; emits the same
    tool_declare / tool_calls / tool / thinking-effort framing as the GGUF's
    chat_template. Any GGUF without them falls back to plain "role: content".
    """
    level = _thinking_level(reasoning_effort)
    effort_line = ("<|message_system|><|content_text|>Thinking effort level: %s"
                   "<|end_message|>" % _fmt_level(level))
    if has_inkling_template(bpe):
        out = []
        if tools and has_inkling_tools(bpe):
            decl = _tool_declare_segment(tools)
            if decl:
                out.append(decl)
        effort_done = False
        prior_calls = {}
        for msg in messages:
            role = str(msg.get("role", "user"))
            text = _flatten_content(msg.get("content"))
            if not effort_done and role != "system":
                out.append(effort_line)
                effort_done = True
            if role == "tool":
                tname = _resolve_tool_name(msg, prior_calls)
                rtok = "<|message_tool|>" if "<|message_tool|>" in bpe.special_tokens \
                    else INKLING_ROLE_TOKEN["user"]
                out.append(rtok + tname + "<|content_text|>" + text
                           + "<|end_message|>")
                continue
            rtok = INKLING_ROLE_TOKEN.get(role, INKLING_ROLE_TOKEN["user"])
            if role == "assistant":
                rc = msg.get("reasoning_content")
                if isinstance(rc, str) and rc:
                    out.append(rtok + "<|content_thinking|>" + rc
                               + "<|end_message|>")
                tcs = msg.get("tool_calls") or []
                if text or (not tcs and not rc):
                    out.append(rtok + "<|content_text|>" + text
                               + "<|end_message|>")
                if isinstance(tcs, list):
                    for tc in tcs:
                        if not isinstance(tc, dict):
                            continue
                        fn = tc.get("function") or {}
                        fname = fn.get("name")
                        if not isinstance(fname, str) or not fname:
                            continue
                        if isinstance(tc.get("id"), str):
                            prior_calls[tc["id"]] = fname
                        args_obj = _parse_json_obj(fn.get("arguments"))
                        payload = ('{"name":%s,"args":%s}'
                                   % (json.dumps(fname, ensure_ascii=False),
                                      json.dumps(args_obj, sort_keys=True,
                                                 separators=(",", ":"),
                                                 ensure_ascii=False)))
                        out.append("<|message_model|>" + fname
                                   + "<|content_invoke_tool_json|>" + payload
                                   + "<|end_message|>")
                out.append("<|content_model_end_sampling|>")
            else:
                out.append(rtok + "<|content_text|>" + text + "<|end_message|>")
        if not effort_done:
            out.append(effort_line)
        if is_structured_mode(bpe, tools, reasoning_effort):
            out.append("<|message_model|>")
        else:
            out.append("<|message_model|><|content_text|>")
        return "".join(out)
    # Fallback template: works with any tokenizer (no special tokens needed).
    out = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        out.append("%s: %s\n" % (role, _flatten_content(msg.get("content"))))
    out.append("assistant:")
    return "".join(out)


def split_static_chat_prefix(messages, bpe, reasoning_effort="none", tools=None):
    """Static head of the Inkling prompt: tool_declare + leading system
    messages + thinking-effort line — everything before the FIRST non-system
    message. Returns ("", []) when the GGUF has no Inkling template or the
    head fails to tokenize.

    This is an exact string prefix of format_chat_prompt's output, so the
    server can WARM-pin it in the worker and every turn of every conversation
    reuses it bitwise-exactly (KV + ShortConv prefix caching).
    """
    if not has_inkling_template(bpe):
        return "", []
    out = []
    if tools and has_inkling_tools(bpe):
        decl = _tool_declare_segment(tools)
        if decl:
            out.append(decl)
    level = _thinking_level(reasoning_effort)
    effort_line = ("<|message_system|><|content_text|>Thinking effort level: %s"
                   "<|end_message|>" % _fmt_level(level))
    for msg in messages:
        if str(msg.get("role", "user")) != "system":
            break
        text = _flatten_content(msg.get("content"))
        out.append(INKLING_ROLE_TOKEN["system"] + "<|content_text|>" + text
                   + "<|end_message|>")
    out.append(effort_line)
    prefix = "".join(out)
    try:
        ids = bpe.encode(prefix)
    except ValueError:
        return "", []
    return prefix, ids


def resolve_eos_ids(bpe, structured=False):
    """Token ids that must end generation as finish_reason=stop."""
    ids = []
    end_msg_id = bpe.id_of.get("<|end_message|>")

    def add(i):
        if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(bpe.tokens):
            if structured and i == end_msg_id:
                return
            if i not in ids:
                ids.append(i)

    add(bpe.id_of.get("<|content_model_end_sampling|>"))
    if not structured:
        add(end_msg_id)
    add(bpe.id_of.get("<|endoftext|>"))
    add(bpe.eos_token_id)
    add(bpe.eot_token_id)
    return ids


class EngineWorker:
    """One persistent `m64run --worker` subprocess; a lock serializes GEN
    commands so concurrent HTTP requests never interleave on the KV cache."""

    def __init__(self, engine, pack, backend, ctx, arena_gb, io_mode,
                 worker_cmd=None, extra_args=None):
        if worker_cmd:
            cmd = shlex.split(worker_cmd)
        else:
            cmd = [engine, pack, "--worker", "--backend", backend,
                   "--ctx", str(ctx), "--arena-gb", str(arena_gb), "--io", io_mode]
            if extra_args:
                cmd += list(extra_args)
        self.cmd = cmd
        self.lock = threading.Lock()
        self._closed = False
        # stderr is inherited: engine startup logs (pack / resident bytes /
        # arena report) stay visible in the server console.
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     text=True, bufsize=1)
        line = self.proc.stdout.readline() if self.proc.stdout else ""
        if not line or not line.startswith("READY"):
            rc = self.proc.poll()
            tail = ""
            if rc is not None and self.proc.stdout:
                tail = (self.proc.stdout.read() or "")[-400:].strip()
            raise RuntimeError(
                "worker exited before READY (rc=%s): %s%s"
                % (rc, " ".join(cmd[:4]), (" | " + tail) if tail else ""))
        self.ready = line.strip()

    def generate(self, prompt_ids, max_tokens, temperature, top_p, seed, eos_ids):
        """Yield ("tok", id) then ("done", n_gen, elapsed_ms, finish_reason).

        The worker lock is held for the whole exchange; abandoning the
        generator releases it (the worker stays in sync because the caller
        must drain to DONE — see _iter_events)."""
        with self.lock:
            if self._closed or self.proc.poll() is not None:
                raise RuntimeError("worker is not running")
            eos = ",".join(str(i) for i in eos_ids) if eos_ids else "-1"
            line = "GEN %d %s %s %d %s -- %s\n" % (
                int(max_tokens), repr(float(temperature)), repr(float(top_p)),
                int(seed) & 0xFFFFFFFFFFFFFFFF, eos,
                " ".join(str(i) for i in prompt_ids))
            try:
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                raise RuntimeError("worker pipe closed")
            while True:
                out = self.proc.stdout.readline()
                if not out:
                    raise RuntimeError("worker exited mid-generation")
                parts = out.split()
                if not parts:
                    continue
                if parts[0] == "TOK":
                    yield ("tok", int(parts[1]))
                elif parts[0] == "DONE":
                    yield ("done", int(parts[1]), float(parts[2]), parts[3])
                    return
                elif parts[0] == "ERR":
                    raise RuntimeError("worker error: %s"
                                       % (parts[1] if len(parts) > 1 else "unknown"))

    def warm(self, prompt_ids):
        """Pin a static prefix (tool_declare + system head) in the worker so
        brand-new conversations still hit the prefix cache.

        Returns True on "WARMED <n> computed|cached|disk"; False on
        "WARMED 0 disabled/failed" or if the worker predates the WARM
        command (replies ERR ...) — a legacy worker must not break serving.
        """
        with self.lock:
            if self._closed or self.proc.poll() is not None:
                raise RuntimeError("worker is not running")
            line = "WARM " + " ".join(str(int(i)) for i in prompt_ids) + "\n"
            try:
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
                out = self.proc.stdout.readline()
            except (BrokenPipeError, OSError):
                raise RuntimeError("worker pipe closed")
            if not out:
                raise RuntimeError("worker exited during warm")
            parts = out.split()
            if not parts or parts[0] != "WARMED":
                return False
            # "WARMED <n> computed|cached|disk" -> True; the worker only
            # replies "WARMED 0 disabled" / "WARMED 0 failed" when it cannot
            # warm at all, which the server must treat as a no-op.
            return len(parts) < 2 or parts[-1] not in ("disabled", "failed")

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.proc.poll() is None:
                try:
                    self.proc.stdin.write("QUIT\n")
                    self.proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.terminate()
                    try:
                        self.proc.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.proc.kill()
        except Exception:
            pass


class _Stop(Exception):
    """Internal: stop-string hit while consuming worker events."""


def _parse_tool_invocation(raw_text, header_name):
    """Parse Inkling `<|content_invoke_tool_json|>` payload into an OpenAI
    tool_call object (handles dict/string args and minor trailing junk)."""
    s = raw_text.strip()
    if not s:
        return None
    obj = None
    try:
        obj = json.loads(s)
    except ValueError:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                obj = json.loads(s[start:end + 1])
            except ValueError:
                obj = None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or header_name.strip()
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("args")
    if args is None:
        args = obj.get("arguments", {})
    if isinstance(args, dict):
        args_str = json.dumps(args, ensure_ascii=False)
    elif isinstance(args, str):
        args_str = args
    else:
        args_str = "{}"
    return {
        "id": "call_" + uuid.uuid4().hex[:12],
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


def _make_stream_fn(worker, bpe, ctx):
    """Run one generation over EngineWorker events and apply decoding rules.

    yield_fn(kind, payload) is called with kind in {"delta", "tool_call"}.
    """
    tid_text = bpe.id_of.get("<|content_text|>")
    tid_think = bpe.id_of.get("<|content_thinking|>")
    tid_tool_json = bpe.id_of.get("<|content_invoke_tool_json|>")
    tid_end_msg = bpe.id_of.get("<|end_message|>")
    tid_msg_model = bpe.id_of.get("<|message_model|>")

    def run(prompt_ids, max_tokens, temperature, top_p, seed, stop_strings,
            yield_fn, structured=False):
        eos_ids = resolve_eos_ids(bpe, structured=structured)
        stops = [s for s in (stop_strings or []) if isinstance(s, str) and s]
        hold = max((len(s) for s in stops), default=0)
        dec = StreamDecoder(bpe, skip_special_tokens=False)
        gen = worker.generate(prompt_ids, max_tokens, temperature, top_p,
                              seed, eos_ids)
        text = []
        reasoning = []
        tool_calls = []
        buf = ""
        stop_hit = False
        n_out, ms, finish = 0, 0.0, "length"
        # Structured state: "header" -> "<|content_text|>" / "<|content_thinking|>" / "<|content_invoke_tool_json|>"
        state = "header"
        header_buf = []
        header_name = ""
        tool_buf = []

        def emit_text(piece):
            if piece:
                text.append(piece)
                yield_fn("delta", piece)

        def push_text(chunk):
            nonlocal buf, stop_hit
            if stop_hit or not chunk:
                return
            buf += chunk
            safe = len(buf) - hold
            if safe > 0:
                emit_text(buf[:safe])
                buf = buf[safe:]
            for s in stops:
                pos = buf.find(s)
                if pos >= 0:
                    emit_text(buf[:pos])
                    buf = ""
                    stop_hit = True
                    break

        def flush_text():
            nonlocal buf, stop_hit
            if stop_hit:
                buf = ""
                return
            if buf:
                for s in stops:
                    pos = buf.find(s)
                    if pos >= 0:
                        emit_text(buf[:pos])
                        buf = ""
                        stop_hit = True
                        return
                emit_text(buf)
                buf = ""

        def finish_tool_segment(tail_piece=""):
            raw = "".join(tool_buf) + tail_piece
            tc = _parse_tool_invocation(raw, header_name)
            if tc is not None:
                idx = len(tool_calls)
                tool_calls.append(tc)
                yield_fn("tool_call", (idx, tc))

        def reset_seg():
            nonlocal dec
            dec = StreamDecoder(bpe, skip_special_tokens=False)

        try:
            for ev in gen:
                if ev[0] == "done":
                    n_out, ms, finish = ev[1], ev[2], ev[3]
                    break
                tid = ev[1]
                if not structured:
                    if 0 <= tid < len(bpe.tokens) and tid not in bpe.special_ids:
                        push_text(dec.push(tid))
                    continue

                # Structured Inkling segment routing by exact token id.
                if tid == tid_msg_model:
                    reset_seg()
                    header_buf = []
                    header_name = ""
                    state = "header"
                elif tid == tid_text:
                    reset_seg()
                    header_buf = []
                    state = "text"
                elif tid == tid_think:
                    reset_seg()
                    header_buf = []
                    state = "thinking"
                elif tid == tid_tool_json:
                    header_name = ("".join(header_buf) + dec.flush()).strip()
                    reset_seg()
                    tool_buf = []
                    state = "tool_json"
                elif tid == tid_end_msg:
                    tail = dec.flush()
                    if state == "text":
                        push_text(tail)
                        flush_text()
                    elif state == "thinking" and tail:
                        reasoning.append(tail)
                    elif state == "tool_json":
                        finish_tool_segment(tail)
                    reset_seg()
                    header_buf = []
                    header_name = ""
                    tool_buf = []
                    state = "header"
                elif tid in bpe.special_ids:
                    continue
                else:
                    piece = dec.push(tid)
                    if not piece:
                        continue
                    if state == "header":
                        header_buf.append(piece)
                    elif state == "text":
                        push_text(piece)
                    elif state == "thinking":
                        reasoning.append(piece)
                    elif state == "tool_json":
                        tool_buf.append(piece)

            # Drain final incremental decoder bytes on DONE.
            tail = dec.flush()
            if not structured or state == "text":
                push_text(tail)
                flush_text()
            elif state == "thinking" and tail:
                reasoning.append(tail)
            elif state == "tool_json":
                finish_tool_segment(tail)

            if stop_hit:
                finish = "stop"
            if tool_calls and finish == "stop":
                finish = "tool_calls"
        except _Stop:  # pragma: no cover - client disconnected mid-SSE
            finish = "stop" if stop_hit else ("tool_calls" if tool_calls else "length")
        return {
            "text": "".join(text),
            "reasoning": "".join(reasoning),
            "tool_calls": tool_calls,
            "finish": finish,
            "n_out": n_out,
            "n_in": len(prompt_ids),
            "ms": ms,
        }
    return run


class Handler(BaseHTTPRequestHandler):
    server_version = "TakuAn-MoE/0.1"
    cfg = None  # injected namespace: model, ctx, bpe, worker, run_stream

    # ---------------- plumbing ----------------
    def log_message(self, fmt, *args):
        sys.stderr.write("takuan: %s - %s\n"
                         % (self.address_string(), fmt % args))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message, err_type="invalid_request_error", code_str=None):
        self._json(code, {"error": {"message": message, "type": err_type,
                                    "param": None, "code": code_str}})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n > 0 else b""
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    # ---------------- routes ----------------
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._json(200, {"status": "ok", "model": self.cfg.model,
                             "ctx": self.cfg.ctx})
        elif path == "/v1/models":
            self._json(200, {"object": "list", "data": [{
                "id": self.cfg.model, "object": "model",
                "created": int(self.cfg.created), "owned_by": "takuan-moe"}]})
        else:
            self._error(404, "unknown path %s" % path, code_str="not_found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._body()
        except (ValueError, UnicodeDecodeError):
            self._error(400, "invalid JSON body", code_str="invalid_json")
            return
        try:
            if path == "/v1/chat/completions":
                self._chat(body)
            elif path == "/v1/completions":
                self._completions(body)
            else:
                self._error(404, "unknown path %s" % path, code_str="not_found")
        except RuntimeError as e:
            msg = str(e)
            if "prompt_exceeds_ctx" in msg:
                self._error(400, "prompt exceeds context length",
                            code_str="prompt_exceeds_ctx")
            else:
                self._error(500, msg, err_type="server_error")
        except (BrokenPipeError, ConnectionResetError):
            pass  # client disconnected mid-stream
        except Exception as e:  # pragma: no cover - defensive
            try:
                self._error(500, "internal error: %s" % e, err_type="server_error")
            except Exception:
                pass

    # ---------------- request handling ----------------
    def _params(self, body):
        raw_mt = body.get("max_tokens")
        if raw_mt is None:
            raw_mt = body.get("max_completion_tokens")
        raw_temp = body.get("temperature")
        raw_top_p = body.get("top_p")
        raw_seed = body.get("seed")
        top_p = float(0.95 if raw_top_p is None else raw_top_p)
        top_p = max(1e-6, min(1.0, top_p))
        return {
            "max_tokens": max(1, int(32768 if raw_mt is None else raw_mt)),
            "temperature": max(0.0, float(0.0 if raw_temp is None else raw_temp)),
            "top_p": top_p,
            "seed": int(0 if raw_seed is None else raw_seed),
            "stream": bool(body.get("stream", False)),
            "stop": _as_stop_list(body.get("stop")),
            "reasoning_effort": body.get("reasoning_effort", "none"),
        }

    def _encode_prompt(self, prompt_text):
        try:
            return self.cfg.bpe.encode(prompt_text)
        except ValueError as e:
            raise ValueError("prompt tokenization failed: %s" % e)

    def _chat(self, body):
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs or not all(
                isinstance(m, dict) for m in msgs):
            self._error(400, "'messages' must be a non-empty list of objects",
                        code_str="invalid_messages")
            return
        p = self._params(body)
        tools = body.get("tools")
        if body.get("tool_choice") == "none" or not isinstance(tools, list):
            tools = None
        structured = is_structured_mode(self.cfg.bpe, tools,
                                        p["reasoning_effort"])
        model = body.get("model") or self.cfg.model
        prompt = format_chat_prompt(msgs, self.cfg.bpe, p["reasoning_effort"],
                                    tools=tools)
        self._run(prompt, p, model, chat=True, structured=structured,
                  include_usage=bool((body.get("stream_options") or {})
                                     .get("include_usage", False)),
                  warm_fn=lambda: self._maybe_warm_static_prefix(
                      msgs, p, tools))

    def _maybe_warm_static_prefix(self, msgs, p, tools):
        """Pin the static tool_declare + system head in the worker. The GEN
        LRU only covers repeats of the SAME conversation; this warm_state
        survives across brand-new conversations sharing one system prompt."""
        if not has_inkling_template(self.cfg.bpe):
            return
        prefix_text, prefix_ids = split_static_chat_prefix(
            msgs, self.cfg.bpe, p["reasoning_effort"], tools=tools)
        if len(prefix_ids) < 64 or len(prefix_ids) >= self.cfg.ctx:
            return
        with self.cfg.warm_lock:
            last_len = getattr(self.cfg, "last_warm_len", 0)
            needs_sync = getattr(self.cfg, "warm_needs_sync", False)
            # Don't let tiny background prompts (<512 tok) clobber an active
            # long static warm prefix (>=1024 tok); re-sync on the next long turn.
            if last_len >= 1024 and len(prefix_ids) < 512:
                self.cfg.warm_needs_sync = True
                return
            if self.cfg.last_warm_prefix == prefix_text and not needs_sync:
                return
            self.cfg.last_warm_prefix = prefix_text
            self.cfg.last_warm_len = len(prefix_ids)
            self.cfg.last_warm_ids = prefix_ids
            self.cfg.warm_needs_sync = False
        try:
            self.cfg.worker.warm(prefix_ids)
        except RuntimeError:
            with self.cfg.warm_lock:
                self.cfg.warm_needs_sync = True
            pass  # worker path reports the real error on the GEN itself

    def _completions(self, body):
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = "".join(str(x) for x in prompt)
        if not isinstance(prompt, str):
            self._error(400, "'prompt' must be a string", code_str="invalid_prompt")
            return
        p = self._params(body)
        model = body.get("model") or self.cfg.model
        self._run(prompt, p, model, chat=False, structured=False,
                  include_usage=bool((body.get("stream_options") or {})
                                     .get("include_usage", False)))

    def _run(self, prompt_text, p, model, chat, structured, include_usage,
             warm_fn=None):
        try:
            ids = self._encode_prompt(prompt_text)
        except ValueError as e:
            self._error(400, str(e), code_str="invalid_prompt")
            return
        if len(ids) >= self.cfg.ctx:
            self._error(400, "prompt exceeds context length",
                        code_str="prompt_exceeds_ctx")
            return
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
        created = int(time.time())

        def chunk_obj(delta_obj, finish):
            choice = {"index": 0,
                      ("delta" if chat else "text"): delta_obj,
                      "finish_reason": finish}
            if not chat:
                choice = {"index": 0, "text": delta_obj, "finish_reason": finish}
            return {"id": rid, "object": "chat.completion.chunk" if chat
                    else "text_completion", "created": created,
                    "model": model, "choices": [choice]}

        def prepare_warm():
            if warm_fn:
                warm_fn()
            with self.cfg.warm_lock:
                wid = getattr(self.cfg, "last_warm_ids", None)
                if wid and ids[:len(wid)] != wid:
                    self.cfg.warm_needs_sync = True

        if p["stream"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.close_connection = True

            wlock = threading.Lock()
            last_write = [time.monotonic()]

            def write_raw(raw_bytes):
                with wlock:
                    self.wfile.write(raw_bytes)
                    self.wfile.flush()
                    last_write[0] = time.monotonic()

            def w(obj):
                write_raw(b"data: " + json.dumps(obj, ensure_ascii=False)
                          .encode("utf-8") + b"\n\n")

            first = chunk_obj({"role": "assistant", "content": ""} if chat else "",
                              None)
            w(first)

            try:
                ka_sec = float(os.environ.get("TAKUAN_SSE_KEEPALIVE_SEC", "5.0"))
            except ValueError:
                ka_sec = 5.0
            stop_ka = threading.Event()
            ka_thread = None
            if ka_sec > 0.0:
                poll_sec = min(ka_sec, 0.5)

                def _ka_loop():
                    while not stop_ka.wait(poll_sec):
                        try:
                            with wlock:
                                if stop_ka.is_set():
                                    break
                                if time.monotonic() - last_write[0] < ka_sec:
                                    continue
                                self.wfile.write(b": keepalive\n\n")
                                self.wfile.flush()
                                last_write[0] = time.monotonic()
                        except OSError:
                            stop_ka.set()
                            break

                ka_thread = threading.Thread(target=_ka_loop, daemon=True)
                ka_thread.start()

            def yield_fn(kind, payload):
                if kind == "delta" and payload:
                    w(chunk_obj({"content": payload} if chat else payload, None))
                elif kind == "tool_call" and chat:
                    idx, tc = payload
                    w(chunk_obj({"tool_calls": [{
                        "index": idx,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }]}, None))

            try:
                prepare_warm()
                res = self.cfg.run_stream(
                    ids, p["max_tokens"], p["temperature"], p["top_p"], p["seed"],
                    p["stop"], yield_fn, structured=structured)
            except Exception as e:
                stop_ka.set()
                if ka_thread is not None:
                    ka_thread.join(timeout=1.0)
                try:
                    w({"error": {"message": str(e), "type": "server_error"}})
                    write_raw(b"data: [DONE]\n\n")
                except OSError:
                    pass
                return
            finally:
                stop_ka.set()
                if ka_thread is not None:
                    ka_thread.join(timeout=1.0)

            finish = res["finish"]
            n_in, n_out = res["n_in"], res["n_out"]
            w(chunk_obj({}, finish) if chat else chunk_obj("", finish))
            if include_usage:
                w({"id": rid, "object": "chat.completion.chunk" if chat
                   else "text_completion", "created": created, "model": model,
                   "choices": [],
                   "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                             "total_tokens": n_in + n_out}})
            write_raw(b"data: [DONE]\n\n")
            return

        prepare_warm()

        def yield_fn(kind, payload):
            pass

        res = self.cfg.run_stream(
            ids, p["max_tokens"], p["temperature"], p["top_p"], p["seed"],
            p["stop"], yield_fn, structured=structured)
        n_in, n_out = res["n_in"], res["n_out"]
        if chat:
            msg_obj = {
                "role": "assistant",
                "content": (res["text"] if res["text"] else None)
                if res["tool_calls"] else res["text"],
            }
            if res["tool_calls"]:
                msg_obj["tool_calls"] = res["tool_calls"]
            if res["reasoning"]:
                msg_obj["reasoning_content"] = res["reasoning"]
            obj = {"id": rid, "object": "chat.completion", "created": created,
                   "model": model,
                   "choices": [{"index": 0,
                                "message": msg_obj,
                                "finish_reason": res["finish"]}],
                   "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                             "total_tokens": n_in + n_out}}
        else:
            obj = {"id": rid, "object": "text_completion", "created": created,
                   "model": model,
                   "choices": [{"index": 0, "text": res["text"],
                                "finish_reason": res["finish"]}],
                   "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                             "total_tokens": n_in + n_out}}
        self._json(200, obj)


def _as_stop_list(stop):
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str)]
    return []


def free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    ap = argparse.ArgumentParser(
        description="TakuAn-MoE OpenAI-compatible backend server (stdlib only).")
    ap.add_argument("--pack", help="path to .m64pack (required unless --worker-cmd)")
    ap.add_argument("--gguf", required=True,
                    help="GGUF with tokenizer metadata for in-process BPE")
    ap.add_argument("--engine", default="build/m64run",
                    help="path to m64run (default: build/m64run)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--arena-gb", type=float, default=0.0)
    ap.add_argument("--io", choices=["pread", "mtlio"], default="mtlio")
    ap.add_argument("--backend", choices=["metal", "cpu"], default="metal")
    ap.add_argument("--prefix-cache", type=int, default=None,
                    help="worker LRU prefix checkpoints (0=off); default: "
                         "let m64run auto-select. Sent to m64run only.")
    ap.add_argument("--cache-dir", default=None,
                    help="SSD-persistent prefix cache dir (.m64kv files kept "
                         "across restarts); sent to m64run only.")
    ap.add_argument("--model-name", default="takuan-inkling-2-mini")
    ap.add_argument("--worker-cmd",
                    help="override worker command (shlex-split; for mock testing)")
    a = ap.parse_args()
    if not a.worker_cmd and not a.pack:
        ap.error("--pack is required unless --worker-cmd is set")
    if not os.path.exists(a.gguf):
        ap.error("--gguf not found: %s" % a.gguf)

    bpe = BPE(a.gguf)
    prefix_extra = [] if a.prefix_cache is None \
        else ["--prefix-cache", str(a.prefix_cache)]
    if a.cache_dir:
        prefix_extra += ["--cache-dir", a.cache_dir]
    worker = EngineWorker(a.engine, a.pack, a.backend, a.ctx, a.arena_gb, a.io,
                          worker_cmd=a.worker_cmd, extra_args=prefix_extra)
    atexit.register(worker.close)

    def _sig(signum, frame):
        sys.stderr.write("takuan: signal %d, shutting down\n" % signum)
        worker.close()
        sys.exit(0)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _sig)
        except (ValueError, OSError):
            pass  # not the main thread (test harness)

    cfg = argparse.Namespace(model=a.model_name, ctx=a.ctx, bpe=bpe,
                             worker=worker, created=time.time(),
                             warm_lock=threading.Lock(), last_warm_prefix=None,
                             last_warm_len=0, last_warm_ids=None,
                             warm_needs_sync=False,
                             run_stream=_make_stream_fn(worker, bpe, a.ctx))
    Handler.cfg = cfg
    httpd = ThreadingHTTPServer((a.host, a.port), Handler)
    port = httpd.server_address[1]
    sys.stderr.write("takuan: serving '%s' (ctx=%d) on http://%s:%d/v1\n"
                     % (a.model_name, a.ctx, a.host, port))
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
