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


# Gemma 4 chat-template markers (GGUF USER_DEFINED / CONTROL tokens).
GEMMA4_REQUIRED = ("<|turn>", "<turn|>", "<|channel>", "<channel|>")


def has_gemma4_template(bpe):
    return all(t in bpe.id_of for t in GEMMA4_REQUIRED)


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
    elif has_gemma4_template(bpe):
        # Gemma 4: "<|turn>ROLE\n<text><turn|>\n" turns; thinking rides in a
        # "<|channel>thought\n...<channel|>" channel inside the model turn.
        think_on = _thinking_level(reasoning_effort) > 0.0
        out = []
        if getattr(bpe, "add_bos_token", False) and "<bos>" in bpe.id_of:
            out.append("<bos>")
        i = 0
        sys_parts = []
        while i < len(messages) and str(messages[i].get("role", "user")) \
                in ("system", "developer"):
            sys_parts.append(_flatten_content(messages[i].get("content")))
            i += 1
        if sys_parts or think_on:
            out.append("<|turn>system\n")
            if think_on and "<|think|>" in bpe.id_of:
                out.append("<|think|>")
            out.append("\n\n".join(sys_parts))
            out.append("<turn|>\n")
        for msg in messages[i:]:
            role = str(msg.get("role", "user"))
            text = _flatten_content(msg.get("content"))
            g_role = "model" if role == "assistant" else "user"
            rc = msg.get("reasoning_content")
            if role == "assistant" and isinstance(rc, str) and rc:
                out.append("<|turn>model\n<|channel>thought\n" + rc
                           + "\n<channel|>" + text + "<turn|>\n")
            else:
                out.append("<|turn>" + g_role + "\n" + text + "<turn|>\n")
        out.append("<|turn>model\n")
        if not think_on:
            # Empty thought channel == "answer directly, no thinking".
            out.append("<|channel>thought\n<channel|>")
        return "".join(out)
    # Fallback template: works with any tokenizer (no special tokens needed).
    out = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        out.append("%s: %s\n" % (role, _flatten_content(msg.get("content"))))
    out.append("assistant:")
    return "".join(out)


def format_static_chat_prefix_text(messages, bpe, reasoning_effort="none", tools=None):
    """Static head STRING of the Inkling prompt (see split_static_chat_prefix)
    WITHOUT tokenizing — plain concatenation only. Returns "" when the GGUF
    has no Inkling template. Milestone 14: lets the warm path skip every
    bpe.encode() call for an unchanged prefix."""
    if not has_inkling_template(bpe):
        return ""
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
    prefix = format_static_chat_prefix_text(messages, bpe, reasoning_effort,
                                            tools=tools)
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
    add(bpe.id_of.get("<turn|>"))   # Gemma 4 end-of-turn
    add(bpe.id_of.get("<eos>"))     # Gemma 4 / llama EOS
    add(bpe.eos_token_id)
    add(bpe.eot_token_id)
    return ids


class EngineWorker:
    """One persistent `m64run --worker` subprocess; a lock serializes GEN
    commands so concurrent HTTP requests never interleave on the KV cache."""

    def __init__(self, engine, pack, backend, ctx, arena_gb, io_mode,
                 kv_type="f16", worker_cmd=None, extra_args=None):
        if worker_cmd:
            cmd = shlex.split(worker_cmd)
        else:
            cmd = [engine, pack, "--worker", "--backend", backend,
                   "--ctx", str(ctx), "--arena-gb", str(arena_gb), "--io", io_mode,
                   "--kv-type", kv_type]
            if extra_args:
                cmd += list(extra_args)
        self.cmd = cmd
        self.lock = threading.Lock()
        self._closed = False
        # Only the real m64run binary understands SIGUSR1 GEN cancellation;
        # a custom --worker-cmd (tests) may not, so never signal it.
        self._can_sigusr1 = (worker_cmd is None)
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

    def generate(self, prompt_ids, max_tokens, temperature, top_p, seed,
                 eos_ids, presence_penalty=0.0, frequency_penalty=0.0,
                 repetition_penalty=1.0, penalty_last_n=256):
        """Yield ("tok", id) then ("done", n_gen, elapsed_ms, finish_reason).

        The worker lock is held for the whole exchange. If the consumer
        abandons the generator mid-stream (client disconnect / stop), the
        finally below cancels the GEN (SIGUSR1 on the real m64run) and drains
        stdout up to DONE/ERR *before* releasing the lock, so the worker is
        back at its command prompt and the pipe never fills up to deadlock.

        Penalties (Milestone 8B) are appended as optional k=v tokens before
        "--"; with all penalties disabled the line stays in the legacy
        format byte-for-byte (backward compat with mock/old workers).
        """
        with self.lock:
            if self._closed or self.proc.poll() is not None:
                raise RuntimeError("worker is not running")
            pipe = self.proc.stdout
            if pipe is None:
                raise RuntimeError("worker has no stdout pipe")
            eos = ",".join(str(i) for i in eos_ids) if eos_ids else "-1"
            pen_opts = []
            if presence_penalty:
                pen_opts.append("presence_penalty=%r" % float(presence_penalty))
            if frequency_penalty:
                pen_opts.append("frequency_penalty=%r" % float(frequency_penalty))
            if repetition_penalty != 1.0:
                pen_opts.append("repetition_penalty=%r" % float(repetition_penalty))
            if penalty_last_n != 256:
                pen_opts.append("penalty_last_n=%d" % int(penalty_last_n))
            pen_seg = (" ".join(pen_opts) + " ") if pen_opts else ""
            line = "GEN %d %s %s %d %s %s-- %s\n" % (
                int(max_tokens), repr(float(temperature)), repr(float(top_p)),
                int(seed) & 0xFFFFFFFFFFFFFFFF, eos, pen_seg,
                " ".join(str(i) for i in prompt_ids))
            try:
                self.proc.stdin.write(line)
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                raise RuntimeError("worker pipe closed")
            finished = False
            try:
                while True:
                    out = pipe.readline()
                    if not out:
                        finished = True  # EOF: nothing left to drain
                        raise RuntimeError("worker exited mid-generation")
                    parts = out.split()
                    if not parts:
                        continue
                    if parts[0] == "TOK":
                        yield ("tok", int(parts[1]))
                    elif parts[0] == "DONE":
                        finished = True
                        yield ("done", int(parts[1]), float(parts[2]), parts[3])
                        return
                    elif parts[0] == "ERR":
                        finished = True
                        raise RuntimeError("worker error: %s"
                                           % (parts[1] if len(parts) > 1
                                              else "unknown"))
            finally:
                # Consumer left before DONE/ERR/EOF (GeneratorExit from
                # gen.close(), or an exception raised out of the loop):
                # tell the real worker to stop emitting, then consume every
                # remaining line up to its terminal DONE/ERR, still holding
                # the lock so the next request never sees stale tokens.
                if not finished and self.proc.poll() is None:
                    if self._can_sigusr1:
                        try:
                            self.proc.send_signal(signal.SIGUSR1)
                        except OSError:
                            pass  # exited in the meantime
                    while True:
                        try:
                            out = pipe.readline()
                        except (OSError, ValueError):
                            break
                        if not out:
                            break
                        parts = out.split()
                        if parts and parts[0] in ("DONE", "ERR"):
                            break

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

    yield_fn(kind, payload) is called with kind in {"delta", "reasoning_delta",
    "tool_call"}.
    """
    tid_text = bpe.id_of.get("<|content_text|>")
    tid_think = bpe.id_of.get("<|content_thinking|>")
    tid_tool_json = bpe.id_of.get("<|content_invoke_tool_json|>")
    tid_end_msg = bpe.id_of.get("<|end_message|>")
    tid_msg_model = bpe.id_of.get("<|message_model|>")
    # Gemma 4 channel routing: <|channel>NAME\n ... <channel|> (thinking
    # arrives live as reasoning_delta, exactly like the Inkling path).
    g4 = has_gemma4_template(bpe)
    tid_ch_open = bpe.id_of.get("<|channel>")
    tid_ch_close = bpe.id_of.get("<channel|>")

    def run(prompt_ids, max_tokens, temperature, top_p, seed, stop_strings,
            yield_fn, structured=False, presence_penalty=0.0,
            frequency_penalty=0.0, repetition_penalty=1.0):
        eos_ids = resolve_eos_ids(bpe, structured=structured)
        stops = [s for s in (stop_strings or []) if isinstance(s, str) and s]
        hold = max((len(s) for s in stops), default=0)
        dec = StreamDecoder(bpe, skip_special_tokens=False)
        # Only forward penalties when active so legacy 6-arg worker objects
        # (test mocks) keep working unchanged.
        gen_kwargs = {}
        if presence_penalty or frequency_penalty or repetition_penalty != 1.0:
            gen_kwargs = {"presence_penalty": float(presence_penalty),
                          "frequency_penalty": float(frequency_penalty),
                          "repetition_penalty": float(repetition_penalty)}
        gen = worker.generate(prompt_ids, max_tokens, temperature, top_p,
                              seed, eos_ids, **gen_kwargs)
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
        # Gemma 4 channel state: "text" | "ch_header" | "thinking" | "other"
        g4_state = "text"
        g4_ch_buf = []

        def emit_text(piece):
            if piece:
                text.append(piece)
                yield_fn("delta", piece)

        def emit_reasoning(piece):
            """Stream thinking tokens out live (like llama.cpp/Splash
            delta.reasoning_content) so clients never see a dead-air gap >60s
            while the model is thinking; also accumulate for the final
            reasoning_content field."""
            if piece:
                reasoning.append(piece)
                yield_fn("reasoning_delta", piece)

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
                    if not g4:
                        if 0 <= tid < len(bpe.tokens) \
                                and tid not in bpe.special_ids:
                            push_text(dec.push(tid))
                        continue
                    # Gemma 4 channel routing by exact token id.
                    if tid == tid_ch_open:
                        tail = dec.flush()
                        if tail:
                            push_text(tail)
                        reset_seg()
                        g4_state = "ch_header"
                        g4_ch_buf = []
                    elif tid == tid_ch_close:
                        tail = dec.flush()
                        if g4_state == "thinking" and tail:
                            emit_reasoning(tail)
                        reset_seg()
                        g4_state = "text"
                    elif tid in bpe.special_ids:
                        continue
                    else:
                        piece = dec.push(tid)
                        if not piece:
                            continue
                        if g4_state == "ch_header":
                            g4_ch_buf.append(piece)
                            joined = "".join(g4_ch_buf)
                            if "\n" in joined:
                                hdr, rest = joined.split("\n", 1)
                                g4_ch_buf = []
                                g4_state = ("thinking"
                                            if hdr.strip() == "thought"
                                            else "other")
                                if g4_state == "thinking" and rest:
                                    emit_reasoning(rest)
                        elif g4_state == "thinking":
                            emit_reasoning(piece)
                        elif g4_state == "text":
                            push_text(piece)
                        # "other": non-thinking channel -> content ignored.
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
                        emit_reasoning(tail)
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
                        emit_reasoning(piece)
                    elif state == "tool_json":
                        tool_buf.append(piece)

            # Drain final incremental decoder bytes on DONE.
            tail = dec.flush()
            if g4:
                if g4_state == "thinking" and tail:
                    emit_reasoning(tail)
                elif g4_state == "text":
                    push_text(tail)
                    flush_text()
                # "ch_header"/"other": incomplete channel header or ignored.
            elif not structured or state == "text":
                push_text(tail)
                flush_text()
            elif state == "thinking" and tail:
                emit_reasoning(tail)
            elif state == "tool_json":
                finish_tool_segment(tail)

            if stop_hit:
                finish = "stop"
            if tool_calls and finish == "stop":
                finish = "tool_calls"
        except _Stop:  # pragma: no cover - client disconnected mid-SSE
            finish = "stop" if stop_hit else ("tool_calls" if tool_calls else "length")
        finally:
            # Close the worker generator deterministically on EVERY exit path
            # (BrokenPipeError/OSError from yield_fn on client disconnect,
            # _Stop, or any other exception): EngineWorker.generate's finally
            # then cancels + drains stdout to DONE before the lock is released,
            # instead of leaving the worker emitting into an unread pipe.
            gen.close()
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
    # SSE per-token chunks are tiny (< MTU): with Nagle on, the kernel holds
    # each chunk waiting for the peer's ACK of the previous one, which makes
    # the stream visibly bursty in clients. TCP_NODELAY sends every write now.
    disable_nagle_algorithm = True
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
        raw_presence = body.get("presence_penalty")
        raw_frequency = body.get("frequency_penalty")
        raw_repetition = body.get("repetition_penalty")
        top_p = float(0.95 if raw_top_p is None else raw_top_p)
        top_p = max(1e-6, min(1.0, top_p))
        # Milestone 8B: penalties default from server flags (body wins; an
        # explicit 0.0 in the body disables a server default too).
        presence = float(getattr(self.cfg, "default_presence_penalty", 0.0)
                         if raw_presence is None else raw_presence)
        frequency = float(0.0 if raw_frequency is None else raw_frequency)
        repetition = float(1.0 if raw_repetition is None else raw_repetition)
        if repetition <= 0.0:
            repetition = 1.0  # nonsensical <= 0 disables (worker rejects it)
        return {
            "max_tokens": max(1, int(32768 if raw_mt is None else raw_mt)),
            "temperature": max(0.0, float(getattr(self.cfg, "default_temp", 0.0)
                                          if raw_temp is None else raw_temp)),
            "top_p": top_p,
            "presence_penalty": max(-2.0, min(2.0, presence)),
            "frequency_penalty": max(-2.0, min(2.0, frequency)),
            "repetition_penalty": repetition,
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
        survives across brand-new conversations sharing one system prompt.
        Milestone 14: the unchanged-prefix fast path is a pure string compare
        — zero bpe.encode() AND zero worker.warm() per turn."""
        prefix_text = format_static_chat_prefix_text(
            msgs, self.cfg.bpe, p["reasoning_effort"], tools=tools)
        if not prefix_text:
            return  # no Inkling template
        prefix_ids = None
        with self.cfg.warm_lock:
            if self.cfg.last_warm_prefix == prefix_text:
                if not getattr(self.cfg, "warm_needs_sync", False):
                    return  # pinned + in sync: nothing to do at all
                ids = getattr(self.cfg, "last_warm_ids", None)
                if ids and getattr(self.cfg, "last_warm_len", 0) >= 64:
                    # Re-sync the worker with the cached ids — no re-tokenize.
                    prefix_ids = ids
                    self.cfg.warm_needs_sync = False
        if prefix_ids is None:
            try:
                prefix_ids = self.cfg.bpe.encode(prefix_text)
            except ValueError:
                if getattr(self.cfg, "last_warm_len", 0) >= 1024:
                    self.cfg.warm_needs_sync = True
                return
            if len(prefix_ids) < 64 or len(prefix_ids) >= self.cfg.ctx:
                # Ultra-tiny prefixes bail before the <512 guard below; if a long
                # static warm is pinned, flag re-sync anyway — this GEN resets
                # the worker's warm_state, so the next long turn must re-WARM.
                if len(prefix_ids) < 64 and \
                        getattr(self.cfg, "last_warm_len", 0) >= 1024:
                    self.cfg.warm_needs_sync = True
                return
            with self.cfg.warm_lock:
                last_len = getattr(self.cfg, "last_warm_len", 0)
                # Don't let tiny background prompts (<512 tok) clobber an active
                # long static warm prefix (>=1024 tok); re-sync on the next long turn.
                if last_len >= 1024 and len(prefix_ids) < 512:
                    self.cfg.warm_needs_sync = True
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
                elif kind == "reasoning_delta" and payload and chat:
                    # OpenAI-compatible thinking stream (same field as
                    # llama.cpp/Splash): keeps data: chunks flowing while the
                    # model thinks, so stale-stream watchdogs never fire.
                    w(chunk_obj({"reasoning_content": payload}, None))
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
                    p["stop"], yield_fn, structured=structured,
                    presence_penalty=p["presence_penalty"],
                    frequency_penalty=p["frequency_penalty"],
                    repetition_penalty=p["repetition_penalty"])
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
            p["stop"], yield_fn, structured=structured,
            presence_penalty=p["presence_penalty"],
            frequency_penalty=p["frequency_penalty"],
            repetition_penalty=p["repetition_penalty"])
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
    ap.add_argument("--kv-type", choices=["f16", "q8_0"], default="f16",
                    help="worker KV cache dtype (q8_0: Metal only, ~47%% of FP16 bytes)")
    ap.add_argument("--prefix-cache", type=int, default=None,
                    help="worker LRU prefix checkpoints (0=off); default: "
                         "let m64run auto-select. Sent to m64run only.")
    ap.add_argument("--cache-dir", default=None,
                    help="SSD-persistent prefix cache dir (.m64kv files kept "
                         "across restarts); sent to m64run only.")
    ap.add_argument("--model-name", default="takuan-inkling-2-mini")
    ap.add_argument("--speculative", type=int, default=0,
                    help="prompt-lookup speculative decoding draft cap "
                         "0..8; 0 = disabled (greedy parity)")
    ap.add_argument("--draft-tokens", type=int, default=0,
                    help="alias for --speculative (overrides it when > 0)")
    ap.add_argument("--worker-cmd",
                    help="override worker command (shlex-split; for mock testing)")
    ap.add_argument("--default-temp", type=float, default=0.0,
                    help="temperature used when the request body omits it "
                         "(default 0.0 = greedy, legacy behavior)")
    ap.add_argument("--default-presence-penalty", type=float, default=0.0,
                    help="presence_penalty used when the request body omits "
                         "it (default 0.0 = disabled); > 0 discourages "
                         "repeats of recently generated tokens")
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
    if not 0 <= a.speculative <= 8:
        ap.error("--speculative must be 0..8")
    if not 0 <= a.draft_tokens <= 8:
        ap.error("--draft-tokens must be 0..8")
    spec_k = a.draft_tokens if a.draft_tokens > 0 else a.speculative
    if spec_k > 0:
        prefix_extra += ["--speculative", str(spec_k)]
    worker = EngineWorker(a.engine, a.pack, a.backend, a.ctx, a.arena_gb, a.io,
                          kv_type=a.kv_type,
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
                             default_temp=a.default_temp,
                             default_presence_penalty=a.default_presence_penalty,
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
