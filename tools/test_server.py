#!/usr/bin/env python3
"""test_server.py — automated tests for the TakuAn-MoE OpenAI-compatible
server stack (tokenizer, m64run --worker protocol, HTTP + SSE endpoints).

Zero external dependencies and zero model downloads:
  * tokenizer / StreamDecoder tests run against build/fixture/tiny.gguf
  * worker-protocol tests spawn build/m64run <tiny.m64pack> --worker directly
  * HTTP/SSE tests run tools/takuan_server.py backed by a deterministic
    PYTHON mock worker (multi-byte UTF-8 token streams, EOS, stop strings)
    with a stub tokenizer — full coverage without the 26 GiB pack.

Exit code 0 = all checks pass.
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

PASSED = 0
FAILED = []


def check(name, cond, detail=""):
    global PASSED
    if cond:
        PASSED += 1
        print("  ok   %s" % name)
    else:
        FAILED.append(name)
        print("  FAIL %s %s" % (name, ("| " + str(detail)) if detail else ""))


def _fixture_dir():
    d = os.path.join(ROOT, "build", "fixture")
    if not (os.path.exists(os.path.join(d, "tiny.gguf"))
            and os.path.exists(os.path.join(d, "tiny.m64pack"))):
        tmp = tempfile.mkdtemp(prefix="takuan_fixture_")
        subprocess.run([sys.executable, os.path.join(HERE, "make_test_fixture.py"),
                        tmp], check=True, stdout=subprocess.DEVNULL)
        return tmp
    return d


# --------------------------------------------------------------------------
# 1. BPE + StreamDecoder unit tests
# --------------------------------------------------------------------------
def test_bpe(fix):
    print("[1] BPE / StreamDecoder")
    from bpe_gguf import BPE, StreamDecoder
    b = BPE(os.path.join(fix, "tiny.gguf"))
    check("vocab loads", len(b.tokens) == 33, len(b.tokens))
    check("special token detected", b.special_tokens.get("<|eos|>") == 32,
          b.special_tokens)
    check("eos id from metadata", b.eos_token_id == 32, b.eos_token_id)
    ids = b.encode("hello STOP")
    check("encode roundtrip", b.decode(ids) == "hello STOP", b.decode(ids))
    check("special split encode", b.encode("a<|eos|>d") == [0, 32, 1],
          b.encode("a<|eos|>d"))
    check("skip_special decode", b.decode(ids + [32], skip_special_tokens=True)
          == "hello STOP")
    check("keep_special decode", "<|eos|>" in b.decode(ids + [32],
                                                       skip_special_tokens=False))
    mb = [i for i, t in enumerate(b.tokens) if len(t) == 1 and ord(t) == 0xE3][0]
    sd = StreamDecoder(b)
    got = "".join(sd.push(c) for c in (mb, mb + 1, mb + 2)) + sd.flush()
    check("incremental UTF-8 (no U+FFFD)", got == "あ" and "\ufffd" not in got,
          repr(got))
    sd2 = StreamDecoder(b)
    check("StreamDecoder skips specials",
          sd2.push(32) == "" and sd2.push(ord_and_id(b, "a")) == "a")
    check("StreamDecoder OOR guard", StreamDecoder(b).push(99999) == "")


def ord_and_id(b, ch):
    from bpe_gguf import bytes_to_unicode
    want = bytes_to_unicode()[ord(ch)]
    return b.tokens.index(want)


# --------------------------------------------------------------------------
# 2. Direct m64run --worker protocol tests
# --------------------------------------------------------------------------
class PWorker:
    """Tiny harness around the raw line protocol."""

    def __init__(self, cmd):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)

    def line(self):
        l = self.proc.stdout.readline()
        assert l, "worker closed stdout"
        return l.strip()

    def gen(self, spec):
        self.proc.stdin.write(spec + "\n")
        self.proc.stdin.flush()
        toks = []
        while True:
            l = self.line()
            if l.startswith("TOK "):
                toks.append(int(l.split()[1]))
            elif l.startswith("DONE"):
                return toks, l
            elif l.startswith("ERR"):
                return None, l

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
        except OSError:
            pass
        return self.proc.wait(timeout=10)


def test_worker(fix):
    print("[2] m64run --worker CLI & protocol")
    exe = os.path.join(ROOT, "build", "m64run")
    check("build/m64run exists", os.path.exists(exe), "run `make build` first")
    if os.path.exists(exe):
        # Verify CLI accepts --worker and --ctx (tiny.m64pack has NE=32 while
        # m64run enforces Inkling-2-Mini hparams NE=2304, so init returns rc=5,
        # NOT rc=2 usage/flag error).
        pack = os.path.join(fix, "tiny.m64pack")
        res = subprocess.run([exe, pack, "--worker", "--ctx", "128"],
                             stdin=subprocess.DEVNULL, capture_output=True, text=True)
        check("m64run accepts --worker --ctx",
              res.returncode == 5 and "unexpected hparams" in res.stderr,
              (res.returncode, res.stderr.strip()))
        bad = subprocess.run([exe, pack, "--no-such-flag"],
                             stdin=subprocess.DEVNULL, capture_output=True, text=True)
        check("m64run rejects unknown flag (rc=2)", bad.returncode == 2, bad.returncode)

    # Always test the full line protocol against MOCK_WORKER (no 26 GiB pack needed).
    tmp = tempfile.mkdtemp(prefix="takuan_worker_")
    mock_py = os.path.join(tmp, "mock_worker.py")
    with open(mock_py, "w") as f:
        f.write(MOCK_WORKER)
    w = PWorker([sys.executable, mock_py])
    check("READY line (mock worker)", w.line().startswith("READY vocab=4 ctx=64"),
          "first line")
    toks1, done1 = w.gen("GEN 6 0 1.0 42 -1 -- 1 2 3")
    toks2, done2 = w.gen("GEN 6 0 1.0 42 -1 -- 1 2 3")
    check("GEN emits tokens", toks1 == [29, 30, 31, 1, 2, 2], toks1)
    check("deterministic (temp=0)", toks1 == toks2, (toks1, toks2))
    check("DONE length finish", done1.split()[3] == "length", done1)
    try:
        float(done1.split()[2])
        ok_ms = True
    except ValueError:
        ok_ms = False
    check("DONE elapsed_ms parses", ok_ms, done1)
    toks3, done3 = w.gen("GEN 8 0 1.0 7 28 -- 1 2 3")
    check("EOS stop finish", done3.split()[3] == "stop" and toks3 == [29, 30, 31, 1, 2, 2],
          (toks3, done3))
    _, err = w.gen("GEN 8 0 1.0 42 -1 -- " + " ".join(["1"] * 64))
    check("prompt_exceeds_ctx", err == "ERR prompt_exceeds_ctx", err)
    toks4, done4 = w.gen("GEN 4 0 1.0 42 -1 -- 1 2 3")
    check("worker alive after error", toks4 is not None and len(toks4) == 4,
          (toks4, done4))
    _, err2 = w.gen("nope")
    check("bad_command", err2 == "ERR bad_command", err2)
    check("QUIT exits 0", w.close() == 0)

    real_pack = os.environ.get("TAKUAN_REAL_PACK")
    if real_pack and os.path.exists(real_pack) and os.path.exists(exe):
        rw = PWorker([exe, real_pack, "--worker", "--backend", "metal",
                      "--ctx", "128", "--arena-gb", "4", "--io", "pread"])
        check("real pack READY ctx=128", rw.line().endswith("ctx=128"))
        rtoks, rdone = rw.gen("GEN 4 0 1.0 42 -1 -- 1 2 3")
        check("real pack GEN emits 4 tokens", rtoks is not None and len(rtoks) == 4,
              (rtoks, rdone))
        _, rerr = rw.gen("GEN 4 0 1.0 42 -1 -- 9999999")
        check("real pack bad_token_id", rerr == "ERR bad_token_id", rerr)
        check("real pack QUIT exits 0", rw.close() == 0)


# --------------------------------------------------------------------------
# 3. HTTP / SSE integration tests via mock worker
# --------------------------------------------------------------------------
MOCK_WORKER = r'''
import sys
vocab = ["a", "b", "c", "<|eos|>"]
sys.stdout.write("READY vocab=4 ctx=64\n"); sys.stdout.flush()
while True:
    line = sys.stdin.readline()
    if not line or line.strip() == "QUIT":
        break
    parts = line.split()
    if parts[0] != "GEN":
        sys.stdout.write("ERR bad_command\n"); sys.stdout.flush(); continue
    n = int(parts[1])
    eos = parts[5]
    ids = parts[7:]
    if len(ids) >= 64:
        sys.stdout.write("ERR prompt_exceeds_ctx\n"); sys.stdout.flush(); continue
    emitted = 0
    for t in [29, 30, 31, 1, 2, 2, 99, 28]:  # 'あ' split + 'bcc' + OOR + EOS
        if t == 28 and str(t) in eos.split(","):
            sys.stdout.write("DONE %d 12.5 stop\n" % emitted); sys.stdout.flush()
            break
        if t == 99:
            continue
        if emitted >= n:
            sys.stdout.write("DONE %d 12.5 length\n" % emitted); sys.stdout.flush()
            break
        sys.stdout.write("TOK %d\n" % t); sys.stdout.flush()
        emitted += 1
    else:
        sys.stdout.write("DONE %d 12.5 length\n" % emitted); sys.stdout.flush()
'''

STUB_BPE_SRC = '''
import codecs
class BPE:
    tokens = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
              "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
              " ", "OK", "<|eos|>", chr(0xE3), chr(0x81), chr(0x82)]
    id_of = {t: i for i, t in enumerate(tokens)}
    special_tokens = {"<|eos|>": 28}
    special_ids = {28}
    eos_token_id = 28
    eot_token_id = None
    def __init__(self, gguf_path=None):
        pass
    def encode(self, text, allow_special=True):
        ids = []
        for ch in text:
            if ch in self.id_of:
                ids.append(self.id_of[ch])
            elif ch.lower() in self.id_of:
                ids.append(self.id_of[ch.lower()])
            else:
                ids.append(self.id_of[" "])
        return ids
    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.tokens[i] for i in ids
                       if 0 <= i < len(self.tokens)
                       and not (skip_special_tokens and i in self.special_ids))

class StreamDecoder:
    def __init__(self, bpe, skip_special_tokens=True):
        self.bpe = bpe
        self.skip = skip_special_tokens
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")
    def push(self, tid):
        if tid < 0 or tid >= len(self.bpe.tokens):
            return ""
        if self.skip and tid in self.bpe.special_ids:
            return ""
        return self._dec.decode(self.bpe.tokens[tid].encode("latin-1"),
                                final=False)
    def flush(self):
        return self._dec.decode(b"", final=True)
'''


class Server:
    def __init__(self, args):
        env = dict(os.environ, PYTHONPATH=HERE)
        self.proc = subprocess.Popen(
            [sys.executable, os.path.join(HERE, "takuan_server.py")] + args,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, text=True)
        line = self.proc.stderr.readline()
        if "serving" not in line:
            raise RuntimeError("server did not start: " + line)
        self.port = int(line.rsplit(":", 1)[1].split("/")[0])
        self.base = "http://127.0.0.1:%d" % self.port

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def http(method, url, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read().decode("utf-8")
            ct = r.headers.get("Content-Type", "")
            return r.status, raw, ct
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8"), e.headers.get("Content-Type", "")


def sse_events(raw):
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            events.append(block[6:])
    return events


def test_http(fix):
    print("[3] HTTP / SSE (mock worker + stub tokenizer)")
    tmp = tempfile.mkdtemp(prefix="takuan_http_")
    mock_py = os.path.join(tmp, "mock_worker.py")
    with open(mock_py, "w") as f:
        f.write(MOCK_WORKER)
    stub_dir = os.path.join(tmp, "stubpkg")
    os.makedirs(stub_dir, exist_ok=True)
    with open(os.path.join(stub_dir, "bpe_gguf.py"), "w") as f:
        f.write(STUB_BPE_SRC)
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    env = dict(os.environ, PYTHONPATH=HERE)
    # Pre-import the stub bpe_gguf module before takuan_server inserts tools/
    # at sys.path[0], so takuan_server runs unmodified against STUB_BPE_SRC.
    boot = ("import sys; sys.path.insert(0, %r); import bpe_gguf; "
            "import takuan_server; takuan_server.main()" % stub_dir)
    proc = subprocess.Popen(
        [sys.executable, "-c", boot,
         "--worker-cmd", sys.executable + " " + mock_py,
         "--gguf", os.path.join(fix, "tiny.gguf"),
         "--port", str(port), "--ctx", "64"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, text=True)
    ready_line = ""
    try:
        ready_line = proc.stderr.readline()
    except Exception:
        pass
    ok_start = "serving" in ready_line
    check("server starts (mock worker)", ok_start, ready_line.strip())
    base = "http://127.0.0.1:%d" % port
    try:
        # -- health / models --
        st, raw, _ = http("GET", base + "/health")
        j = json.loads(raw)
        check("GET /health 200 ok", st == 200 and j.get("status") == "ok", raw)
        check("health reports model+ctx",
              j.get("model") == "takuan-inkling-2-mini" and j.get("ctx") == 64, j)
        st, raw, _ = http("GET", base + "/v1/models")
        j = json.loads(raw)
        check("GET /v1/models lists model",
              st == 200 and j["data"][0]["id"] == "takuan-inkling-2-mini"
              and j["data"][0]["object"] == "model", raw)

        # -- chat non-stream --
        st, raw, _ = http("POST", base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10})
        j = json.loads(raw)
        check("chat 200", st == 200, raw)
        check("chat object", j.get("object") == "chat.completion", raw)
        msg = j["choices"][0]["message"]
        check("chat role assistant", msg["role"] == "assistant", msg)
        # mock stream: あ (30,31,32) + b c c then EOS(3) with stop finish
        check("chat content UTF-8 intact", msg["content"] == "あbcc",
              repr(msg["content"]))
        check("chat finish stop (EOS)", j["choices"][0]["finish_reason"] == "stop",
              j["choices"][0])
        u = j["usage"]
        check("chat usage fields",
              u["completion_tokens"] == 6 and
              u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"], u)

        # -- chat stream --
        st, raw, ct = http("POST", base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10, "stream": True,
            "stream_options": {"include_usage": True}})
        check("stream 200 SSE", st == 200 and "text/event-stream" in ct, ct)
        evs = sse_events(raw)
        check("stream ends [DONE]", evs and evs[-1] == "[DONE]", evs[-3:])
        chunks = [json.loads(e) for e in evs if e != "[DONE]"]
        check("chunk object type",
              all(c["object"] == "chat.completion.chunk" for c in chunks), chunks)
        first = chunks[0]["choices"][0]["delta"]
        check("initial role delta", first.get("role") == "assistant", first)
        text = "".join(c["choices"][0]["delta"].get("content", "")
                       for c in chunks if c["choices"])
        check("streamed text UTF-8 intact", text == "あbcc", repr(text))
        last = [c for c in chunks if c["choices"]][-1]["choices"][0]
        check("final chunk finish_reason", last["finish_reason"] == "stop", last)
        usage_ev = [c for c in chunks if not c["choices"]]
        check("include_usage chunk", usage_ev and
              usage_ev[0]["usage"]["completion_tokens"] == 6, usage_ev)

        # -- stop string trimming --
        st, raw, _ = http("POST", base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10, "stop": "cc"})
        j = json.loads(raw)
        check("stop string trims content",
              j["choices"][0]["message"]["content"] == "あb",
              repr(j["choices"][0]["message"]["content"]))
        check("stop string finish",
              j["choices"][0]["finish_reason"] == "stop",
              j["choices"][0]["finish_reason"])

        # -- completions endpoint --
        st, raw, _ = http("POST", base + "/v1/completions", {
            "prompt": "hi", "max_tokens": 10})
        j = json.loads(raw)
        check("completions 200 json",
              st == 200 and j.get("object") == "text_completion"
              and j["choices"][0]["text"] == "あbcc"
              and j["choices"][0]["finish_reason"] == "stop", raw)
        st, raw, ct = http("POST", base + "/v1/completions", {
            "prompt": "hi", "max_tokens": 10, "stream": True})
        evs = sse_events(raw)
        check("completions SSE", st == 200 and evs[-1] == "[DONE]", raw[:120])
        chunks = [json.loads(e) for e in evs if e != "[DONE]"]
        text = "".join(c["choices"][0]["text"] for c in chunks if c["choices"])
        check("completions streamed text", text == "あbcc", repr(text))

        # -- errors --
        st, raw, _ = http("POST", base + "/v1/chat/completions",
                          {"messages": "notalist"})
        j = json.loads(raw)
        check("400 invalid messages", st == 400 and "error" in j, raw)
        long_prompt = "a" * 100  # stub bpe: 1 char = 1 token >= ctx 64
        st, raw, _ = http("POST", base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": long_prompt}]})
        j = json.loads(raw)
        check("400 prompt_exceeds_ctx",
              st == 400 and j["error"].get("code") == "prompt_exceeds_ctx", raw)
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     data=b"{not json", method="POST")
        try:
            urllib.request.urlopen(req, timeout=10)
            check("400 invalid JSON", False, "no error raised")
        except urllib.error.HTTPError as e:
            check("400 invalid JSON", e.code == 400, e.code)
        st, raw, _ = http("GET", base + "/nope")
        check("404 unknown path", st == 404, st)

        # -- OPTIONS CORS --
        req = urllib.request.Request(base + "/v1/chat/completions",
                                     method="OPTIONS")
        with urllib.request.urlopen(req, timeout=10) as r:
            check("OPTIONS 204 CORS",
                  r.status == 204
                  and r.headers.get("Access-Control-Allow-Origin") == "*"
                  and "Authorization" in (r.headers.get("Access-Control-Allow-Headers") or ""),
                  dict(r.headers))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------------------------------
# 4. OpenAI Tool Calling tests (Inkling special tokens + HTTP + SSE)
# --------------------------------------------------------------------------
TOOL_STUB_BPE_SRC = '''
import codecs, re
_SPECIALS = [
    "<|message_system|>", "<|message_user|>", "<|message_model|>",
    "<|message_tool|>", "<|content_text|>", "<|content_thinking|>",
    "<|content_xml|>", "<|content_invoke_tool_json|>",
    "<|end_message|>", "<|content_model_end_sampling|>", "<|endoftext|>",
]
_TOKENS = [chr(c) for c in range(32, 127)] + ["\\n"] + _SPECIALS
_ID_OF = {t: i for i, t in enumerate(_TOKENS)}
_SPEC_MAP = {t: _ID_OF[t] for t in _SPECIALS}
class BPE:
    tokens = _TOKENS
    id_of = _ID_OF
    special_tokens = _SPEC_MAP
    special_ids = set(_SPEC_MAP.values())
    eos_token_id = _ID_OF["<|content_model_end_sampling|>"]
    eot_token_id = eos_token_id
    _re = re.compile("(" + "|".join(re.escape(t) for t in _SPECIALS) + ")")
    def __init__(self, gguf_path=None):
        pass
    def encode(self, text, allow_special=True):
        out = []
        parts = self._re.split(text) if allow_special else [text]
        for part in parts:
            if not part:
                continue
            if allow_special and part in self.special_tokens:
                out.append(self.special_tokens[part])
            else:
                for ch in part:
                    out.append(self.id_of[ch])
        return out
    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.tokens[i] for i in ids
                       if 0 <= i < len(self.tokens)
                       and not (skip_special_tokens and i in self.special_ids))
    def token_bytes(self, tid):
        return self.tokens[tid].encode("utf-8")

class StreamDecoder:
    def __init__(self, bpe, skip_special_tokens=True):
        self.bpe = bpe
        self.skip = skip_special_tokens
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")
    def push(self, tid):
        if tid < 0 or tid >= len(self.bpe.tokens):
            return ""
        if self.skip and tid in self.bpe.special_ids:
            return ""
        return self._dec.decode(self.bpe.tokens[tid].encode("utf-8"),
                                final=False)
    def flush(self):
        return self._dec.decode(b"", final=True)
'''

TOOL_MOCK_WORKER = r'''
import sys, os
sys.path.insert(0, os.environ["TOOL_STUB_DIR"])
from bpe_gguf import BPE
b = BPE()
reply = (
    '<|content_thinking|>need ls<|end_message|>'
    '<|message_model|><|content_text|>Running tool.<|end_message|>'
    '<|message_model|>run<|content_invoke_tool_json|>'
    '{"name":"run","args":{"cmd":"ls"}}<|end_message|>'
    '<|content_model_end_sampling|>'
)
seq = b.encode(reply)
sys.stdout.write("READY vocab=%d ctx=2048\n" % len(b.tokens)); sys.stdout.flush()
while True:
    line = sys.stdin.readline()
    if not line or line.strip() == "QUIT":
        break
    parts = line.split()
    if parts[0] != "GEN":
        sys.stdout.write("ERR bad_command\n"); sys.stdout.flush(); continue
    eos_set = set(int(x) for x in parts[5].split(",") if x not in ("-1", "none"))
    n = int(parts[1])
    em = 0
    for tid in seq:
        if tid in eos_set:
            sys.stdout.write("DONE %d 5.0 stop\n" % em); sys.stdout.flush()
            break
        if em >= n:
            sys.stdout.write("DONE %d 5.0 length\n" % em); sys.stdout.flush()
            break
        sys.stdout.write("TOK %d\n" % tid); sys.stdout.flush()
        em += 1
    else:
        sys.stdout.write("DONE %d 5.0 length\n" % em); sys.stdout.flush()
'''


def test_tool_calling(fix):
    print("[4] OpenAI Tool Calling (prompt + JSON + SSE)")
    import takuan_server
    ns = {}
    exec(TOOL_STUB_BPE_SRC, ns)
    tbpe = ns["BPE"]()
    tools = [{
        "type": "function",
        "function": {
            "name": "run",
            "description": "Execute command",
            "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
        },
    }]
    hist = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "List dir"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1", "type": "function",
            "function": {"name": "run", "arguments": '{"cmd": "ls"}'},
        }]},
        {"role": "tool", "tool_call_id": "call_1", "content": "README.md"},
    ]
    prompt = takuan_server.format_chat_prompt(hist, tbpe, tools=tools)
    check("tool_declare in prompt",
          "<|message_system|>tool_declare<|content_xml|>" in prompt, prompt[:120])
    check("assistant tool_call history",
          '<|message_model|>run<|content_invoke_tool_json|>{"name":"run","args":{"cmd":"ls"}}<|end_message|>' in prompt,
          prompt)
    check("tool result role formatting",
          "<|message_tool|>run<|content_text|>README.md<|end_message|>" in prompt,
          prompt)
    check("structured prompt ends <|message_model|>",
          prompt.endswith("<|message_model|>"), prompt[-40:])
    eos_struct = takuan_server.resolve_eos_ids(tbpe, structured=True)
    check("structured eos excludes <|end_message|>",
          tbpe.id_of["<|end_message|>"] not in eos_struct
          and tbpe.id_of["<|content_model_end_sampling|>"] in eos_struct,
          eos_struct)

    tmp = tempfile.mkdtemp(prefix="takuan_tools_")
    mock_py = os.path.join(tmp, "mock_tool_worker.py")
    with open(mock_py, "w") as f:
        f.write(TOOL_MOCK_WORKER)
    stub_dir = os.path.join(tmp, "stubpkg")
    os.makedirs(stub_dir, exist_ok=True)
    with open(os.path.join(stub_dir, "bpe_gguf.py"), "w") as f:
        f.write(TOOL_STUB_BPE_SRC)

    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    env = dict(os.environ, PYTHONPATH=HERE, TOOL_STUB_DIR=stub_dir)
    boot = ("import sys; sys.path.insert(0, %r); import bpe_gguf; "
            "import takuan_server; takuan_server.main()" % stub_dir)
    proc = subprocess.Popen(
        [sys.executable, "-c", boot,
         "--worker-cmd", sys.executable + " " + mock_py,
         "--gguf", os.path.join(fix, "tiny.gguf"),
         "--port", str(port), "--ctx", "2048"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, text=True)
    ready_line = proc.stderr.readline()
    check("tool server starts", "serving" in ready_line, ready_line.strip())
    base = "http://127.0.0.1:%d" % port
    try:
        # Non-streaming tool call
        st, raw, _ = http("POST", base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "List dir"}],
            "tools": tools, "max_tokens": 256,
        })
        j = json.loads(raw)
        ch = j["choices"][0]
        check("tool chat 200", st == 200, raw)
        check("tool finish_reason tool_calls", ch["finish_reason"] == "tool_calls", ch)
        check("tool text intact", ch["message"]["content"] == "Running tool.", ch["message"])
        check("tool reasoning captured", ch["message"].get("reasoning_content") == "need ls", ch["message"])
        tc = ch["message"]["tool_calls"][0]
        check("tool call parsed name+args",
              tc["type"] == "function" and tc["function"]["name"] == "run"
              and json.loads(tc["function"]["arguments"]) == {"cmd": "ls"}, tc)

        # Streaming SSE tool call
        st, raw, ct = http("POST", base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "List dir"}],
            "tools": tools, "max_tokens": 256, "stream": True,
        })
        evs = sse_events(raw)
        chunks = [json.loads(e) for e in evs if e != "[DONE]"]
        txt = "".join(c["choices"][0]["delta"].get("content", "")
                      for c in chunks if c.get("choices"))
        tcs = [t for c in chunks if c.get("choices")
               for t in c["choices"][0]["delta"].get("tool_calls", [])]
        last = [c for c in chunks if c.get("choices")][-1]["choices"][0]
        check("tool stream 200 DONE", st == 200 and evs[-1] == "[DONE]", evs[-2:])
        check("tool stream content", txt == "Running tool.", repr(txt))
        check("tool stream delta tool_calls",
              len(tcs) == 1 and tcs[0]["function"]["name"] == "run"
              and json.loads(tcs[0]["function"]["arguments"]) == {"cmd": "ls"}, tcs)
        check("tool stream finish tool_calls", last["finish_reason"] == "tool_calls", last)
        # Thinking must stream live as delta.reasoning_content (never buffered
        # until the end: a >60s thinking phase would trip stale-stream
        # watchdogs in clients such as Hermes Agent).
        rds = [c["choices"][0]["delta"]["reasoning_content"]
               for c in chunks if c.get("choices")
               and c["choices"][0]["delta"].get("reasoning_content") is not None]
        check("tool stream reasoning_content chunks emitted live",
              len(rds) >= 6, (len(rds), rds))
        check("tool stream reasoning_content joins intact",
              "".join(rds) == "need ls", repr("".join(rds)))
        order = [("r" if c["choices"][0]["delta"].get("reasoning_content") is not None
                  else "c" if c["choices"][0]["delta"].get("content")
                  else "-")
                 for c in chunks if c.get("choices")]
        check("tool stream reasoning chunks precede content chunks",
              "r" in order and "c" in order
              and max(i for i, k in enumerate(order) if k == "r")
                  < min(i for i, k in enumerate(order) if k == "c"), order)
        check("tool stream reasoning chunks carry finish_reason null",
              all(c["choices"][0]["finish_reason"] is None for c in chunks
                  if c.get("choices")
                  and c["choices"][0]["delta"].get("reasoning_content") is not None),
              [c for c in chunks if c.get("choices")
               and c["choices"][0]["delta"].get("reasoning_content") is not None][:2])
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------------------------------
# 5. Bitwise-exact prefix caching (WARM protocol + server auto-warm)
# --------------------------------------------------------------------------
PREFIX_MOCK_WORKER = r'''
import sys, os
sys.path.insert(0, os.environ["TOOL_STUB_DIR"])
from bpe_gguf import BPE
b = BPE()
seq = b.encode('<|content_text|>Hi there<|content_model_end_sampling|>')
log_path = os.environ["WARM_LOG"]
warm_calls = 0
warm_lens = []
last_warm = None
def log():
    with open(log_path, "w") as f:
        f.write("%d\n%s\n" % (warm_calls, ",".join(str(x) for x in warm_lens)))
sys.stdout.write("READY vocab=%d ctx=2048\n" % len(b.tokens)); sys.stdout.flush()
log()
while True:
    line = sys.stdin.readline()
    if not line or line.strip() == "QUIT":
        break
    parts = line.split()
    if parts[0] == "WARM":
        warm_calls += 1
        ids = [p for p in parts[1:] if p != "--"]
        warm_lens.append(len(ids))
        if ids == last_warm:
            sys.stdout.write("WARMED %d cached\n" % len(ids)); sys.stdout.flush()
            log(); continue
        last_warm = ids
        sys.stdout.write("WARMED %d computed\n" % len(ids)); sys.stdout.flush()
        log()
        continue
    if parts[0] != "GEN":
        sys.stdout.write("ERR bad_command\n"); sys.stdout.flush(); continue
    eos_set = set(int(x) for x in parts[5].split(",") if x not in ("-1", "none"))
    n = int(parts[1])
    em = 0
    for tid in seq:
        if tid in eos_set:
            sys.stdout.write("DONE %d 5.0 stop\n" % em); sys.stdout.flush()
            break
        if em >= n:
            sys.stdout.write("DONE %d 5.0 length\n" % em); sys.stdout.flush()
            break
        sys.stdout.write("TOK %d\n" % tid); sys.stdout.flush()
        em += 1
    else:
        sys.stdout.write("DONE %d 5.0 length\n" % em); sys.stdout.flush()
'''


def test_prefix_cache(fix):
    print("[5] Prefix caching (WARM + auto-warm + split_static_chat_prefix)")
    import takuan_server

    # ---- m64run CLI: --prefix-cache is a real flag (rc=5 hparams, not rc=2)
    exe = os.path.join(ROOT, "build", "m64run")
    if os.path.exists(exe):
        pack = os.path.join(fix, "tiny.m64pack")
        res = subprocess.run([exe, pack, "--worker", "--prefix-cache", "2",
                              "--cache-dir", os.path.join(fix, "kvc")],
                             stdin=subprocess.DEVNULL, capture_output=True,
                             text=True)
        check("m64run accepts --prefix-cache 2 --cache-dir",
              res.returncode == 5 and "unexpected hparams" in res.stderr,
              (res.returncode, res.stderr.strip()[:80]))
    help_res = subprocess.run(
        [sys.executable, os.path.join(HERE, "takuan_server.py"), "--help"],
        capture_output=True, text=True)
    check("takuan_server exposes --prefix-cache",
          help_res.returncode == 0 and "--prefix-cache" in help_res.stdout,
          help_res.returncode)
    check("takuan_server exposes --cache-dir",
          help_res.returncode == 0 and "--cache-dir" in help_res.stdout,
          help_res.returncode)

    # ---- legacy worker: warm() swallows ERR bad_command, worker stays usable
    tmp = tempfile.mkdtemp(prefix="takuan_prefix_")
    mock_py = os.path.join(tmp, "mock_worker.py")
    with open(mock_py, "w") as f:
        f.write(MOCK_WORKER)
    ew = takuan_server.EngineWorker(None, None, "metal", 64, 0.0, "pread",
                                    worker_cmd=sys.executable + " " + mock_py)
    check("EngineWorker.warm legacy worker -> False", ew.warm([1, 2, 3]) is False)
    got = [ev for ev in ew.generate([1, 2, 3], 4, 0.0, 1.0, 42, [])]
    check("worker usable after legacy WARM",
          sum(1 for g in got if g[0] == "tok") == 4, got)
    ew.close()

    # ---- disk-warm worker: WARMED <n> disk -> EngineWorker.warm() True
    disk_mock = os.path.join(tmp, "disk_mock.py")
    with open(disk_mock, "w") as f:
        f.write(
            'import sys\n'
            'sys.stdout.write("READY vocab=32 ctx=64\\n"); sys.stdout.flush()\n'
            'while True:\n'
            '    line = sys.stdin.readline()\n'
            '    if not line or line.strip() == "QUIT": break\n'
            '    if line.startswith("WARM "):\n'
            '        sys.stdout.write("WARMED %d disk\\n" % len(line.split()[1:]))\n'
            '        sys.stdout.flush()\n'
        )
    ewd = takuan_server.EngineWorker(None, None, "metal", 64, 0.0, "pread",
                                     worker_cmd=sys.executable + " " + disk_mock)
    check("EngineWorker.warm disk hit -> True", ewd.warm([1, 2, 3]) is True)
    ewd.close()

    # ---- split_static_chat_prefix unit checks (Inkling stub tokenizer)
    ns = {}
    exec(TOOL_STUB_BPE_SRC, ns)
    tbpe = ns["BPE"]()
    tools = [{
        "type": "function",
        "function": {
            "name": "run",
            "description": "Execute command",
            "parameters": {"type": "object",
                           "properties": {"cmd": {"type": "string"}}},
        },
    }]
    hist = [
        {"role": "system", "content": "B" * 120},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Yo"},
        {"role": "user", "content": "again"},
    ]
    prefix_text, prefix_ids = takuan_server.split_static_chat_prefix(
        hist, tbpe, tools=tools)
    full = takuan_server.format_chat_prompt(hist, tbpe, tools=tools)
    check("static prefix is exact head of prompt",
          bool(prefix_text) and full.startswith(prefix_text), prefix_text[:80])
    check("prefix includes tool_declare", "<|message_system|>tool_declare" in prefix_text)
    check("prefix ends with effort line",
          "Thinking effort level" in prefix_text
          and prefix_text.endswith("<|end_message|>"), prefix_text[-60:])
    check("prefix excludes first user turn", "<|message_user|>" not in prefix_text)
    check("prefix ids roundtrip", tbpe.encode(prefix_text) == prefix_ids,
          len(prefix_ids))
    empty_ns = {}
    exec(STUB_BPE_SRC, empty_ns)
    pb, pi = takuan_server.split_static_chat_prefix(hist, empty_ns["BPE"]())
    check("no-inkling bpe yields empty prefix", pb == "" and pi == [], repr(pb))

    # ---- HTTP: WARM-supporting worker warmed ONCE across 2 turns sharing
    #      the same static system + tools prefix (warm_count == 1)
    mock_py = os.path.join(tmp, "prefix_worker.py")
    with open(mock_py, "w") as f:
        f.write(PREFIX_MOCK_WORKER)
    stub_dir = os.path.join(tmp, "stubpkg")
    os.makedirs(stub_dir, exist_ok=True)
    with open(os.path.join(stub_dir, "bpe_gguf.py"), "w") as f:
        f.write(TOOL_STUB_BPE_SRC)
    warm_log = os.path.join(tmp, "warm_log.txt")
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    env = dict(os.environ, PYTHONPATH=HERE, TOOL_STUB_DIR=stub_dir,
               WARM_LOG=warm_log)
    boot = ("import sys; sys.path.insert(0, %r); import bpe_gguf; "
            "import takuan_server; takuan_server.main()" % stub_dir)
    proc = subprocess.Popen(
        [sys.executable, "-c", boot,
         "--worker-cmd", sys.executable + " " + mock_py,
         "--gguf", os.path.join(fix, "tiny.gguf"),
         "--port", str(port), "--ctx", "2048"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, text=True)
    ready_line = proc.stderr.readline()
    check("prefix server starts", "serving" in ready_line, ready_line.strip())
    base = "http://127.0.0.1:%d" % port
    try:
        turn1 = [{"role": "system", "content": "B" * 120},
                 {"role": "user", "content": "hi"}]
        st, raw, _ = http("POST", base + "/v1/chat/completions",
                          {"messages": turn1, "tools": tools, "max_tokens": 64})
        j = json.loads(raw)
        check("prefix turn1 200", st == 200 and j["choices"][0]["message"]["content"]
              == "Hi there", raw[:120])
        turn2 = turn1 + [{"role": "assistant", "content": "Hi there"},
                         {"role": "user", "content": "again"}]
        st, raw, _ = http("POST", base + "/v1/chat/completions",
                          {"messages": turn2, "tools": tools, "max_tokens": 64})
        check("prefix turn2 200", st == 200, raw[:120])
        def read_warm_log():
            time.sleep(0.15)
            with open(warm_log) as f:
                lines = [ln.strip() for ln in f.readlines()]
            cnt = int(lines[0])
            lens = [int(x) for x in lines[1].split(",") if x] if len(lines) > 1 else []
            return cnt, lens

        warm_calls, warm_lens = read_warm_log()
        check("static prefix warmed once across 2 chat turns",
              warm_calls == 1 and len(warm_lens) == 1 and warm_lens[0] < 512,
              (warm_calls, warm_lens))

        # Long static prefix (>=1024 tok) must not be clobbered by a tiny
        # background prompt (<512 tok), and re-syncs when resuming long turns.
        long_turn1 = [{"role": "system", "content": "B" * 1100},
                      {"role": "user", "content": "hi"}]
        st, raw, _ = http("POST", base + "/v1/chat/completions",
                          {"messages": long_turn1, "tools": tools, "max_tokens": 64})
        warm_calls, warm_lens = read_warm_log()
        check("long prefix warmed (>=1024 tok)",
              st == 200 and warm_calls == 2 and warm_lens[-1] >= 1024,
              (st, warm_calls, warm_lens))

        tiny_bg = [{"role": "system", "content": "B" * 120},
                   {"role": "user", "content": "title?"}]
        st, raw, _ = http("POST", base + "/v1/chat/completions",
                          {"messages": tiny_bg, "max_tokens": 64})
        warm_calls, warm_lens = read_warm_log()
        check("tiny background prompt skips WARM overwrite",
              st == 200 and warm_calls == 2, (st, warm_calls, warm_lens))

        long_turn2 = long_turn1 + [{"role": "assistant", "content": "Hi there"},
                                   {"role": "user", "content": "again"}]
        st, raw, _ = http("POST", base + "/v1/chat/completions",
                          {"messages": long_turn2, "tools": tools, "max_tokens": 64})
        warm_calls, warm_lens = read_warm_log()
        check("long prefix re-syncs after intervening tiny prompt",
              st == 200 and warm_calls == 3 and warm_lens[-1] >= 1024,
              (st, warm_calls, warm_lens))

        long_turn3 = long_turn2 + [{"role": "assistant", "content": "Hi there"},
                                   {"role": "user", "content": "third"}]
        st, raw, _ = http("POST", base + "/v1/chat/completions",
                          {"messages": long_turn3, "tools": tools, "max_tokens": 64})
        warm_calls, warm_lens = read_warm_log()
        check("consecutive long turn avoids redundant WARM",
              st == 200 and warm_calls == 3, (st, warm_calls, warm_lens))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_sse_keepalive(fix):
    print("[6] SSE keepalive heartbeat during slow prefill")
    tmp = tempfile.mkdtemp(prefix="takuan_ka_")
    slow_worker = os.path.join(tmp, "slow_worker.py")
    with open(slow_worker, "w") as f:
        f.write(
            'import sys, time\n'
            'sys.stdout.write("READY vocab=33 ctx=2048\\n"); sys.stdout.flush()\n'
            'while True:\n'
            '    line = sys.stdin.readline()\n'
            '    if not line or line.strip() == "QUIT": break\n'
            '    line = line.strip()\n'
            '    if line.startswith("WARM "):\n'
            '        time.sleep(0.15)\n'
            '        sys.stdout.write("WARMED %d computed\\n" % len(line.split()[1:]))\n'
            '        sys.stdout.flush()\n'
            '    elif line.startswith("GEN "):\n'
            '        time.sleep(0.15)\n'
            '        sys.stdout.write("TOK 10\\n"); sys.stdout.flush()\n'
            '        time.sleep(0.45)\n'
            '        for tok in [11, 32]:\n'
            '            sys.stdout.write("TOK %d\\n" % tok)\n'
            '        sys.stdout.write("DONE 2 300.0 stop\\n"); sys.stdout.flush()\n'
        )
    stub_dir = os.path.join(tmp, "stubpkg")
    os.makedirs(stub_dir, exist_ok=True)
    with open(os.path.join(stub_dir, "bpe_gguf.py"), "w") as f:
        f.write(TOOL_STUB_BPE_SRC)
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    env = dict(os.environ, PYTHONPATH=HERE, TOOL_STUB_DIR=stub_dir,
               TAKUAN_SSE_KEEPALIVE_SEC="0.04")
    boot = ("import sys; sys.path.insert(0, %r); import bpe_gguf; "
            "import takuan_server; takuan_server.main()" % stub_dir)
    proc = subprocess.Popen(
        [sys.executable, "-c", boot,
         "--worker-cmd", sys.executable + " " + slow_worker,
         "--gguf", os.path.join(fix, "tiny.gguf"),
         "--port", str(port), "--ctx", "2048"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, text=True)
    ready_line = proc.stderr.readline()
    check("keepalive server starts", "serving" in ready_line, ready_line.strip())
    base = "http://127.0.0.1:%d" % port
    try:
        st, raw, _ = http("POST", base + "/v1/chat/completions",
                          {"messages": [{"role": "system", "content": "B" * 120},
                                        {"role": "user", "content": "hi"}],
                           "max_tokens": 16, "stream": True})
        check("keepalive stream 200", st == 200, (st, raw[:160]))
        ka_count = raw.count(": keepalive\n\n")
        check("keepalive comments emitted >= 4", ka_count >= 4,
              (ka_count, raw[:200]))
        first_tok_idx = raw.find('"*"')
        check("keepalive continues after first text chunk",
              first_tok_idx != -1
              and raw.count(": keepalive\n\n", first_tok_idx) >= 2,
              (first_tok_idx, raw[first_tok_idx:] if first_tok_idx != -1 else raw[:200]))
        data_lines = [ln[6:] for ln in raw.splitlines()
                      if ln.startswith("data: ")]
        check("keepalive stream ends [DONE]",
              bool(data_lines) and data_lines[-1] == "[DONE]", data_lines[-2:])
        objs = [json.loads(x) for x in data_lines[:-1]]
        got_text = "".join(
            o["choices"][0]["delta"].get("content", "")
            for o in objs if o.get("choices"))
        check("keepalive stream payload intact", got_text == "*+@", repr(got_text))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


# --------------------------------------------------------------------------
# 7. Thinking tokens stream live as delta.reasoning_content (_make_stream_fn)
# --------------------------------------------------------------------------
def test_reasoning_stream(fix):
    print("[7] Thinking tokens stream live (unit: _make_stream_fn)")
    import takuan_server
    ns = {}
    exec(TOOL_STUB_BPE_SRC, ns)
    bpe = ns["BPE"]()

    reply = ('<|content_thinking|>need ls<|end_message|>'
             '<|message_model|><|content_text|>Running tool.<|end_message|>'
             '<|content_model_end_sampling|>')

    class FakeWorker:
        def __init__(self, seq):
            self.seq = seq

        def generate(self, prompt_ids, max_tokens, temperature, top_p,
                     seed, eos_ids):
            for tid in self.seq:
                yield ("tok", tid)
            yield ("done", len(self.seq), 1.0, "stop")

    events = []
    run = takuan_server._make_stream_fn(FakeWorker(bpe.encode(reply)),
                                        bpe, 2048)
    res = run([0], 512, 1.0, 1.0, 0, [],
              lambda kind, payload: events.append((kind, payload)),
              structured=True)

    kinds = [k for k, _ in events]
    rds = [p for k, p in events if k == "reasoning_delta"]
    check("thinking yields one reasoning_delta per token (7)",
          len(rds) == 7, (len(rds), rds))
    check("reasoning_delta payloads join to thinking text",
          "".join(rds) == "need ls", repr("".join(rds)))
    check("all thinking yields precede any text delta (no buffering)",
          "reasoning_delta" in kinds and "delta" in kinds
          and max(i for i, k in enumerate(kinds) if k == "reasoning_delta")
              < min(i for i, k in enumerate(kinds) if k == "delta"),
          kinds[:14])
    check("result.reasoning still accumulated for final field",
          res["reasoning"] == "need ls", res.get("reasoning"))
    check("result.text unaffected by thinking streaming",
          res["text"] == "Running tool.", repr(res["text"]))
    check("finish_reason intact after streaming thinking",
          res["finish"] == "stop", res["finish"])

    # Plain (non-structured) mode must keep the old no-reasoning behaviour.
    events2 = []
    res2 = takuan_server._make_stream_fn(FakeWorker(bpe.encode(reply)),
                                         bpe, 2048)(
        [0], 512, 1.0, 1.0, 0, [],
        lambda kind, payload: events2.append((kind, payload)),
        structured=False)
    check("plain mode emits no reasoning_delta",
          not any(k == "reasoning_delta" for k, _ in events2),
          [k for k, _ in events2][:8])
    check("plain mode reasoning stays empty", res2["reasoning"] == "",
          repr(res2["reasoning"]))


# --------------------------------------------------------------------------
# 8. Client disconnect mid-SSE: the server must cancel the in-flight GEN and
#    drain stdout to DONE *before* releasing the worker lock, so the very
#    next request returns ONLY its own tokens (no stale stream, no deadlock).
# --------------------------------------------------------------------------
DISCONNECT_MOCK_WORKER = r'''
import sys, time
sys.stdout.write("READY vocab=32 ctx=2048\n"); sys.stdout.flush()
call = 0
while True:
    line = sys.stdin.readline()
    if not line or line.strip() == "QUIT":
        break
    parts = line.split()
    if parts[0] != "GEN":
        sys.stdout.write("ERR bad_command\n"); sys.stdout.flush(); continue
    call += 1
    n = int(parts[1])
    if call == 1:
        seq = [16] * 150        # 'q' x150, one every 5 ms (aborted request)
    else:
        seq = [25] * 8 + [28]   # 'z' x8 + EOS(28) -> DONE stop
    emitted = 0
    for t in seq:
        if emitted >= n:
            sys.stdout.write("DONE %d 10.0 length\n" % emitted)
            sys.stdout.flush(); break
        sys.stdout.write("TOK %d\n" % t); sys.stdout.flush()
        emitted += 1
        time.sleep(0.005)
    else:
        sys.stdout.write("DONE %d 10.0 stop\n" % emitted); sys.stdout.flush()
'''


def _sse_abort(base, body, want_chunks):
    """POST a streaming request over a raw socket, wait for `want_chunks`
    SSE data events, then close the socket mid-stream (abrupt client
    disconnect). Returns how many data events were seen before closing."""
    host, port = base.rsplit("://", 1)[1].rsplit(":", 1)
    payload = json.dumps(body).encode("utf-8")
    head = ("POST /v1/chat/completions HTTP/1.1\r\n"
            "Host: %s\r\nContent-Type: application/json\r\n"
            "Content-Length: %d\r\nConnection: close\r\n\r\n"
            % (host, len(payload))).encode("ascii")
    s = socket.create_connection((host, int(port)), timeout=20)
    try:
        s.sendall(head + payload)
        buf = b""
        while buf.count(b"data: ") < want_chunks:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
        got = buf.count(b"data: ")
    finally:
        s.close()  # client aborts mid-stream
    return got


def test_disconnect_drain(fix):
    print("[8] Client disconnect mid-stream: cancel + drain, next request clean")
    tmp = tempfile.mkdtemp(prefix="takuan_disc_")
    mock_py = os.path.join(tmp, "disconnect_worker.py")
    with open(mock_py, "w") as f:
        f.write(DISCONNECT_MOCK_WORKER)
    stub_dir = os.path.join(tmp, "stubpkg")
    os.makedirs(stub_dir, exist_ok=True)
    with open(os.path.join(stub_dir, "bpe_gguf.py"), "w") as f:
        f.write(STUB_BPE_SRC)
    free = socket.socket()
    free.bind(("127.0.0.1", 0))
    port = free.getsockname()[1]
    free.close()
    env = dict(os.environ, PYTHONPATH=HERE)
    boot = ("import sys; sys.path.insert(0, %r); import bpe_gguf; "
            "import takuan_server; takuan_server.main()" % stub_dir)
    proc = subprocess.Popen(
        [sys.executable, "-c", boot,
         "--worker-cmd", sys.executable + " " + mock_py,
         "--gguf", os.path.join(fix, "tiny.gguf"),
         "--port", str(port), "--ctx", "2048"],
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env, text=True)
    ready_line = proc.stderr.readline()
    check("disconnect server starts", "serving" in ready_line, ready_line.strip())
    base = "http://127.0.0.1:%d" % port
    try:
        got = _sse_abort(base, {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 200, "stream": True}, want_chunks=3)
        check("aborted stream delivered initial SSE chunks", got >= 3, got)

        # The mock worker is STILL emitting 'q' tokens into the pipe. The
        # server must cancel it and drain up to DONE before serving again;
        # a deadlock here would blow the http() 30 s timeout.
        j = {}
        try:
            st, raw, _ = http("POST", base + "/v1/chat/completions", {
                "messages": [{"role": "user", "content": "again"}],
                "max_tokens": 20})
            j = json.loads(raw)
            content = j["choices"][0]["message"]["content"]
        except Exception as e:
            st, content, raw = 0, "", "second request failed: %r" % e
        check("second request after abort returns 200", st == 200, raw[:200])
        check("no stale 'q' tokens leak from aborted request",
              "q" not in content, repr(content))
        check("second request content is its own stream only",
              st == 200 and content == "zzzzzzzz"
              and j["choices"][0]["finish_reason"] == "stop",
              (repr(content), st == 200 and j["choices"][0].get("finish_reason")))

        # Streaming + stop string right after the abort cycle must also be
        # clean (drain left the worker exactly at its command prompt).
        st, raw, ct = http("POST", base + "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "again"}],
            "max_tokens": 20, "stream": True, "stop": ["zz"]})
        evs = sse_events(raw)
        chunks = [json.loads(e) for e in evs if e != "[DONE]"]
        text = "".join(c["choices"][0]["delta"].get("content", "")
                       for c in chunks if c["choices"])
        finals = [c for c in chunks if c["choices"]]
        last = finals[-1]["choices"][0] if finals else {}
        check("post-abort streaming+stop clean",
              st == 200 and evs and evs[-1] == "[DONE]"
              and "q" not in text and text == "" and last.get("finish_reason") == "stop",
              (st, repr(text), last))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    fix = _fixture_dir()
    t0 = time.time()
    test_bpe(fix)
    test_worker(fix)
    test_http(fix)
    test_tool_calling(fix)
    test_prefix_cache(fix)
    test_sse_keepalive(fix)
    test_reasoning_stream(fix)
    test_disconnect_drain(fix)
    print("----")
    print("%d passed, %d failed (%.1fs)" % (PASSED, len(FAILED), time.time() - t0))
    if FAILED:
        print("failed: " + ", ".join(FAILED))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
