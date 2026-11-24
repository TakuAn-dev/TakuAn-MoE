#!/usr/bin/env python3
"""GPT-2 byte-level BPE tokenizer extracted straight from a GGUF's
tokenizer metadata. Self-contained (stdlib only) so the engine toolchain has
no HF dependency.

Pre-tokenizer: simplified word/digit/space/punct split covering plain ASCII
text (sufficient for parity prompts; not a full GPT-2 \p{L} port).

Usage:
  bpe_gguf.py tokenize   GGUF "text"     -> space-separated ids on stdout
  bpe_gguf.py detokenize GGUF id id ..   -> text on stdout
  bpe_gguf.py ids        GGUF "text" OUT -> binary u32 little-endian ids file
"""
import codecs
import re
import struct
import sys


def _gguf_meta(path):
    f = open(path, "rb")

    def rd(n):
        b = f.read(n)
        assert len(b) == n, "eof"
        return b

    def u32():
        return struct.unpack("<I", rd(4))[0]

    def u64():
        return struct.unpack("<Q", rd(8))[0]

    def s():
        return rd(u64()).decode("utf8")

    assert rd(4) == b"GGUF"
    u32()
    nt = u64()
    nkv = u64()
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
           10: "Q", 11: "q", 12: "d"}

    def val():
        t = u32()
        if t in sizes:
            return struct.unpack("<" + fmt[t], rd(sizes[t]))[0]
        if t == 8:
            return s()
        if t == 9:
            at, n = u32(), u64()
            if at == 8:
                return [s() for _ in range(n)]
            return [struct.unpack("<" + fmt[at], rd(sizes[at]))[0] for _ in range(n)]
        raise ValueError(f"gguf type {t}")

    meta = {}
    for _k_i in range(nkv):
        _k = s()
        try:
            meta[_k] = val()
        except Exception as e:
            raise RuntimeError(f"kv#{_k_i} key={_k!r} pos={f.tell():#x}: {e}") from None
    f.close()
    return meta


def bytes_to_unicode():
    """Exact HF GPT-2 bytes_to_unicode."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    c = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            c.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, (chr(x) for x in c)))


def _pre_tokenize(text):
    """Simplified GPT-2 split: contractions, ' +word, digit runs, alpha runs,
    space runs, punct runs."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        j = i
        if c == "'":
            m = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d").match(text, i)
            j = m.end() if m else i + 1
        elif c == " ":
            j = i + 1
            while j < n and text[j] not in " \t\n\r" and not (
                    text[j].isspace()):
                j += 1
        elif c.isspace():
            while j < n and text[j].isspace():
                j += 1
        elif c.isalpha():
            while j < n and text[j].isalpha():
                j += 1
        elif c.isdigit():
            while j < n and text[j].isdigit():
                j += 1
        else:
            while j < n and not text[j].isspace() and not text[j].isalpha() \
                    and not text[j].isdigit() and text[j] != "'":
                j += 1
        out.append(text[i:j])
        i = j
    return [p for p in out if p]


class BPE:
    def __init__(self, gguf_path):
        m = _gguf_meta(gguf_path)
        self.tokens = m["tokenizer.ggml.tokens"]
        self.id_of = {t: i for i, t in enumerate(self.tokens)}
        self.rank = {tuple(x.split(" ")): i
                     for i, x in enumerate(m.get("tokenizer.ggml.merges", []))}
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        # Optional tokenizer metadata (a minimal synthetic GGUF may omit all
        # of these — every access below must degrade gracefully).
        self.token_types = m.get("tokenizer.ggml.token_type", [])
        self.eos_token_id = m.get("tokenizer.ggml.eos_token_id", None)
        self.eot_token_id = m.get("tokenizer.ggml.eot_token_id", None)
        self.chat_template = m.get("tokenizer.chat_template", "")
        # GGUF token_type == 3 is CONTROL (special): e.g. <|endoftext|>,
        # <|message_user|> ... Split encode() on these so chat-template
        # control sequences tokenize to exactly one id.
        self.special_tokens = {t: i for i, t in enumerate(self.tokens)
                               if i < len(self.token_types)
                               and self.token_types[i] == 3}
        self.special_ids = set(self.special_tokens.values())
        if self.special_tokens:
            pat = "|".join(re.escape(t) for t in sorted(
                self.special_tokens, key=len, reverse=True))
            self._special_re = re.compile("(" + pat + ")")
        else:
            self._special_re = None

    def encode(self, text, allow_special=True):
        if allow_special and self._special_re is not None:
            ids = []
            for part in self._special_re.split(text):
                if not part:
                    continue
                sid = self.special_tokens.get(part)
                if sid is not None:
                    ids.append(sid)
                else:
                    for piece in _pre_tokenize(part):
                        ids += self._encode_piece(piece)
            return ids
        ids = []
        for piece in _pre_tokenize(text):
            ids += self._encode_piece(piece)
        return ids

    def _encode_piece(self, piece):
        s = "".join(self.b2u[b] for b in piece.encode("utf8"))
        word = list(s)
        while len(word) > 1:
            best, br = -1, None
            for i in range(len(word) - 1):
                r = self.rank.get((word[i], word[i + 1]))
                if r is not None and (br is None or r < br):
                    best, br = i, r
            if best < 0:
                break
            word[best:best + 2] = [word[best] + word[best + 1]]
        out = []
        for w in word:
            i = self.id_of.get(w)
            if i is None:
                raise ValueError(f"BPE: unknown token {w!r}")
            out.append(i)
        return out

    def token_bytes(self, tid):
        """Raw bytes of token id `tid` (GPT-2 byte-space chars reversed
        through u2b; any non-byte-space chars, e.g. the literal text of a
        special token like '<|endoftext|>', are appended as UTF-8 so nothing
        is silently dropped)."""
        t = self.tokens[tid]
        out = bytearray()
        buf = bytearray()
        for c in t:
            b = self.u2b.get(c)
            if b is not None:
                buf.append(b)
            else:
                if buf:
                    out += buf
                    buf = bytearray()
                out += c.encode("utf8")
        out += buf
        return bytes(out)

    def decode(self, ids, skip_special_tokens=False):
        out = bytearray()
        for i in ids:
            if not isinstance(i, int) or i < 0 or i >= len(self.tokens):
                continue  # out-of-range guard
            if skip_special_tokens and i in self.special_ids:
                continue
            out += self.token_bytes(i)
        return out.decode("utf8", "replace")


class StreamDecoder:
    """Incremental token-id -> text decoder for streaming responses.

    Tokens may split a UTF-8 character across ids (byte-level BPE); the
    held-back bytes are carried in an incremental UTF-8 decoder so partial
    sequences never surface as U+FFFD replacement chars.
    """

    def __init__(self, bpe, skip_special_tokens=True):
        self.bpe = bpe
        self.skip_special_tokens = skip_special_tokens
        self._dec = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, token_id):
        if token_id < 0 or token_id >= len(self.bpe.tokens):
            return ""
        if self.skip_special_tokens and token_id in self.bpe.special_ids:
            return ""
        return self._dec.decode(self.bpe.token_bytes(token_id), final=False)

    def flush(self):
        return self._dec.decode(b"", final=True)



def main():
    mode, gg = sys.argv[1], sys.argv[2]
    bpe = BPE(gg)
    if mode == "tokenize":
        print(" ".join(map(str, bpe.encode(sys.argv[3]))))
    elif mode == "detokenize":
        print(bpe.decode([int(x) for x in sys.argv[3:]]), end="")
    elif mode == "ids":
        ids = bpe.encode(sys.argv[3])
        with open(sys.argv[4], "wb") as f:
            f.write(struct.pack(f"<{len(ids)}I", *ids))
        print(f"wrote {len(ids)} ids -> {sys.argv[4]}")
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
