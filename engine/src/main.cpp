// main.cpp — m64run CLI: run the resident Inkling-2-Mini forward on a .m64pack,
// optionally comparing per-step logits against a llama.cpp oracle dump, or
// serving as a persistent line-protocol worker for tools/takuan_server.py.
//
//   build/m64run <pack.m64pack> [--ids-file F | --text "..."] [--steps N]
//                [--out-prefix P] [--bench] [--compare ref_logits.bin ref_tokens.bin]
//                [--backend metal|cpu] [--arena-gb N] [--io pread|mtlio]
//                [--ctx N] [--worker] [--prefix-cache N] [--cache-dir DIR]
//
// With --compare the run is TEACHER-FORCED on the oracle tokens: for step k the
// engine's argmax is checked against ref token k, then the ref token k is fed
// for step k+1 (independent of the engine's own argmax).
//
// With --worker the process initializes the model once, prints
// "READY vocab=<n> ctx=<n>" on stdout, then serves a line protocol on stdin:
//   GEN <max_tokens> <temp> <top_p> <seed> <eos_csv> -- <id0> <id1> ...
//   WARM [-- ]<id0> <id1> ...        (pin a static prefix; replies WARMED ...)
//   QUIT
// Protocol output (stdout): "TOK <id>", "DONE <n> <ms> <finish>"
// (finish in {stop, length, cancelled}; SIGUSR1 cancels the in-flight GEN),
// "WARMED <n> {computed|cached|disk|disabled}", "ERR <reason>".
// All diagnostics stay on stderr so stdout is machine-parseable.
//
// With --prefix-cache N (N>0, Metal backend) GEN reuses bitwise-exact KV +
// ShortConv checkpoints: a warm prefix (WARM) plus up to N LRU conversation
// checkpoints. A hit restores the checkpoint (~memcpy) and prefills ONLY the
// remaining tail tokens, so a growing chat re-prefills just the new turn.
//
// With --cache-dir DIR (Metal backend) WARM additionally persists each warm
// prefix as one self-contained .m64kv file on SSD (see prefix_disk.hpp):
// ShortConv state + SWA rings + used full-attention KV, xxh64-guarded. After
// a restart the whole warmed state is restored bit-for-bit in one read
// (<0.5 s for ~15k tokens on NVMe) instead of re-prefilling.
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <signal.h>
#include <sstream>
#include <string>
#include <vector>

#include "forward.hpp"
#include "forward_metal.hpp"
#include "pack_reader.hpp"
#include "prefix_disk.hpp"

namespace {

// Worker-mode GEN cancellation (tools/takuan_server.py sends SIGUSR1 when a
// client disconnects mid-stream): the handler only sets this flag, which the
// GEN loop checks between decode steps so the worker finishes cleanly with
// "DONE <n> <ms> cancelled" instead of deadlocking on a full stdout pipe.
static volatile sig_atomic_t g_cancel_gen = 0;

void on_sigusr1(int /*sig*/) { g_cancel_gen = 1; }

struct RefBins {
    int32_t n = 0;
    std::vector<float> logits;   // n * nv
    int32_t nv = 0;
    std::vector<int32_t> tokens;  // m
};

bool load_logits(const char * path, RefBins & rb, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = std::string("cannot open ") + path; return false; }
    int32_t n, nv;
    f.read((char *)&n, 4);
    f.read((char *)&nv, 4);
    if (!f || n <= 0 || nv <= 0) { err = "bad logits header"; return false; }
    rb.n = n; rb.nv = nv;
    rb.logits.resize((size_t)n * nv);
    f.read((char *)rb.logits.data(), (long)(sizeof(float) * (size_t)n * nv));
    if (!f) { err = "logits truncated"; return false; }
    return true;
}

bool load_tokens(const char * path, RefBins & rb, std::string & err) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { err = std::string("cannot open ") + path; return false; }
    int32_t m;
    f.read((char *)&m, 4);
    if (!f || m <= 0) { err = "bad tokens header"; return false; }
    rb.tokens.resize(m);
    f.read((char *)rb.tokens.data(), (long)(sizeof(int32_t) * m));
    if (!f) { err = "tokens truncated"; return false; }
    return true;
}

double cosine(const float * a, const float * b, size_t n) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-300);
}

std::vector<int32_t> read_ids(const std::string & path) {
    std::ifstream f(path);
    std::vector<int32_t> ids;
    long v;
    while (f >> v) ids.push_back((int32_t)v);
    return ids;
}

bool write_logits(const std::string & prefix, const std::vector<std::vector<float>> & rows,
                  int64_t nv) {
    std::string path = prefix + "_logits.bin";
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t n = (int32_t)rows.size(), v = (int32_t)nv;
    f.write((const char *)&n, 4);
    f.write((const char *)&v, 4);
    for (const auto & r : rows)
        f.write((const char *)r.data(), (long)(sizeof(float) * r.size()));
    return f.good();
}

// Temperature-scaled softmax + top-p nucleus sampling over logits[0..nv).
// temp <= 0 is greedy (returns the precomputed argmax). Stable softmax:
// p_i ∝ exp((logit_i - max) / temp). When top_p < 1 the probs are sorted
// descending and truncated at the nucleus whose cumulative mass >= top_p;
// the kept mass is renormalized during a uniform [0, sum) scan.
int32_t sample_token(const float * logits, int64_t nv, float temp, float top_p,
                     int32_t argmax, std::mt19937_64 & rng) {
    if (temp <= 0.0f) return argmax;
    if (nv <= 0) return argmax;
    float mx = logits[0];
    for (int64_t i = 1; i < nv; i++) mx = std::max(mx, logits[i]);
    std::vector<double> prob((size_t)nv);
    double sum = 0.0;
    const double inv_t = 1.0 / (double)temp;
    for (int64_t i = 0; i < nv; i++) {
        double p = std::exp(((double)logits[i] - (double)mx) * inv_t);
        prob[(size_t)i] = p;
        sum += p;
    }
    if (!(sum > 0.0)) return argmax;
    std::vector<size_t> keep;
    if (top_p < 1.0f) {
        std::vector<size_t> idx((size_t)nv);
        std::iota(idx.begin(), idx.end(), (size_t)0);
        std::sort(idx.begin(), idx.end(),
                  [&](size_t a, size_t b) { return prob[a] > prob[b]; });
        double cum = 0.0;
        for (size_t i : idx) {
            keep.push_back(i);
            cum += prob[i] / sum;
            if (cum >= top_p) break;
        }
    } else {
        keep.resize((size_t)nv);
        std::iota(keep.begin(), keep.end(), (size_t)0);
    }
    double ksum = 0.0;
    for (size_t i : keep) ksum += prob[i];
    std::uniform_real_distribution<double> uni(0.0, ksum);
    const double target = uni(rng);
    double acc = 0.0;
    for (size_t i : keep) {
        acc += prob[i];
        if (acc >= target) return (int32_t)i;
    }
    return (int32_t)keep.back();
}

int usage() {
    std::fprintf(stderr,
        "usage: m64run <pack.m64pack> [--ids-file F | --text \"...\"] [--steps N]\n"
        "              [--out-prefix P] [--bench] [--compare ref_logits.bin ref_tokens.bin]\n"
        "              [--backend metal|cpu] [--arena-gb N] [--io pread|mtlio]\n"
        "              [--ctx N] [--worker] [--prefix-cache N] [--cache-dir DIR]\n"
        "  --prefix-cache N  worker LRU prefix checkpoints (0=off; -1/auto: 4 full-pack,\n"
        "                    1 when --arena-gb > 6, else 0); pinned via the WARM command\n"
        "  --cache-dir DIR   persist WARM prefixes as .m64kv files on SSD (Metal only);\n"
        "                    restores warmed state across restarts (WARMED <n> disk)\n");
    return 2;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) return usage();
    std::string pack_path = argv[1];
    std::string ids_file, text, out_prefix, backend = "metal";
    int steps = -1;
    int ctx = 64;                     // context length (clamped to [4, 131072] below,
                                      // matching InklingForwardMetal::Impl::init)
    bool bench = false;
    bool worker = false;              // persistent line-protocol worker mode
    double arena_gb = 0.0;          // 0 = full-pack (Phase 2A), >0 = bounded mlock arena
    std::string io_mode = "pread";  // pread | mtlio
    int prefix_cache = -1;            // -1 = auto, else explicit N >= 0 (clamped below)
    std::string cache_dir;            // "" = SSD prefix cache off; DIR = .m64kv store
    const char * cmp_logits = nullptr, * cmp_tokens = nullptr;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--ids-file" && i + 1 < argc) ids_file = argv[++i];
        else if (a == "--text" && i + 1 < argc) text = argv[++i];
        else if (a == "--steps" && i + 1 < argc) steps = std::atoi(argv[++i]);
        else if (a == "--backend" && i + 1 < argc) backend = argv[++i];
        else if (a == "--out-prefix" && i + 1 < argc) out_prefix = argv[++i];
        else if (a == "--bench") bench = true;
        else if (a == "--worker") worker = true;
        else if (a == "--ctx" && i + 1 < argc) ctx = std::atoi(argv[++i]);
        else if (a == "--arena-gb" && i + 1 < argc) arena_gb = std::atof(argv[++i]);
        else if (a == "--io" && i + 1 < argc) io_mode = argv[++i];
        else if (a == "--prefix-cache" && i + 1 < argc) prefix_cache = std::atoi(argv[++i]);
        else if (a == "--cache-dir" && i + 1 < argc) cache_dir = argv[++i];
        else if (a == "--compare" && i + 2 < argc) { cmp_logits = argv[++i]; cmp_tokens = argv[++i]; }
        else { std::fprintf(stderr, "unknown/complete flag: %s\n", a.c_str()); return usage(); }
    }
    if (arena_gb < 0.0) { std::fprintf(stderr, "error: --arena-gb must be >= 0\n"); return 2; }
    if (ctx < 4) ctx = 4;
    if (ctx > 131072) ctx = 131072;
    if (io_mode != "pread" && io_mode != "mtlio") {
        std::fprintf(stderr, "error: --io must be pread|mtlio\n");
        return 2;
    }
    // --prefix-cache auto defaults: full-pack keeps 4 checkpoints, a big
    // arena (>6 GiB) keeps 1, a tight arena (<=6 GiB) stays off. An explicit
    // --cache-dir forces at least 1 (WARM needs the RAM prefix path alive),
    // while an explicit --prefix-cache 0 always stays off.
    if (prefix_cache < 0) {
        prefix_cache = (arena_gb == 0.0) ? 4 : (arena_gb > 6.0 ? 1 : 0);
        if (!cache_dir.empty() && prefix_cache < 1) prefix_cache = 1;
    }
    if (prefix_cache > 8) prefix_cache = 8;   // each checkpoint is ~kv-sized; cap

    std::vector<int32_t> ids;
    if (!ids_file.empty()) {
        ids = read_ids(ids_file);
        if (ids.empty()) { std::fprintf(stderr, "no ids in %s\n", ids_file.c_str()); return 2; }
    } else if (!text.empty()) {
        std::fprintf(stderr, "error: BPE tokenizer is not compiled into m64run; use --ids-file\n");
        return 3;
    }

    std::string err;
    m64::PackReader pk;
    try { pk.open(pack_path, /*verify_hashes=*/false); }
    catch (const std::exception & e) { std::fprintf(stderr, "pack open failed: %s\n", e.what()); return 4; }
    std::fprintf(stderr, "pack: %zu tensors, %s\n", pk.n_tensors(), pk.header().arch.c_str());

    struct Backend {
        m64::InklingForward cpu;
        m64::InklingForwardMetal metal;
        bool use_metal = true;
        double arena_gb = 0.0;
        std::string io_mode = "pread";
        bool init(m64::PackReader & p, int ctx, std::string & e) {
            return use_metal ? metal.init(p, ctx, e, arena_gb, io_mode) : cpu.init(p, ctx, e);
        }
        m64::StepResult step(int32_t t) { return use_metal ? metal.step(t) : cpu.step(t); }
        m64::StepResult prefill(const std::vector<int32_t> & v) {
            return use_metal ? metal.prefill(v) : cpu.prefill(v);
        }
        void reset() { if (use_metal) metal.reset(); else cpu.reset(); }
        int64_t vocab_effective() const { return use_metal ? metal.vocab_effective() : cpu.vocab_effective(); }
        int64_t vocab_logit_rows() const { return use_metal ? metal.vocab_logit_rows() : cpu.vocab_logit_rows(); }
        size_t resident_bytes() const { return use_metal ? metal.resident_bytes() : cpu.resident_bytes(); }
        std::string arena_report() const { return use_metal ? metal.arena_report() : std::string(); }
    } fwd;
    fwd.use_metal = backend != "cpu";
    fwd.arena_gb = arena_gb;
    fwd.io_mode = io_mode;
    if (!fwd.init(pk, /*context_len=*/ctx, err)) {
        std::fprintf(stderr, "init failed: %s\n", err.c_str());
        return 5;
    }
    std::fprintf(stderr, "resident bytes: %.2f GB (incl. mmap accounting)\n",
                 fwd.resident_bytes() / 1e9);

    // ---------------- worker mode: persistent line protocol (stdin/stdout) ----
    if (worker) {
        // SIGUSR1 = cancel the in-flight GEN (see g_cancel_gen). SA_RESTART so
        // the blocking stdin getline never fails with EINTR mid-command.
        struct sigaction sa;
        sa.sa_handler = on_sigusr1;
        sigemptyset(&sa.sa_mask);
        sa.sa_flags = SA_RESTART;
        if (sigaction(SIGUSR1, &sa, nullptr) != 0) {
            std::perror("sigaction(SIGUSR1)");
            return 7;
        }
        const int64_t vocab = fwd.vocab_effective();
        std::printf("READY vocab=%lld ctx=%d\n", (long long)vocab, ctx);
        std::fflush(stdout);
        // Bitwise-exact prefix cache: one pinned warm prefix (the static
        // system + tool_declare head every turn shares) plus N LRU
        // conversation checkpoints, most-recent-used at index 0.
        const size_t prefix_cache_slots = (size_t)(prefix_cache > 0 ? prefix_cache : 0);
        bool has_warm = false;
        m64::PrefixState warm_state;
        std::vector<m64::PrefixState> lru_states;
        // Longest cached entry whose tokens are a STRICT prefix of prompt
        // (strict so at least prompt.back() flows through prefill and
        // r.logits / r.argmax are always freshly produced).
        auto find_best_prefix = [&](const std::vector<int32_t> & prompt)
                -> const m64::PrefixState * {
            const m64::PrefixState * best = nullptr;
            auto consider = [&](const m64::PrefixState & st) {
                if (st.pos <= 0 || st.pos >= (int64_t)ctx) return;
                if (st.tokens.empty() || st.tokens.size() >= prompt.size()) return;
                if (!std::equal(st.tokens.begin(), st.tokens.end(), prompt.begin())) return;
                if (!best || st.tokens.size() > best->tokens.size()) best = &st;
            };
            if (has_warm) consider(warm_state);
            for (const auto & st : lru_states) consider(st);
            return best;
        };
        // Full-attention KV slots [start_pos..ctx) are about to be overwritten
        // in-place, so drop any RAM checkpoint whose tokens exceed start_pos.
        auto invalidate_after = [&](size_t start_pos) {
            if (has_warm && warm_state.tokens.size() > start_pos) has_warm = false;
            lru_states.erase(
                std::remove_if(lru_states.begin(), lru_states.end(),
                               [&](const m64::PrefixState & st) {
                                   return st.tokens.size() > start_pos;
                               }),
                lru_states.end());
        };
        auto save_lru_checkpoint = [&](const std::vector<int32_t> & toks) {
            if (prefix_cache_slots == 0 || !fwd.use_metal) return;
            if (toks.empty() || toks.size() >= (size_t)ctx) return;
            if (toks.size() != (size_t)fwd.metal.current_pos()) return;
            m64::PrefixState st;
            if (!fwd.metal.save_prefix_state(st, toks)) return;
            size_t dup = lru_states.size();
            for (size_t i = 0; i < lru_states.size(); i++) {
                if (lru_states[i].tokens == toks) { dup = i; break; }
            }
            if (dup < lru_states.size()) lru_states[dup] = std::move(st);
            else lru_states.insert(lru_states.begin(), std::move(st));
            if (lru_states.size() > prefix_cache_slots)
                lru_states.resize(prefix_cache_slots);
        };
        std::string line;
        while (std::getline(std::cin, line)) {
            size_t b = line.find_first_not_of(" \t\r\n");
            if (b == std::string::npos) continue;  // empty / whitespace-only
            line = line.substr(b, line.find_last_not_of(" \t\r\n") - b + 1);
            if (line == "QUIT") break;
            std::istringstream iss(line);
            std::string cmd;
            iss >> cmd;
            if (cmd != "GEN" && cmd != "WARM") {
                std::printf("ERR bad_command\n");
                std::fflush(stdout);
                continue;
            }
            // ---------- WARM [-- ]<id0> <id1> ...: pin a static prefix -------
            if (cmd == "WARM") {
                if (prefix_cache_slots == 0 || !fwd.use_metal) {
                    std::printf("WARMED 0 disabled\n");
                    std::fflush(stdout);
                    continue;
                }
                std::string first;
                if (!(iss >> first)) {
                    std::printf("ERR bad_command\n");
                    std::fflush(stdout);
                    continue;
                }
                bool warm_syntax = false, warm_bad_tok = false;
                std::vector<int32_t> ids;
                auto push_tok = [&](const std::string & tok_str) {
                    try {
                        size_t idx = 0;
                        long long wv = std::stoll(tok_str, &idx, 10);
                        if (idx != tok_str.size()) { warm_syntax = true; return; }
                        if (wv < 0 || wv >= vocab) { warm_bad_tok = true; return; }
                        ids.push_back((int32_t)wv);
                    } catch (const std::exception &) {
                        warm_syntax = true;
                    }
                };
                if (first != "--") push_tok(first);
                std::string tok_str;
                while (!warm_syntax && !warm_bad_tok && (iss >> tok_str)) push_tok(tok_str);
                if (ids.empty() || ids.size() >= (size_t)ctx) warm_syntax = true;
                if (warm_syntax) {
                    std::printf("ERR bad_command\n");
                    std::fflush(stdout);
                    continue;
                }
                if (warm_bad_tok) {
                    std::printf("ERR bad_token_id\n");
                    std::fflush(stdout);
                    continue;
                }
                if (has_warm && warm_state.tokens == ids) {
                    std::printf("WARMED %zu cached\n", ids.size());
                    std::fflush(stdout);
                    continue;
                }
                // SSD persistent prefix cache: one .m64kv per (pack, tokens).
                // A hit restores pos + rs + SWA rings + used full KV bit-for-bit
                // in a single read — no prefill at all.
                if (!cache_dir.empty() && fwd.use_metal) {
                    const std::string dpath =
                        m64::prefix_cache_path(cache_dir, fwd.metal.pack_fingerprint(), ids);
                    std::error_code dec;
                    if (std::filesystem::exists(dpath, dec) && !dec) {
                        const auto t0 = std::chrono::steady_clock::now();
                        m64::PrefixState dstate;
                        std::string d_err;
                        if (fwd.metal.load_disk_prefix(dpath, dstate, d_err)) {
                            const double ms =
                                std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0).count();
                            warm_state = std::move(dstate);
                            has_warm = true;
                            // Disk load rewrites [0..ids.size()) in kvBuf, so drop
                            // any RAM conversation checkpoints from earlier prompts.
                            lru_states.clear();
                            std::fprintf(stderr,
                                         "prefix cache: loaded %zu tokens from SSD in %.1f ms (%s)\n",
                                         ids.size(), ms, dpath.c_str());
                            std::printf("WARMED %zu disk\n", ids.size());
                            std::fflush(stdout);
                            continue;
                        }
                        std::fprintf(stderr, "prefix cache: disk load skipped (%s)\n",
                                     d_err.c_str());
                    }
                }
                size_t restored = 0;
                const m64::PrefixState * hit = find_best_prefix(ids);
                if (hit && fwd.metal.restore_prefix_state(*hit)) {
                    restored = hit->tokens.size();
                } else {
                    fwd.reset();
                }
                // Strict prefix guarantees the tail has >= 1 token.
                invalidate_after(restored);
                const std::vector<int32_t> tail(ids.begin() + (long)restored, ids.end());
                fwd.prefill(tail);
                if (fwd.metal.save_prefix_state(warm_state, ids)) {
                    has_warm = true;
                    std::fprintf(stderr, "prefix cache: warmed %zu tokens (restored %zu)\n",
                                 ids.size(), restored);
                    if (!cache_dir.empty() && fwd.use_metal) {
                        const std::string dpath =
                            m64::prefix_cache_path(cache_dir, fwd.metal.pack_fingerprint(), ids);
                        std::string d_err;
                        if (fwd.metal.save_disk_prefix(dpath, ids, d_err)) {
                            std::fprintf(stderr, "prefix cache: saved %zu tokens to SSD (%s)\n",
                                         ids.size(), dpath.c_str());
                        } else {
                            std::fprintf(stderr, "prefix cache: disk save skipped (%s)\n",
                                         d_err.c_str());
                        }
                    }
                    std::printf("WARMED %zu computed\n", ids.size());
                } else {
                    std::fprintf(stderr, "prefix cache: warm save failed at pos %lld\n",
                                 (long long)fwd.metal.current_pos());
                    std::printf("WARMED 0 failed\n");
                }
                std::fflush(stdout);
                continue;
            }
            // GEN <max_tokens> <temp> <top_p> <seed> <eos_csv> -- <ids>
            g_cancel_gen = 0;
            int max_tokens = 0;
            float temp = 0.0f, top_p = 0.0f;
            unsigned long long seed = 0;
            std::string eos_csv, dash;
            if (!(iss >> max_tokens >> temp >> top_p >> seed >> eos_csv >> dash) ||
                dash != "--" || max_tokens < 1 || temp < 0.0f ||
                !(top_p > 0.0f && top_p <= 1.0f)) {
                std::printf("ERR bad_command\n");
                std::fflush(stdout);
                continue;
            }
            std::set<int32_t> eos_set;
            bool bad_eos = false;
            if (eos_csv != "-1" && eos_csv != "none") {
                std::istringstream ein(eos_csv);
                std::string item;
                while (std::getline(ein, item, ',')) {
                    try { eos_set.insert((int32_t)std::stol(item)); }
                    catch (const std::exception &) { bad_eos = true; break; }
                }
                if (eos_csv.empty()) bad_eos = true;
            }
            if (bad_eos) {
                std::printf("ERR bad_eos\n");
                std::fflush(stdout);
                continue;
            }
            std::vector<int32_t> prompt;
            bool bad_tok = false, bad_syntax = false;
            long long v = 0;
            while (iss >> v) {
                if (v < 0 || v >= vocab) { bad_tok = true; break; }
                prompt.push_back((int32_t)v);
            }
            if (!bad_tok && !iss.eof()) bad_syntax = true;
            if (bad_syntax) {
                std::printf("ERR bad_command\n");
                std::fflush(stdout);
                continue;
            }
            if (bad_tok) {
                std::printf("ERR bad_token_id\n");
                std::fflush(stdout);
                continue;
            }
            if (prompt.empty()) {
                std::printf("ERR bad_command\n");
                std::fflush(stdout);
                continue;
            }
            if (prompt.size() >= (size_t)ctx) {
                std::printf("ERR prompt_exceeds_ctx\n");
                std::fflush(stdout);
                continue;
            }
            std::mt19937_64 rng(seed != 0 ? std::mt19937_64::result_type(seed)
                                          : std::random_device{}());
            auto t0 = std::chrono::steady_clock::now();
            // ---- prefix cache reuse: restore + prefill ONLY the tail -------
            size_t prefix_hit = 0;
            if (prefix_cache_slots > 0 && fwd.use_metal) {
                const m64::PrefixState * hit = find_best_prefix(prompt);
                if (hit && fwd.metal.restore_prefix_state(*hit)) {
                    prefix_hit = hit->tokens.size();
                    std::fprintf(stderr, "prefix cache: hit %zu / %zu tokens\n",
                                 prefix_hit, prompt.size());
                    // LRU bump AFTER restore (rotating would invalidate `hit`).
                    for (size_t i = 0; i < lru_states.size(); i++) {
                        if (&lru_states[i] == hit) {
                            if (i > 0)
                                std::rotate(lru_states.begin(), lru_states.begin() + (long)i,
                                            lru_states.begin() + (long)i + 1);
                            break;
                        }
                    }
                } else {
                    fwd.reset();
                }
                invalidate_after(prefix_hit);
            } else {
                fwd.reset();
            }
            // Strict prefix guarantees prefix_hit < prompt.size(): the final
            // prompt token always runs through prefill, so r.logits/argmax
            // below are freshly produced for prompt.back().
            const std::vector<int32_t> tail(prompt.begin() + (long)prefix_hit, prompt.end());
            m64::StepResult r = fwd.prefill(tail);
            // Keep pre-decode `prompt` in the LRU so follow-up turns still hit
            // even if assistant reasoning or tool-call JSON is re-serialized.
            if (prompt.size() > prefix_hit) save_lru_checkpoint(prompt);
            int avail = ctx - (int)prompt.size();
            int limit = std::min(max_tokens, avail);
            int n_gen = 0;
            const char * finish = "length";
            // Cancelled while prefill ran: skip decode entirely, report clean DONE.
            if (g_cancel_gen) finish = "cancelled";
            // Tokens whose KV+rs state is in the engine = prompt + every
            // token actually consumed by fwd.step() (NOT the last sampled
            // token when we stop before stepping it).
            std::vector<int32_t> processed = prompt;
            for (int k = 0; k < limit; k++) {
                if (g_cancel_gen) { finish = "cancelled"; break; }
                int32_t tok = sample_token(r.logits, vocab, temp, top_p, r.argmax, rng);
                if (eos_set.count(tok)) { finish = "stop"; break; }
                std::printf("TOK %d\n", tok);
                std::fflush(stdout);
                n_gen++;
                if (k + 1 < limit) {
                    r = fwd.step(tok);
                    processed.push_back(tok);
                }
            }
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0).count();
            std::printf("DONE %d %.1f %s\n", n_gen, ms, finish);
            std::fflush(stdout);
            // Checkpoint the just-generated conversation into the LRU so the
            // next turn (prompt = this prompt + reply + new user msg) hits.
            if (processed.size() > prompt.size() || prompt.size() == prefix_hit)
                save_lru_checkpoint(processed);
        }
        std::string rep = fwd.arena_report();
        if (!rep.empty()) std::fprintf(stderr, "%s\n", rep.c_str());
        return 0;
    }

    // ---------------- compare mode: teacher-forced parity ----------------
    if (cmp_logits) {
        RefBins rb;
        if (!load_logits(cmp_logits, rb, err) || !load_tokens(cmp_tokens, rb, err)) {
            std::fprintf(stderr, "ref load failed: %s\n", err.c_str());
            return 6;
        }
        if (ids.empty()) { std::fprintf(stderr, "--compare needs --ids-file\n"); return usage(); }
        int n = rb.n;
        if (steps > 0 && steps + 1 < n) n = steps + 1;  // steps = extra decodes
        if (rb.tokens.size() < (size_t)n) n = (int)rb.tokens.size();
        std::fprintf(stderr, "compare: %d steps (prefill of %zu prompt tokens = step 0), nv=%d\n",
                     n, ids.size(), rb.nv);

        fwd.reset();
        std::vector<std::vector<float>> rows;
        int agree = 0;
        bool cos_ok = true;
        const int64_t nv_cmp = std::min<int64_t>(rb.nv, fwd.vocab_effective());
        std::printf("step  argmax     ref_token  match  cosine\n");
        m64::StepResult r = fwd.prefill(ids);  // logits after last prompt token = step 0
        for (int k = 0; k < n; k++) {
            const float * mine = r.logits;
            std::vector<float> row(mine, mine + nv_cmp);
            rows.push_back(row);
            double c = cosine(row.data(), rb.logits.data() + (size_t)k * rb.nv, (size_t)nv_cmp);
            int32_t am = r.argmax;
            bool m = am == rb.tokens[k];
            if (m) agree++;
            if (c < 0.999) cos_ok = false;
            std::printf("%4d  %-10d %-10d %-6s %.6f\n", k, am, rb.tokens[k],
                        m ? "yes" : "NO", c);
            if (k + 1 < n) r = fwd.step(rb.tokens[k]);  // teacher-forced
        }
        std::printf("cosine min gate 0.999: %s\n", cos_ok ? "OK" : "FAIL");
        std::printf("argmax agreement: %d/%d\n", agree, n);
        bool pass = cos_ok && agree == n;
        std::printf("%s\n", pass ? "PASS" : "FAIL");
        if (!out_prefix.empty()) {
            if (write_logits(out_prefix, rows, nv_cmp))
                std::printf("wrote %s_logits.bin\n", out_prefix.c_str());
            else
                std::fprintf(stderr, "failed to write %s_logits.bin\n", out_prefix.c_str());
        }
        std::string rep = fwd.arena_report();
        if (!rep.empty()) std::fprintf(stderr, "%s\n", rep.c_str());
        return pass ? 0 : 1;
    }

    // ---------------- plain / bench mode: greedy self-decode ----------------
    if (ids.empty()) { std::fprintf(stderr, "no --ids-file given\n"); return usage(); }
    int nsteps = steps > 0 ? steps : 8;
    fwd.reset();
    m64::StepResult r = fwd.prefill(ids);
    int32_t tok = r.argmax;
    std::vector<std::vector<float>> rows;
    rows.assign(1, std::vector<float>(r.logits, r.logits + fwd.vocab_effective()));
    std::printf("prompt=%zu prefill_argmax=%d\n", ids.size(), tok);
    double t0 = 0;
    (void)t0;
    for (int k = 0; k < nsteps; k++) {
        std::printf("%d ", tok);
        fflush(stdout);
        r = fwd.step(tok);
        tok = r.argmax;
        if (bench) rows.assign(1, std::vector<float>());  // no logits storage in bench
    }
    std::printf("%d\n", tok);
    if (bench) {
        auto t1 = std::chrono::steady_clock::now().time_since_epoch();
        (void)t1;
        // timed decode: run a fresh sequence with timing around decode steps
        fwd.reset();
        fwd.prefill(ids);
        auto t_start = std::chrono::steady_clock::now();
        int32_t t2 = r.argmax;
        for (int k = 0; k < 8; k++) { t2 = fwd.step(t2).argmax; }
        auto t_end = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        std::printf("bench: 8 decode steps in %.1f ms => %.2f tok/s\n", ms, 8000.0 / ms);
        (void)t0;
    }
    if (!out_prefix.empty() && !bench) write_logits(out_prefix, rows, fwd.vocab_effective());
    std::string rep = fwd.arena_report();
    if (!rep.empty()) std::fprintf(stderr, "%s\n", rep.c_str());
    return 0;
}
