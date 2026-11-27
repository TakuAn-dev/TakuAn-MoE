// forward_metal.hpp — Inkling-2-Mini forward pass on the native Metal GPU path.
// Public API mirrors InklingForward (CPU oracle) exactly so m64run can select
// either backend; greedy parity vs the CPU/llama.cpp oracle is the Phase 2A gate.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "forward.hpp"   // StepResult
#include "pack_reader.hpp"
#include "prefix_disk.hpp"  // DiskPrefixHeader (DiskPrefixSnapshot, Milestone 14)

namespace m64 {

// Bitwise-exact checkpoint of the running sequence at position `pos`:
// ShortConv recurrent state + SWA ring slots. Full-attention KV slots are
// linearly indexed 0..pos-1 and are never overwritten by future positions
// >= pos, so only the 40 SWA ring slots need snapshotting. Restoring jumps
// straight back to `pos` with zero numerical error vs a full re-prefill.
struct PrefixState {
    int64_t pos = 0;
    std::vector<int32_t> tokens;     // exact token sequence processed up to pos
    std::vector<float> rs;           // ShortConv recurrent state (kNSLOT * kCH * kHR floats)
    std::vector<uint16_t> swa_kv;    // K+V snapshot of SWA ring slots (FP16 or Q8_0 units)
    // Self-contained resume ONLY (RAM stash across interrupting prompts, see
    // prefix_stash.hpp): K+V of [0..pos) for every full-attention slot, in
    // the same per-slot K-then-V order as the .m64kv `full` array. Stays
    // EMPTY for normal in-place LRU checkpoints -> zero extra overhead there.
    std::vector<uint16_t> full_kv;
};

// Milestone 14: an in-RAM .m64kv snapshot (header + every payload array) that
// snapshot_disk_prefix() captures synchronously while the GPU is idle, so the
// slow SSD write can run on a background thread (see main.cpp WARM handler).
struct DiskPrefixSnapshot {
    std::string path;
    DiskPrefixHeader hdr{};
    std::vector<int32_t> tokens;
    std::vector<float> rs;
    std::vector<uint16_t> swa;
    std::vector<uint16_t> full;
};

class InklingForwardMetal {
public:
    // Out-of-line so std::unique_ptr<Impl> instantiates with a complete Impl.
    InklingForwardMetal();
    ~InklingForwardMetal();

    // arena_gb > 0 selects the Phase 2B bounded path: compact mlock'd resident
    // weights + mlock'd Hot Expert LRU Arena streamed from NVMe (io_mode
    // "pread" | "mtlio"). arena_gb == 0 keeps the Phase 2A-opt full-pack path.
    // kv_q8 selects the Q8_0 KV cache (1088 B / 544 u16 units per token slot
    // vs 2048 B / 1024 for FP16); see the k_*_q8 kernels (Milestone 5A).
    bool init(PackReader & pack, int context_len, std::string & err,
              double arena_gb = 0.0, const std::string & io_mode = "pread",
              bool kv_q8 = false);
    StepResult step(int32_t token);
    StepResult prefill(const std::vector<int32_t> & prompt);
    // Milestone 7B: batched speculative-decoding verify pass (Metal only).
    // Advances the engine through ALL of toks (<= 9 rows) in ONE pass and
    // returns per-row argmax + last-row logits (see Impl::gpu_verify).
    bool verify_batch(const std::vector<int32_t> &toks, std::vector<int32_t> &amax,
                      std::vector<float> &last_logits, std::string &err);
    // Milestone 7C: state-only replay for the speculative Branch fast path —
    // advance the engine through toks (<= 9 rows) with ZERO LM head rows
    // (pos/KV/conv/SWA advance exactly like step/verify; no logits, no amax).
    bool advance_no_logits(const std::vector<int32_t> &toks, std::string &err);
    void reset();
    // ---- Bitwise-exact prefix caching (save/restore at the current pos) ----
    int64_t current_pos() const;
    // Snapshot rs + SWA ring slots at the current pos; requires
    // 0 < pos < ctx and tokens.size() == pos. Safe to call between
    // prefill/step calls (every gpu_step ends with waitUntilCompleted).
    bool save_prefix_state(PrefixState & out, const std::vector<int32_t> & tokens) const;
    // Jump the engine back to in.pos with the snapshotted state (bit-exact).
    // If in.full_kv is non-empty it must be full_kv_elems(in.pos) elements
    // and the used [0..in.pos) of every full-attention slot is restored too,
    // making the checkpoint self-contained against a reset kvBuf.
    bool restore_prefix_state(const PrefixState & in);
    // uint16 elements PrefixState::full_kv needs for a checkpoint at n_pos.
    size_t full_kv_elems(int64_t n_pos) const;
    // Copy [0..st.pos) K/V of every full-attention slot from the live kvBuf
    // into st.full_kv (same per-slot K-then-V order as save_disk_prefix).
    // Requires 0 < st.pos <= current pos and slot capacity >= st.pos;
    // leaves st.full_kv EMPTY otherwise. Safe between prefill/step calls.
    void capture_full_kv(PrefixState & st) const;
    // ---- SSD-persistent prefix cache (.m64kv, see prefix_disk.hpp) ----
    // Stable fingerprint of pack bytes + KV-relevant hparams (deliberately
    // WITHOUT ctx, so a prefix cached at --ctx 131072 loads at --ctx 32768 too).
    uint64_t pack_fingerprint() const;
    // Snapshot the current pos to a self-contained .m64kv on disk (rs + SWA
    // rings + used 0..pos-1 of every full-attention slot, xxh64-guarded).
    bool save_disk_prefix(const std::string & path, const std::vector<int32_t> & tokens,
                          std::string & err) const;
    // Milestone 14: same snapshot as save_disk_prefix, but captured into RAM
    // (DiskPrefixSnapshot) with ZERO disk I/O, so the caller can hand the
    // bytes to a background writer thread. Requires GPU-idle between
    // prefill/step calls, like save_prefix_state.
    bool snapshot_disk_prefix(const std::string & path, const std::vector<int32_t> & tokens,
                              DiskPrefixSnapshot & out, std::string & err) const;
    // Restore the whole warmed state IN-ENGINE (kvBuf + o.rs + pos) and hand
    // back the same state as a RAM PrefixState. False -> file invalid/corrupt.
    bool load_disk_prefix(const std::string & path, PrefixState & out_ram_state,
                          std::string & err);
    int64_t vocab_effective() const;
    int64_t vocab_logit_rows() const;
    size_t resident_bytes() const;
    // Phase 2B telemetry line (empty string when arena mode is off).
    std::string arena_report() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace m64
