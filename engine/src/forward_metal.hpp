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
    std::vector<uint16_t> swa_kv;    // FP16 K+V snapshot of SWA ring slots only
};

class InklingForwardMetal {
public:
    // Out-of-line so std::unique_ptr<Impl> instantiates with a complete Impl.
    InklingForwardMetal();
    ~InklingForwardMetal();

    // arena_gb > 0 selects the Phase 2B bounded path: compact mlock'd resident
    // weights + mlock'd Hot Expert LRU Arena streamed from NVMe (io_mode
    // "pread" | "mtlio"). arena_gb == 0 keeps the Phase 2A-opt full-pack path.
    bool init(PackReader & pack, int context_len, std::string & err,
              double arena_gb = 0.0, const std::string & io_mode = "pread");
    StepResult step(int32_t token);
    StepResult prefill(const std::vector<int32_t> & prompt);
    void reset();
    // ---- Bitwise-exact prefix caching (save/restore at the current pos) ----
    int64_t current_pos() const;
    // Snapshot rs + SWA ring slots at the current pos; requires
    // 0 < pos < ctx and tokens.size() == pos. Safe to call between
    // prefill/step calls (every gpu_step ends with waitUntilCompleted).
    bool save_prefix_state(PrefixState & out, const std::vector<int32_t> & tokens) const;
    // Jump the engine back to in.pos with the snapshotted state (bit-exact).
    bool restore_prefix_state(const PrefixState & in);
    // ---- SSD-persistent prefix cache (.m64kv, see prefix_disk.hpp) ----
    // Stable fingerprint of pack bytes + KV-relevant hparams (deliberately
    // WITHOUT ctx, so a prefix cached at --ctx 131072 loads at --ctx 32768 too).
    uint64_t pack_fingerprint() const;
    // Snapshot the current pos to a self-contained .m64kv on disk (rs + SWA
    // rings + used 0..pos-1 of every full-attention slot, xxh64-guarded).
    bool save_disk_prefix(const std::string & path, const std::vector<int32_t> & tokens,
                          std::string & err) const;
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
