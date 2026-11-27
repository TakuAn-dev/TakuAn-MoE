// gemma4_forward_metal.hpp — Gemma 4 (arch "gemma4") resident Metal GPU forward.
//
// GPU execution of the exact semantics pinned in gemma4_forward.cpp (CPU
// oracle): per-layer sliding_window_pattern, per-layer head_count_kv,
// full-attention tied-V layers (V = K BEFORE K-norm/RoPE), per-head Q/K
// RMSNorm (K weighted, V unweighted), NEOX RoPE with rope_freqs division on
// full layers, f_attention_scale = 1.0, router input
// rms_norm * (1/sqrt(n_embd)) * ffn_gate_inp.scale, parallel shared-MLP +
// routed MoE, post_ffw_norm_1/2 + post_ffw_norm, layer_output_scale, tied LM
// head fallback, final softcap. All math is f32 on GPU through the g4_*
// kernels in engine/kernels/m64_gemma4.metal; parity vs the CPU oracle is
// asserted by engine/tests/test_gemma4_metal.mm (cos >= 0.9999 FP16 KV).
//
// Milestone 6B: resident-only (whole pack wrapped zero-copy, KV in Metal
// buffers). Expert arena streaming is NOT implemented yet: init() rejects
// arena_gb > 0.0 cleanly. io_mode is accepted for API symmetry and ignored.
#ifndef M64_GEMMA4_FORWARD_METAL_HPP
#define M64_GEMMA4_FORWARD_METAL_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "forward.hpp"  // StepResult (shared result type)

namespace m64 {

class PackReader;

class Gemma4ForwardMetal {
public:
    Gemma4ForwardMetal();
    // Out-of-line so std::unique_ptr<Impl> works with the forward-declared
    // Impl (Apple clang instantiates ~unique_ptr<Incomplete> eagerly).
    ~Gemma4ForwardMetal();

    bool init(PackReader & pack, int context_len, std::string & err,
              double arena_gb = 0.0, const std::string & io_mode = "pread",
              bool kv_q8 = false);
    StepResult step(int32_t token);
    StepResult prefill(const std::vector<int32_t> & prompt);
    // Milestone 8A: rows per batched-prefill chunk (1..64; 0 = default 64).
    void set_prefill_chunk(int n);
    void reset();
    int64_t current_pos() const;
    int64_t vocab_effective() const;    // real vocab (unpadded)
    int64_t vocab_logit_rows() const;   // LM-head tensor rows (padded)
    size_t resident_bytes() const;
    std::string arena_report() const;   // "" until arena streaming exists

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace m64

#endif  // M64_GEMMA4_FORWARD_METAL_HPP
