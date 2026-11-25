// gemma4_forward.hpp — Gemma 4 (arch "gemma4") resident CPU forward oracle.
//
// Standalone reference oracle for Gemma 4 dense MoE packs (gemma-4-26B-A4B and
// the tiny_gemma4 fixture). Semantics pinned to the upstream oracle build
// llama.cpp/src/models/gemma4.cpp (per-layer SWA pattern, per-layer KV-head
// counts, per-head Q/K RMSNorm, NEOX RoPE with rope_freqs division on full
// layers, tied-V layers with V = K before norm/RoPE, f_attention_scale = 1.0
// (no 1/sqrt(d) scaling), parallel shared-MLP + routed-MoE FFN with router
// RMSNorm/scale, expert renorm (w/(sum+1e-20)) then ffn_down_exps.scale, and
// final logit softcap). Numerics mirror forward.cpp conventions: f32 with
// -ffp-contract=off, fixed sequential reduction orders -> bitwise deterministic.
//
// Unlike InklingForward this oracle dequantizes weights to f32 per dot (no
// quant_dot fast path) so it supports every dequant.hpp dtype per part:
// f32/f16/bf16/q8_0/q5_0/q5_1/q4_k weights and per-part expert dtypes.
#ifndef M64_GEMMA4_FORWARD_HPP
#define M64_GEMMA4_FORWARD_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "forward.hpp"  // StepResult (shared result type)

namespace m64 {

class PackReader;

class Gemma4Forward {
public:
    Gemma4Forward();
    // Out-of-line so std::unique_ptr<Impl> works with the forward-declared Impl
    // (Apple clang 21 + libc++ instantiate ~unique_ptr<Incomplete> eagerly).
    ~Gemma4Forward();

    bool init(PackReader & pack, int context_len, std::string & err);
    StepResult step(int32_t token);
    StepResult prefill(const std::vector<int32_t> & prompt);
    void reset();
    int64_t vocab_effective() const;    // real vocab (unpadded)
    int64_t vocab_logit_rows() const;   // LM-head tensor rows (padded)
    size_t resident_bytes() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace m64

#endif
