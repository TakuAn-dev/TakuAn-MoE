// forward.hpp — Inkling-2-Mini (inkling2_loop) resident forward pass, CPU reference path.
//
// Semantics are pinned 1:1 to the llama.cpp oracle graph:
//   llama.cpp src/models/inkling2.cpp        (mix/convolve/control/build_layer/build_output)
//   llama.cpp src/models/inkling2-loop.cpp   (pass scheduling, bank/message passing)
//   llama.cpp build_moe_ffn (LLM_FFN_SITU, sigmoid router, expert_weights_scale)
//   ggml_ssm_conv / ggml_mha(kq_b, sinks) / rel_pos table generation
//
// The engine is RESIDENT: all pack tensors are mmapped once and all activations,
// KV caches (one bank per cache slot incl. loop passes) and recurrent conv states
// are preallocated; no allocation happens inside forward().
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pack_reader.hpp"

namespace m64 {

// Per-token logits after the final layer: [n_vocab_eff] f32, softcapped
// (30 * tanh(z/30)) exactly like build_output.
struct StepResult {
    const float * logits;   // valid until next forward() call
    int64_t n_logits;
    int32_t argmax;         // greedy token for this step
};

class InklingForward {
public:
    // Out-of-line so std::unique_ptr<Impl> instantiates with a complete Impl.
    InklingForward();
    ~InklingForward();

    // Resolve tensors from the pack, hard-check arch == inkling2_loop, allocate
    // all caches (context_len tokens, n_slot = n_phys + (passes-1)*core cache slots).
    bool init(PackReader & pack, int context_len, std::string & err);

    // Feed one greedy token (call once per step after prefill of previous steps).
    // tokens are processed sequentially; returns logits for THIS token position.
    StepResult step(int32_t token);

    // Prefill a prompt (all tokens except last), then return logits for last token.
    StepResult prefill(const std::vector<int32_t> & prompt);

    void reset();  // rewind caches (same session, new sequence)

    int64_t vocab_effective() const;   // n_vocab (200058) per GGUF tokenizer
    int64_t vocab_logit_rows() const;  // output.weight rows (201024)
    size_t resident_bytes() const;     // activations + caches + mapped weights (accounting)

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace m64
