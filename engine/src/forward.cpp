// forward.cpp — Inkling-2-Mini (inkling2_loop) resident CPU forward pass.
//
// Semantics pinned 1:1 to the llama.cpp oracle build at e300919cc704ddde3a8a8346a8714f57eeeee71b:
//   src/models/inkling2.cpp        clip/mix/convolve/control/build_layer/build_output
//   src/models/inkling2-loop.cpp   prelude/core/coda pass scheduling, bank/message passing
//   src/llama-graph.cpp            build_moe_ffn (sigmoid router, argsort_top_k, SITU,
//                                  weights/(sum+eps)*scale after down), build_norm
//   ggml CPU kernels               ssm_conv (ops.cpp), soft_max_ext (sinks: zero sink),
//                                  flash_attn_ext F16 emulation (Q/K/V/bias F16 casts)
// Contract doc: docs/refs/inkling2-oracle-semantics.md
//
// Numerics: f32 everywhere with -ffp-contract=off; f32xf32 dots accumulate in double
// (ggml_vec_dot_f32 semantics), softmax denominators accumulate in double
// (ggml_vec_soft_max_f32 semantics), q8_0 GEMVs accumulate per-block in float.
// Reduction orders are always fixed sequential loops; row-parallel GEMVs use a
// persistent pool with fixed partitions over independent output rows, so results
// are bitwise deterministic.
#include "forward.hpp"

#include "forward_impl.hpp"

namespace m64 {
InklingForward::InklingForward() = default;
InklingForward::~InklingForward() {
    if (impl_) { impl_->pool.stop(); }
}

bool InklingForward::init(PackReader & pack, int context_len, std::string & err) {
    impl_ = std::make_unique<Impl>();
    impl_->mapped_bytes = (size_t)pack.file_size();
    return impl_->init(pack, context_len, err);
}

StepResult InklingForward::step(int32_t token) {
    return impl_->forward_token(token, true);
}

StepResult InklingForward::prefill(const std::vector<int32_t> & prompt) {
    StepResult r{nullptr, 0, 0};
    for (size_t i = 0; i + 1 < prompt.size(); i++) r = impl_->forward_token(prompt[i], false);
    r = impl_->forward_token(prompt.back(), true);
    return r;
}

void InklingForward::reset() { if (impl_) impl_->reset(); }
int64_t InklingForward::vocab_effective() const { return impl_->vocab_eff; }
int64_t InklingForward::vocab_logit_rows() const { return impl_->vocab_rows; }
size_t InklingForward::resident_bytes() const { return impl_ ? impl_->resident_bytes() : 0; }

}  // namespace m64
