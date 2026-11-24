/* m64.h — Mac64-MoE Engine public C API (Phase 1: resident forward) */
#ifndef M64_H
#define M64_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define M64_OK 0
#define M64_ERR_IO (-1)
#define M64_ERR_FORMAT (-2)
#define M64_ERR_ALIGN (-3)
#define M64_ERR_HASH (-4)
#define M64_ERR_UNSUPPORTED (-5)
#define M64_ERR_NOMEM (-6)

typedef struct m64_ctx m64_ctx;

typedef struct {
    const char *pack_path;     /* .m64pack file (required) */
    const char *archive_path;  /* MTLBinaryArchive cache path (NULL = default beside pack) */
    int verbose;               /* non-zero: print residency/compile logs */
    int verify_hashes;         /* non-zero: xxh64-verify all tensors at load */
} m64_create_params;

typedef struct {
    int rc;
    const char *message;       /* static/thread-local text, valid until next call */
} m64_status;

/* Library lifecycle --------------------------------------------------- */
m64_status m64_last_status(void);

/* Create context: reads header, asserts alignment invariants, maps blob,
 * compiles/loads Metal kernels (MTLBinaryArchive-cached). */
m64_ctx *m64_create(const m64_create_params *params);
void m64_destroy(m64_ctx *ctx);

/* Residency accounting ------------------------------------------------- */
typedef struct {
    uint64_t total_weights_bytes;
    uint64_t hot_weights_bytes;
    uint64_t cold_expert_bytes;
    uint64_t gpu_mapped_bytes;   /* noCopy-mapped weight bytes handed to Metal */
    uint64_t cpu_mapped_bytes;
    uint64_t kv_cache_bytes;     /* allocated KV for current capacity */
    uint64_t scratch_bytes;
} m64_residency_info;

int m64_get_residency(m64_ctx *ctx, m64_residency_info *out);

/* Model info ----------------------------------------------------------- */
int m64_get_n_vocab_out(m64_ctx *ctx);   /* logits view width (200058 for Inkling) */
int m64_get_n_vocab_emb(m64_ctx *ctx);   /* embedding rows */
int m64_get_n_layer(m64_ctx *ctx);

/* Prompt + greedy decode ---------------------------------------------- */
/* Feed pre-tokenised prompt ids, run prefill, then n_decode greedy steps.
 * logits_out (optional) receives the *final* logits row (float32).
 * tokens_out receives [n_prompt + n_decode] ids (prompt echoed first). */
int m64_generate(m64_ctx *ctx, const int32_t *prompt_ids, int n_prompt,
                 int n_decode, int32_t *tokens_out, int tokens_cap,
                 float *logits_out);

/* Single forward step with logits capture at every position (parity mode):
 * out_logits must hold (n_ids * n_vocab_out) float32, row-major. */
int m64_forward_logits(m64_ctx *ctx, const int32_t *ids, int n_ids,
                       float *out_logits);

/* Version -------------------------------------------------------------- */
const char *m64_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* M64_H */
