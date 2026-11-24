// expert_arena.hpp — Phase 2B: mlock-pinned Hot Expert LRU Arena + async NVMe
// streaming (F_NOCACHE pread worker pool / MTLIOCommandQueue).
//
// Cold experts are streamed from the .m64pack on NVMe DIRECTLY into the arena
// (never through the OS page cache, never through the mmap'd pack view). The
// arena is one 16 KB page-aligned mlock'd region wrapped zero-copy as an
// MTLBuffer; the MoE batch-8 kernels address it as
//   arena_base + slot_id * expert_stride + part_offset
// where slot_id comes from the LRU resolved per-token top-8 selection.
#pragma once
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace m64 {

// One mlock-pinned, 16 KiB page-aligned anonymous region (mmap MAP_ANON +
// mlock; hard-fails instead of silently degrading if pinning is refused).
class LockedRegion {
public:
    LockedRegion() = default;
    ~LockedRegion();
    LockedRegion(const LockedRegion &) = delete;
    LockedRegion & operator=(const LockedRegion &) = delete;
    bool alloc(size_t bytes, std::string & err);
    void release();
    void * ptr() const { return p_; }
    size_t bytes() const { return n_; }
    bool mlocked() const { return mlocked_; }
private:
    void * p_ = nullptr;
    size_t n_ = 0;
    bool mlocked_ = false;
};

// Persistent worker pool over one O_RDONLY + F_NOCACHE fd. read_batch()
// enqueues all jobs in parallel and joins cleanly before returning.
class PreadPool {
public:
    struct Job { void * dst; uint64_t file_off; };
    ~PreadPool() { close(); }
    bool open(const std::string & path, int nworkers, std::string & err);
    void close();
    bool read_batch(const std::vector<Job> & jobs, uint64_t nbytes,
                    uint64_t * elapsed_ns, std::string & err);
    bool ok() const { return fd_ >= 0; }
    int fd() const { return fd_; }
private:
    void worker();
    int fd_ = -1;
    std::vector<std::thread> workers_;
    std::mutex m_;
    std::condition_variable cv_work_, cv_done_;
    std::deque<std::function<void()>> q_;
    uint64_t pending_ = 0;
    bool stop_ = false, poison_ = false;
    std::string perr_;
};

struct ArenaTelemetry {
    uint64_t hits = 0, misses = 0, evictions = 0;
    uint64_t ssd_read_bytes = 0, ssd_ns = 0;
    size_t arena_slots = 0, arena_bytes = 0;
    bool mtlio = false;
};

struct ExpertArenaCfg {
    std::string pack_path;
    std::vector<uint64_t> layer_base;  // absolute file byte offset per phys layer
    uint64_t exp_stride = 0;           // bytes per expert blob (16 KiB aligned)
    int n_experts = 128;               // experts per layer
    size_t num_slots = 0;              // arena capacity (>= 8 required)
    std::string io_mode = "pread";     // "pread" | "mtlio" (mtlio falls back with warning)
    int nworkers = 8;
};

// One miss-streaming request: dst inside the arena + absolute file offset.
struct ExpertArenaSlotRead { const uint8_t * dst; uint64_t file_off; };

class ExpertArena {
public:
    ~ExpertArena();
    bool create(const ExpertArenaCfg & cfg, std::string & err);

    void * base() const { return arena_.ptr(); }
    size_t slot_bytes() const { return cfg_.exp_stride; }
    size_t num_slots() const { return cfg_.num_slots; }
    size_t total_bytes() const { return arena_.bytes(); }
    bool mlocked() const { return arena_.mlocked(); }
    // MTLBuffer (id) wrapping the arena, created only for the mtlio backend
    // so the engine can bind THE SAME buffer (avoids double wrap_external).
    void * mtl_buffer() const;
    bool mtlio_active() const { return mtl_active_; }

    // Resolve the layer's top-8 experts into arena slots (LRU, never evicting
    // a slot already selected in the SAME call). Streams misses from NVMe.
    bool resolve_top8(int layer, const uint32_t sel[8], uint32_t slot_out[8],
                      std::string & err);

    // Resolve an arbitrary batch of expert IDs (<= num_slots unique) in one pass,
    // pinning every slot in the batch so cold misses stream together cleanly.
    bool resolve_batch(int layer, const uint32_t * exps, int n_exps,
                       uint32_t * slot_out, std::string & err);

    const ArenaTelemetry & telemetry() const { return tel_; }

private:
    ExpertArenaCfg cfg_;
    LockedRegion arena_;
    PreadPool pool_;
    // LRU: array doubly-linked list, head = MRU. table_[key] = slot or -1.
    std::vector<int32_t> table_, slot_key_, prev_, next_;
    int32_t head_ = -1, tail_ = -1;
    std::vector<int32_t> free_;
    ArenaTelemetry tel_;
    void * mtl_ = nullptr;   // opaque MtlIo* (mtlio backend only)
    bool mtl_active_ = false;

    int32_t pick_slot(int layer, uint32_t exp, const uint32_t * pins, int npin);
    void touch(int32_t s);
    void link_front(int32_t s);
    void unlink(int32_t s);
    int32_t victim(const uint32_t * pins, int npin);
};

}  // namespace m64
