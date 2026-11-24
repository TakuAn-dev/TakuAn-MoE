// expert_arena.mm — Phase 2B implementation: LockedRegion / PreadPool /
// ExpertArena LRU + NVMe streaming backends (F_NOCACHE pread pool, optional
// MTLIOCommandQueue with graceful fallback).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "expert_arena.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

namespace m64 {

// ------------------------------- LockedRegion -------------------------------

LockedRegion::~LockedRegion() { release(); }

void LockedRegion::release() {
    if (!p_) return;
    if (mlocked_) munlock(p_, n_);
    munmap(p_, n_);
    p_ = nullptr; n_ = 0; mlocked_ = false;
}

bool LockedRegion::alloc(size_t bytes, std::string & err) {
    release();
    n_ = (bytes + 16383u) & ~size_t(16383u);  // 16 KiB page aligned
    if (n_ == 0) { err = "locked region: zero size"; return false; }
    p_ = mmap(nullptr, n_, PROT_READ | PROT_WRITE, MAP_ANON | MAP_PRIVATE, -1, 0);
    if (p_ == MAP_FAILED) {
        p_ = nullptr; n_ = 0;
        err = std::string("locked region: mmap(") + std::to_string(n_) + " B): " + strerror(errno);
        return false;
    }
    if (mlock(p_, n_) != 0) {
        int e = errno;
        err = std::string("locked region: mlock(") + std::to_string(n_ / (1024 * 1024)) +
              " MiB) failed: " + strerror(e) + " (ulimit -l?)";
        munmap(p_, n_); p_ = nullptr; n_ = 0;
        return false;
    }
    mlocked_ = true;
    return true;
}

// -------------------------------- PreadPool ---------------------------------

static bool pread_exact(int fd, void * dst, uint64_t off, uint64_t len, int * errn) {
    uint8_t * p = (uint8_t *)dst;
    uint64_t done = 0;
    while (done < len) {
        ssize_t n = ::pread(fd, p + done, (size_t)(len - done), (off_t)(off + done));
        if (n > 0) { done += (uint64_t)n; continue; }
        if (n == 0) { *errn = EIO; return false; }  // unexpected EOF
        if (errno == EINTR) continue;
        *errn = errno;
        return false;
    }
    return true;
}

bool PreadPool::open(const std::string & path, int nworkers, std::string & err) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) { err = "pread pool: open " + path + ": " + strerror(errno); return false; }
    if (fcntl(fd_, F_NOCACHE, 1) != 0) {
        fprintf(stderr, "m64arena: WARNING fcntl(F_NOCACHE) failed: %s (page-cache bypass lost)\n",
                strerror(errno));
    }
    if (nworkers < 1) nworkers = 1;
    workers_.reserve((size_t)nworkers);
    for (int i = 0; i < nworkers; i++) workers_.emplace_back([this] { worker(); });
    return true;
}

void PreadPool::close() {
    {
        std::lock_guard<std::mutex> lk(m_);
        if (workers_.empty() && fd_ < 0) return;
        stop_ = true;
    }
    cv_work_.notify_all();
    for (auto & t : workers_) t.join();
    workers_.clear();
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    std::lock_guard<std::mutex> lk(m_);
    stop_ = false; pending_ = 0; q_.clear();
}

void PreadPool::worker() {
    for (;;) {
        std::function<void()> job;
        {
            std::unique_lock<std::mutex> lk(m_);
            cv_work_.wait(lk, [this] { return stop_ || !q_.empty(); });
            if (stop_ && q_.empty()) return;
            job = std::move(q_.front());
            q_.pop_front();
        }
        job();
        {
            std::lock_guard<std::mutex> lk(m_);
            if (--pending_ == 0) cv_done_.notify_all();
        }
    }
}

bool PreadPool::read_batch(const std::vector<Job> & jobs, uint64_t nbytes,
                           uint64_t * elapsed_ns, std::string & err) {
    if (jobs.empty()) { if (elapsed_ns) *elapsed_ns = 0; return fd_ >= 0; }
    {
        std::lock_guard<std::mutex> lk(m_);
        poison_ = false; perr_.clear();
        for (const auto & j : jobs) {
            q_.push_back([this, j, nbytes] {
                int e = 0;
                if (!pread_exact(fd_, j.dst, j.file_off, nbytes, &e)) {
                    std::lock_guard<std::mutex> lk2(m_);
                    if (!poison_)
                        perr_ = std::string("pread(fd, off=") + std::to_string(j.file_off) +
                                ", " + std::to_string(nbytes) + " B): " + strerror(e);
                    poison_ = true;
                }
            });
        }
        pending_ = jobs.size();
    }
    cv_work_.notify_all();
    auto t0 = std::chrono::steady_clock::now();
    {
        std::unique_lock<std::mutex> lk(m_);
        cv_done_.wait(lk, [this] { return pending_ == 0; });
    }
    if (elapsed_ns)
        *elapsed_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - t0).count();
    std::lock_guard<std::mutex> lk(m_);
    if (poison_) { err = perr_; return false; }
    return true;
}

// ------------------------------ MTLIO backend -------------------------------

namespace {
struct MtlIo {
    id<MTLDevice> dev = nil;
    id<MTLIOFileHandle> fh = nil;
    id<MTLIOCommandQueue> q = nil;
    id<MTLBuffer> buf = nil;
};
}  // namespace

void * ExpertArena::mtl_buffer() const {
#ifdef __OBJC__
    if (!mtl_active_ || !mtl_) return nullptr;
    MtlIo * io = (MtlIo *)mtl_;
    return (void *)(__bridge void *)io->buf;
#else
    return nullptr;
#endif
}

#ifdef __OBJC__
static bool mtl_open(void ** out, const std::string & path, void * base, size_t bytes,
                     std::string & err) {
    if (@available(macOS 14.0, *)) {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { err = "mtlio: no Metal device"; return false; }
        NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path.c_str()]];
        NSError * e = nil;
        id<MTLIOFileHandle> fh = [dev newIOFileHandleWithURL:url error:&e];
        if (!fh) {
            err = "mtlio: newIOFileHandleWithURL: " +
                  std::string(e.localizedDescription ? e.localizedDescription.UTF8String : "?");
            return false;
        }
        MTLIOCommandQueueDescriptor * d = [[MTLIOCommandQueueDescriptor alloc] init];
        d.maxCommandBufferCount = 8;
        id<MTLIOCommandQueue> q = [dev newIOCommandQueueWithDescriptor:d error:&e];
        if (!q) {
            err = "mtlio: newIOCommandQueueWithDescriptor: " +
                  std::string(e.localizedDescription ? e.localizedDescription.UTF8String : "?");
            return false;
        }
        id<MTLBuffer> buf = [dev newBufferWithBytesNoCopy:base length:bytes
                                                 options:MTLResourceStorageModeShared
                                            deallocator:nil];
        if (!buf) {
            err = "mtlio: newBufferWithBytesNoCopy(arena) refused";
            return false;
        }
        buf.label = @"m64-arena-mtlio";
        MtlIo * io = new MtlIo{dev, fh, q, buf};
        *out = io;
        return true;
    }
    err = "mtlio: requires macOS 14+";
    return false;
}

static void mtl_close(void * io_) { delete (MtlIo *)io_; }

static bool mtl_read(void * io_, const uint8_t * arena_base,
                     const std::vector<ExpertArenaSlotRead> & jobs, uint64_t nbytes,
                     uint64_t * elapsed_ns, std::string & err) {
    if (@available(macOS 14.0, *)) {
        MtlIo * io = (MtlIo *)io_;
        id<MTLIOCommandBuffer> cb = [io->q commandBuffer];
        if (!cb) { err = "mtlio: commandBuffer alloc failed"; return false; }
        for (const auto & j : jobs)
            [cb loadBuffer:io->buf offset:(NSUInteger)(j.dst - arena_base) size:(NSUInteger)nbytes
                      sourceHandle:io->fh sourceHandleOffset:j.file_off];
        auto t0 = std::chrono::steady_clock::now();
        [cb commit];
        [cb waitUntilCompleted];
        if (elapsed_ns)
            *elapsed_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - t0).count();
        if (cb.status != MTLIOStatusComplete) {
            NSString * msg = cb.error.localizedDescription ?: @"unknown MTLIO error";
            err = "mtlio: status=" + std::to_string((NSInteger)cb.status) + ": " + msg.UTF8String;
            return false;
        }
        return true;
    }
    err = "mtlio: requires macOS 14+";
    return false;
}
#endif  // __OBJC__

// ------------------------------- ExpertArena --------------------------------

ExpertArena::~ExpertArena() {
#ifdef __OBJC__
    if (mtl_) { mtl_close(mtl_); mtl_ = nullptr; }
#endif
}

bool ExpertArena::create(const ExpertArenaCfg & cfg, std::string & err) {
    cfg_ = cfg;
    if (cfg_.layer_base.empty() || cfg_.exp_stride == 0 || cfg_.n_experts <= 0) {
        err = "arena: bad config"; return false;
    }
    if (cfg_.num_slots < 8) {
        err = "arena: expert_budget too small (" + std::to_string(cfg_.num_slots) +
              " slots; need >= 8 for top-8 MoE)";
        return false;
    }
    // Slot spacing (exp_stride) must cover every per-layer pack stride, else a
    // streamed blob would spill into the neighbouring slot.
    for (size_t l = 0; l < cfg_.layer_stride.size(); l++)
        if (cfg_.layer_stride[l] > cfg_.exp_stride) {
            err = "arena: layer " + std::to_string(l) + " pack stride " +
                  std::to_string(cfg_.layer_stride[l]) + " exceeds slot stride " +
                  std::to_string(cfg_.exp_stride);
            return false;
        }
    if (!arena_.alloc(cfg_.num_slots * cfg_.exp_stride, err)) return false;
    const size_t nkeys = cfg_.layer_base.size() * (size_t)cfg_.n_experts;
    table_.assign(nkeys, -1);
    slot_key_.assign(cfg_.num_slots, -1);
    prev_.assign(cfg_.num_slots, -1);
    next_.assign(cfg_.num_slots, -1);
    head_ = tail_ = -1;
    free_.clear();
    for (size_t s = cfg_.num_slots; s-- > 0;) free_.push_back((int32_t)s);  // allocate 0..N-1 in order
    tel_ = ArenaTelemetry{};
    tel_.arena_slots = cfg_.num_slots;
    tel_.arena_bytes = arena_.bytes();
    if (!pool_.open(cfg_.pack_path, cfg_.nworkers, err)) return false;
    if (cfg_.io_mode == "mtlio") {
#ifdef __OBJC__
        std::string merr;
        void * io = nullptr;
        if (mtl_open(&io, cfg_.pack_path, arena_.ptr(), arena_.bytes(), merr)) {
            mtl_ = io; mtl_active_ = true; tel_.mtlio = true;
            fprintf(stderr, "m64arena: MTLIOCommandQueue active\n");
        } else {
            fprintf(stderr, "m64arena: WARNING mtlio unavailable (%s); falling back to F_NOCACHE pread pool\n",
                    merr.c_str());
        }
#else
        fprintf(stderr, "m64arena: WARNING mtlio requested but built without ObjC; using pread pool\n");
#endif
    }
    return true;
}

void ExpertArena::unlink(int32_t s) {
    const int32_t p = prev_[s], n = next_[s];
    if (p >= 0) next_[p] = n; else head_ = n;
    if (n >= 0) prev_[n] = p; else tail_ = p;
    prev_[s] = next_[s] = -1;
}

void ExpertArena::link_front(int32_t s) {
    prev_[s] = -1; next_[s] = head_;
    if (head_ >= 0) prev_[head_] = s; else tail_ = s;
    head_ = s;
}

void ExpertArena::touch(int32_t s) {
    if (head_ == s) return;
    unlink(s);
    link_front(s);
}

int32_t ExpertArena::victim(const uint32_t * pins, int npin) {
    // LRU end -> MRU walk; NEVER evict a slot already selected in this call.
    for (int32_t s = tail_; s >= 0; s = prev_[s]) {
        bool pinned = false;
        for (int i = 0; i < npin; i++) if (pins[i] == (uint32_t)s) { pinned = true; break; }
        if (!pinned) {
            table_[(size_t)slot_key_[s]] = -1;
            unlink(s);  // detach from LRU before pick_slot re-links it
            tel_.evictions++;
            return s;
        }
    }
    return -1;  // all slots pinned (impossible when num_slots >= 8 with distinct top-k)
}

int32_t ExpertArena::pick_slot(int layer, uint32_t exp, const uint32_t * pins, int npin) {
    const int32_t key = layer * cfg_.n_experts + (int32_t)exp;
    int32_t s;
    if (!free_.empty()) {
        s = free_.back();
        free_.pop_back();
    } else {
        s = victim(pins, npin);
        if (s < 0) return -1;
    }
    slot_key_[s] = key;
    table_[(size_t)key] = s;
    link_front(s);
    return s;
}

bool ExpertArena::resolve_top8(int layer, const uint32_t sel[8], uint32_t slot_out[8],
                               std::string & err) {
    return resolve_batch(layer, sel, 8, slot_out, err);
}

bool ExpertArena::resolve_batch(int layer, const uint32_t * exps, int n_exps,
                                uint32_t * slot_out, std::string & err) {
    if (layer < 0 || layer >= (int)cfg_.layer_base.size()) { err = "arena: bad layer"; return false; }
    if (n_exps <= 0) return true;
    std::vector<uint32_t> pins;
    pins.reserve((size_t)n_exps);
    std::vector<PreadPool::Job> jobs;
    jobs.reserve((size_t)n_exps);
    // Milestone 4B: pack stride may differ per layer (mixed-dtype packs);
    // arena slot spacing stays fixed at cfg_.exp_stride (= max stride).
    const uint64_t st = stride_at(layer);
    // Pass 1: hits (touch -> MRU).
    for (int i = 0; i < n_exps; i++) {
        const uint32_t e = exps[i];
        if (e >= (uint32_t)cfg_.n_experts) { err = "arena: bad expert id"; return false; }
        const int32_t key = layer * cfg_.n_experts + (int32_t)e;
        const int32_t s = table_[(size_t)key];
        if (s >= 0) {
            slot_out[i] = (uint32_t)s;
            touch(s);
            tel_.hits++;
            pins.push_back((uint32_t)s);
        } else {
            slot_out[i] = 0xFFFFFFFFu;  // pending miss
        }
    }
    // Pass 2: assign slots for misses (never evicting this batch's pinned slots).
    for (int i = 0; i < n_exps; i++) {
        if (slot_out[i] != 0xFFFFFFFFu) continue;
        const uint32_t e = exps[i];
        const int32_t key = layer * cfg_.n_experts + (int32_t)e;
        if (table_[(size_t)key] >= 0) {
            slot_out[i] = (uint32_t)table_[(size_t)key];  // handle duplicate exps[i]
            continue;
        }
        const int32_t s = pick_slot(layer, e, pins.data(), (int)pins.size());
        if (s < 0) { err = "arena: all slots pinned"; return false; }
        slot_out[i] = (uint32_t)s;
        tel_.misses++;
        jobs.push_back({arena_.ptr() ? (void *)((uint8_t *)arena_.ptr() + (size_t)s * cfg_.exp_stride)
                                     : nullptr,
                        cfg_.layer_base[(size_t)layer] + (uint64_t)e * st});
        pins.push_back((uint32_t)s);
    }
    const int njobs = (int)jobs.size();
    if (njobs == 0) return true;
    // Stream misses from NVMe (mtlio DMA into the arena if active, else pool).
    uint64_t ns = 0;
    bool ok = false;
#ifdef __OBJC__
    if (mtl_active_) {
        std::vector<ExpertArenaSlotRead> mjobs((size_t)njobs);
        for (int i = 0; i < njobs; i++) { mjobs[(size_t)i].dst = (const uint8_t *)jobs[(size_t)i].dst; mjobs[(size_t)i].file_off = jobs[(size_t)i].file_off; }
        std::string merr;
        ok = mtl_read(mtl_, (const uint8_t *)arena_.ptr(), mjobs, st, &ns, merr);
        if (!ok) {
            fprintf(stderr, "m64arena: WARNING mtlio read failed (%s); using pread pool\n", merr.c_str());
            ((MtlIo *)mtl_)->buf = nil;  // deactivate for the rest of the run
            mtl_active_ = false;
            tel_.mtlio = false;
        }
    }
#endif
    if (!ok) {
        if (!pool_.read_batch(jobs, st, &ns, err)) return false;
    }
    tel_.ssd_ns += ns;
    tel_.ssd_read_bytes += (uint64_t)njobs * st;
    return true;
}

}  // namespace m64
