// metal_runtime.hpp — Metal device, MSL-at-load compilation with MTLBinaryArchive
// disk cache, and zero-copy pack-buffer wrapping (Amendment A1).
#pragma once
#ifdef __OBJC__
#import <Metal/Metal.h>
#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>

namespace m64 {

class MetalRuntime {
public:
    // kernel_dir: directory of *.metal sources. archive_path: MTLBinaryArchive file
    // (created on first run; subsequent loads use it to skip recompilation).
    bool init(const std::string &kernel_dir, const std::string &archive_path,
              std::string &err);

    id<MTLDevice> device() const { return device_; }
    id<MTLCommandQueue> queue() const { return queue_; }
    id<MTLLibrary> library() const { return library_; }
    bool used_archive_cache() const { return archive_hit_; }

    // Compute pipeline for a kernel entry point (created lazily, cached).
    id<MTLComputePipelineState> pipeline(const char *fn_name);

    // Wrap a page-aligned host/mmap range as a GPU buffer without copying.
    // Must stay alive for the runtime lifetime (no copy, no retain of contents).
    id<MTLBuffer> wrap_external(void *ptr, size_t len, NSString *label);

    // Allocate a zeroed GPU buffer (Shared storage on Apple Silicon).
    id<MTLBuffer> alloc_shared(size_t bytes, NSString *label);

    // Synchronize: commit and wait.
    void wait(id<MTLCommandBuffer> cmdbuf);

    // GPU working-set limit (recommendedMaxWorkingSetSize bytes).
    uint64_t max_working_set() const;

    ~MetalRuntime();

private:
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    id<MTLBinaryArchive> archive_ = nil;
    bool archive_hit_ = false;
    std::string archive_path_;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
    std::vector<id<MTLBuffer>> externals_;
};

}  // namespace m64
#endif  // __OBJC__
