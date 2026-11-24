// metal_runtime.mm — implementation of metal_runtime.hpp.
//
// Metal 4 (macOS 26+) archive flow on this SDK (no legacy MTLCompileOptions.binaryArchive):
//   load:  MTLBinaryArchive (legacy, macOS 11+) from disk file; its contents are
//          additionally exposed via -[MTLDevice newArchiveWithURL:] (id<MTL4Archive>)
//          and plugged into MTL4CompilerDescriptor.lookupArchives so unchanged
//          functions skip MSL codegen.
//   save:  after compiling the library, addFunctionWithDescriptor:library: into the
//          legacy archive and serializeToURL for next boot.
// Fallback chain: MTL4Compiler(+archive) -> MTL4Compiler -> newLibraryWithSource:.
#import "metal_runtime.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>

namespace m64 {

static std::string read_text(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

MetalRuntime::~MetalRuntime() = default;  // ARC releases ObjC members

bool MetalRuntime::init(const std::string & kernel_dir, const std::string & archive_path,
                        std::string & err) {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) { err = "no Metal device"; return false; }
    queue_ = [device_ newCommandQueue];

    // 1. Gather + concatenate MSL sources (deterministic name order).
    NSString * dir = [NSString stringWithUTF8String:kernel_dir.c_str()];
    NSDirectoryEnumerator * en = [[NSFileManager defaultManager] enumeratorAtPath:dir];
    NSMutableArray<NSString *> * files = [NSMutableArray array];
    for (NSString * name in en)
        if ([[name pathExtension] isEqualToString:@"metal"]) [files addObject:name];
    [files sortUsingSelector:@selector(compare:)];
    NSMutableString * src = [NSMutableString string];
    // Shared POD args header first: newLibraryWithSource from a string cannot resolve
    // local #include, so m64_p2a.metal expects these structs to be pre-concatenated.
    {
        std::string hdr = read_text(std::string([dir UTF8String]) + "/m64_p2a_args.h");
        if (!hdr.empty()) [src appendString:[NSString stringWithUTF8String:hdr.c_str()]];
    }
    for (NSString * name in files) {
        std::string txt = read_text(std::string([dir UTF8String]) + "/" + std::string([name UTF8String]));
        if (txt.empty()) { err = "empty kernel file: " + std::string([name UTF8String]); return false; }
        [src appendString:[NSString stringWithUTF8String:txt.c_str()]];
    }
    if (src.length == 0) { err = "no .metal sources in " + kernel_dir; return false; }

    MTLCompileOptions * opts = [MTLCompileOptions new];
    if ([opts respondsToSelector:@selector(setMathMode:)])
        opts.mathMode = MTLMathModeSafe;  // determinism: no fast-math / FMA surprises

    // 2. Persistent archive (legacy on-disk format, macOS 11+).
    NSURL * arch_url = nil;
    archive_path_ = archive_path;
    if (!archive_path.empty()) {
        NSString * p = [NSString stringWithUTF8String:archive_path.c_str()];
        [[NSFileManager defaultManager] createDirectoryAtPath:[p stringByDeletingLastPathComponent]
                                  withIntermediateDirectories:YES attributes:nil error:nil];
        arch_url = [NSURL fileURLWithPath:p];
        MTLBinaryArchiveDescriptor * ad = [MTLBinaryArchiveDescriptor new];
        if ([[NSFileManager defaultManager] fileExistsAtPath:p]) {
            ad.url = arch_url;
            NSError * ae = nil;
            archive_ = [device_ newBinaryArchiveWithDescriptor:ad error:&ae];
            if (!archive_)
                fprintf(stderr, "warn: archive load failed (%s); will rebuild\n",
                        ae.localizedDescription.UTF8String);
        } else {
            ad.url = nil;  // fresh in-memory archive
            NSError * ae = nil;
            archive_ = [device_ newBinaryArchiveWithDescriptor:ad error:&ae];
        }
    }

    // 3. Compile MSL at load. (Note: this macOS-27 SDK exposes no archive-lookup hook on
    //    MTLCompileOptions/MTL4CompilerDescriptor, so codegen always runs here; the archive
    //    persists compiled functions on disk for durability/inspection and a future MTL4
    //    compiler-task fast path. 9 KB MSL => ~1-2 s, acceptable for Phase 1 boot.)
    {
        NSError * le = nil;
        library_ = [device_ newLibraryWithSource:src options:opts error:&le];
        if (!library_) {
            err = std::string("MSL compile error: ") +
                  std::string(le.localizedDescription ? le.localizedDescription.UTF8String : "?");
            return false;
        }
    }
    return true;
}

id<MTLComputePipelineState> MetalRuntime::pipeline(const char * fn_name) {
    std::string key(fn_name);
    auto it = pipelines_.find(key);
    if (it != pipelines_.end()) return it->second;
    NSError * e = nil;
    id<MTLFunction> f = [library_ newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
    if (!f) { fprintf(stderr, "m64: kernel not found: %s\n", fn_name); return nil; }
    id<MTLComputePipelineState> p = [device_ newComputePipelineStateWithFunction:f error:&e];
    if (!p) {
        fprintf(stderr, "m64: pipeline error %s: %s\n", fn_name,
                e.localizedDescription.UTF8String);
        return nil;
    }
    pipelines_[key] = p;
    // Persist compiled function into the on-disk archive (best effort, first time only).
    if (archive_) {
        static NSMutableArray * saved = [NSMutableArray array];
        NSString * fname = [NSString stringWithUTF8String:fn_name];
        if (![saved containsObject:fname]) {
            MTLFunctionDescriptor * fd = [MTLFunctionDescriptor new];
            fd.name = fname;
            NSError * ae = nil;
            if ([archive_ addFunctionWithDescriptor:fd library:library_ error:&ae]) {
                [saved addObject:fname];
                if (!archive_path_.empty() && [saved count] == 1) {
                    NSURL * url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:archive_path_.c_str()]];
                    NSError * se = nil;
                    if (![archive_ serializeToURL:url error:&se])
                        fprintf(stderr, "warn: archive serialize: %s\n",
                                se.localizedDescription.UTF8String);
                }
            } else {
                static BOOL warned = NO;
                if (!warned) { warned = YES;
                    fprintf(stderr, "warn: archive addFunction: %s\n",
                            ae.localizedDescription.UTF8String ?: "?"); }
            }
        }
    }
    return p;
}

id<MTLBuffer> MetalRuntime::wrap_external(void * ptr, size_t len, NSString * label) {
    NSError * e = nil;
    id<MTLBuffer> b = [device_ newBufferWithBytesNoCopy:ptr length:len
                                               options:MTLResourceStorageModeShared
                                           deallocator:nil];
    if (!b) {
        fprintf(stderr, "m64: wrap_external failed (%s): %s\n",
                label.UTF8String ?: "?", e.localizedDescription.UTF8String);
        return nil;
    }
    b.label = label;
    externals_.push_back(b);
    return b;
}

id<MTLBuffer> MetalRuntime::alloc_shared(size_t bytes, NSString * label) {
    id<MTLBuffer> b = [device_ newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    b.label = label;
    return b;
}

void MetalRuntime::wait(id<MTLCommandBuffer> cmdbuf) {
    [cmdbuf commit];
    [cmdbuf waitUntilCompleted];
}

uint64_t MetalRuntime::max_working_set() const {
    return [device_ recommendedMaxWorkingSetSize];
}

}  // namespace m64
