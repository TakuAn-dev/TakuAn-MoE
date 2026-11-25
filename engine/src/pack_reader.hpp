// pack_reader.hpp — .m64pack v1 reader: mmap, header parse, hard alignment
// assertions, residency accounting, optional full xxh64 verification.
#pragma once
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "json_min.hpp"
#include "xxh64.hpp"

namespace m64 {

enum class DType : uint8_t { F32, F16, BF16, Q8_0, Q8_1, Q4_K, Q6_K, IQ3_XXS, Q5_0, Q5_1, UNKNOWN };

inline DType dtype_from_name(const std::string &n) {
    if (n == "f32") return DType::F32;
    if (n == "f16") return DType::F16;
    if (n == "bf16") return DType::BF16;
    if (n == "q8_0") return DType::Q8_0;
    if (n == "q8_1") return DType::Q8_1;
    if (n == "q4_k") return DType::Q4_K;
    if (n == "q6_k") return DType::Q6_K;
    if (n == "iq3_xxs") return DType::IQ3_XXS;
    if (n == "q5_0") return DType::Q5_0;
    if (n == "q5_1") return DType::Q5_1;
    return DType::UNKNOWN;
}
inline const char *dtype_name(DType d) {
    switch (d) {
        case DType::F32: return "f32"; case DType::F16: return "f16"; case DType::BF16: return "bf16";
        case DType::Q8_0: return "q8_0"; case DType::Q8_1: return "q8_1"; case DType::Q4_K: return "q4_k";
        case DType::Q6_K: return "q6_k"; case DType::IQ3_XXS: return "iq3_xxs";
        case DType::Q5_0: return "q5_0"; case DType::Q5_1: return "q5_1";
        default: return "unknown";
    }
}
// ggml block element counts: q8/q5/q4_0 groups are 32, super-block K formats are 256.
inline int dtype_block_elems(DType d) {
    switch (d) {
        case DType::F32: case DType::F16: case DType::BF16: return 1;
        case DType::Q8_0: case DType::Q8_1: case DType::Q5_0: case DType::Q5_1: return 32;
        case DType::Q4_K: case DType::Q6_K: case DType::IQ3_XXS: return 256;
        default: return 0;
    }
}
// ggml block byte size (type_size) per dtype.
inline int dtype_type_size(DType d) {
    switch (d) {
        case DType::F32: return 4;
        case DType::F16: case DType::BF16: return 2;
        case DType::Q8_0: return 34; case DType::Q8_1: return 36;
        case DType::Q5_0: return 22; case DType::Q5_1: return 24;
        case DType::Q4_K: return 144; case DType::Q6_K: return 210;
        case DType::IQ3_XXS: return 110;
        default: return 0;
    }
}

struct TensorInfo {
    std::string name;
    DType dtype = DType::UNKNOWN;
    std::vector<int64_t> shape;   // logical, row-major: [rows, cols...]
    int64_t offset = 0;           // ABSOLUTE file offset (pack format v1)
    int64_t length = 0;           // bytes
    std::string region;           // "gpu" | "expert" | "cpu"
    std::string residency;        // "hot" | "warm" | "cold"
    int64_t expert_stride = 0;    // bytes per expert (expert tensors only)
    bool is_expert = false;
    int64_t xxh64 = 0;
    // expert part shapes (gate|up|down per-expert shapes, row-major)
    std::vector<int64_t> part_gate, part_up, part_down;
    // per-part dtype names (e.g. gate/up=q4_k, down=q5_0). Empty for legacy
    // packs — callers fall back to the tensor-level dtype in that case.
    std::string part_dtype_gate, part_dtype_up, part_dtype_down;
    // per-part byte lengths (bytes per expert). Zero = unknown (legacy packs);
    // derive from part shapes x block packing when absent.
    int64_t part_bytes_gate = 0, part_bytes_up = 0, part_bytes_down = 0;
    DType part_dtype(DType fallback, int part) const {
        const std::string *s = part == 0 ? &part_dtype_gate : part == 1 ? &part_dtype_up : &part_dtype_down;
        return s->empty() ? fallback : dtype_from_name(*s);
    }
    int64_t part_bytes(int64_t fallback, int part) const {
        int64_t v = part == 0 ? part_bytes_gate : part == 1 ? part_bytes_up : part_bytes_down;
        return v ? v : fallback;
    }
};

struct ExpertIndex {
    bool present = false;
    std::string stride_mode;      // "per_layer"
    std::vector<int64_t> layer_base; // absolute file offsets, [n_layers]
    int64_t expert_stride = 0;    // bytes per expert
    std::string order;            // "gate|up|down"
};

class PackReader {
public:
    struct Header {
        int version = 0;
        int64_t blob_start = 0;
        int64_t blob_bytes = 0;
        int align_gpu_log2 = 14, align_expert_log2 = 14, align_cpu_log2 = 6;
        uint64_t blob_xxh64 = 0;
        std::string arch, model_name;
        std::map<std::string, JValuePtr> hparams;
        ExpertIndex expert_index;
        int64_t total_weights = 0, hot_weights = 0, cold_experts = 0;
    };

    void open(const std::string &path, bool verify_hashes) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("pack: cannot open " + path);
        path_ = path;
        struct stat st;
        if (::fstat(fd_, &st) != 0) throw std::runtime_error("pack: fstat failed");
        file_size_ = st.st_size;
        base_ = (const uint8_t *)mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (base_ == MAP_FAILED) throw std::runtime_error("pack: mmap failed");
        if (file_size_ < 16 || std::memcmp(base_, "M64P", 4) != 0)
            throw std::runtime_error("pack: bad magic");
        uint32_t ver;
        uint64_t hlen;
        std::memcpy(&ver, base_ + 4, 4);
        std::memcpy(&hlen, base_ + 8, 8);
        if (ver != 1) throw std::runtime_error("pack: unsupported version " + std::to_string(ver));
        if (16 + hlen > (uint64_t)file_size_) throw std::runtime_error("pack: header overruns file");
        parse_header(std::string((const char *)base_ + 16, hlen));
        if (verify_hashes) verify_all();
    }

    void close() {
        if (base_) munmap((void *)base_, file_size_);
        if (fd_ >= 0) ::close(fd_);
        base_ = nullptr; fd_ = -1;
    }

    const Header &header() const { return hdr_; }
    size_t n_tensors() const { return tensors_.size(); }
    const TensorInfo &tensor(size_t i) const { return tensors_[i]; }
    const uint8_t *data() const { return base_; }
    int64_t file_size() const { return file_size_; }
    int fd() const { return fd_; }
    const std::string & path() const { return path_; }

    const TensorInfo *find(const std::string &name) const {
        auto it = by_name_.find(name);
        return it == by_name_.end() ? nullptr : it->second;
    }
    const TensorInfo &require(const std::string &name) const {
        const TensorInfo *t = find(name);
        if (!t) throw std::runtime_error("pack: missing tensor '" + name + "'");
        return *t;
    }
    // Pointer into the mapped file for a tensor (absolute offset).
    const void *ptr(const TensorInfo &t) const { return base_ + t.offset; }

    int64_t hparam_i64(const std::string &key, int64_t fallback = 0) const {
        auto it = hdr_.hparams.find(key);
        if (it == hdr_.hparams.end()) return fallback;
        const JValue * v = it->second.get();
        // per-layer arrays (head_count etc.): value == every element
        if (v->type == JValue::Arr) return v->arr.empty() ? fallback : v->arr[0]->as_int();
        return v->as_int();
    }
    double hparam_f64(const std::string &key, double fallback = 0) const {
        auto it = hdr_.hparams.find(key);
        if (it == hdr_.hparams.end()) return fallback;
        const JValue * v = it->second.get();
        if (v->type == JValue::Arr) {
            if (v->arr.empty()) return fallback;
            v = v->arr[0].get();
        }
        if (v->type == JValue::Int) return (double)v->i;
        return v->d;
    }
    std::string hparam_str(const std::string &key, const std::string &fallback = "") const {
        auto it = hdr_.hparams.find(key);
        return it == hdr_.hparams.end() ? fallback : it->second->as_str();
    }
    // Per-layer int arrays (e.g. gemma4.attention.head_count_kv): empty when the
    // key is absent; throws if present but not an array.
    std::vector<int64_t> hparam_i64_arr(const std::string &key) const {
        std::vector<int64_t> out;
        auto it = hdr_.hparams.find(key);
        if (it == hdr_.hparams.end()) return out;
        const JValue * v = it->second.get();
        if (v->type != JValue::Arr) throw std::runtime_error("json: hparam '" + key + "' is not an array");
        for (auto &e : v->arr) out.push_back(e->as_int());
        return out;
    }
    // Per-layer bool arrays (e.g. gemma4.attention.sliding_window_pattern, stored
    // as [1, 1, ..., 0] ints in real Gemma 4 GGUF): accepts JSON bools and ints.
    std::vector<bool> hparam_bool_arr(const std::string &key) const {
        std::vector<bool> out;
        auto it = hdr_.hparams.find(key);
        if (it == hdr_.hparams.end()) return out;
        const JValue * v = it->second.get();
        if (v->type != JValue::Arr) throw std::runtime_error("json: hparam '" + key + "' is not an array");
        for (auto &e : v->arr) out.push_back(e->as_bool());
        return out;
    }

private:
    void parse_header(const std::string &json) {
        JValuePtr root = JsonParser::parse(json);
        hdr_.version = (int)root->at("version").as_int();
        hdr_.blob_start = root->at("blob_start").as_int();
        hdr_.blob_bytes = root->at("blob_bytes").as_int();
        const JValue &al = root->at("alignment_log2");
        hdr_.align_gpu_log2 = (int)al.at("gpu").as_int();
        hdr_.align_expert_log2 = (int)al.at("expert").as_int();
        hdr_.align_cpu_log2 = (int)al.at("cpu").as_int();
        hdr_.blob_xxh64 = (uint64_t)std::stoull(root->at("checksum").at("blob_xxh64").as_str(), nullptr, 16);
        const JValue &md = root->at("model");
        hdr_.arch = md.at("arch").as_str();
        hdr_.model_name = md.find("name") ? md.at("name").as_str() : "";
        if (const JValue *hp = md.find("hparams"))
            for (auto &kv : hp->obj) hdr_.hparams[kv.first] = kv.second;
        if (const JValue *ei = root->find("expert_index")) {
            hdr_.expert_index.present = true;
            hdr_.expert_index.stride_mode = ei->at("stride_mode").as_str();
            for (auto &v : ei->at("layer_base").arr) hdr_.expert_index.layer_base.push_back(v->as_int());
            hdr_.expert_index.expert_stride = ei->at("expert_stride").as_int();
            hdr_.expert_index.order = ei->at("order").as_str();
        }
        if (const JValue *rs = root->find("residency")) {
            hdr_.total_weights = rs->at("total_weights_bytes").as_int();
            hdr_.hot_weights = rs->at("hot_weights_bytes").as_int();
            hdr_.cold_experts = rs->at("cold_expert_bytes").as_int();
        }
        for (auto &tv : root->at("tensors").arr) {
            const JValue &t = *tv;
            TensorInfo ti;
            ti.name = t.at("name").as_str();
            ti.dtype = dtype_from_name(t.at("dtype").as_str());
            if (ti.dtype == DType::UNKNOWN)
                throw std::runtime_error("pack: tensor '" + ti.name + "' unknown dtype '" + t.at("dtype").as_str() + "'");
            for (auto &v : t.at("shape").arr) ti.shape.push_back(v->as_int());
            ti.offset = t.at("offset").as_int();
            ti.length = t.at("length").as_int();
            ti.region = t.at("region").as_str();
            ti.residency = t.at("residency").as_str();
            if (const JValue *x = t.find("xxh64")) ti.xxh64 = (int64_t)std::stoull(x->as_str(), nullptr, 16);
            if (ti.region == "expert") {
                ti.is_expert = true;
                ti.expert_stride = t.at("expert_stride").as_int();
                if (const JValue *p = t.find("parts")) {
                    for (auto &v : p->at("gate").arr) ti.part_gate.push_back(v->as_int());
                    for (auto &v : p->at("up").arr) ti.part_up.push_back(v->as_int());
                    for (auto &v : p->at("down").arr) ti.part_down.push_back(v->as_int());
                }
                if (const JValue *p = t.find("part_dtypes")) {
                    if (const JValue *v = p->find("gate")) ti.part_dtype_gate = v->as_str();
                    if (const JValue *v = p->find("up")) ti.part_dtype_up = v->as_str();
                    if (const JValue *v = p->find("down")) ti.part_dtype_down = v->as_str();
                }
                if (const JValue *p = t.find("part_bytes")) {
                    if (const JValue *v = p->find("gate")) ti.part_bytes_gate = v->as_int();
                    if (const JValue *v = p->find("up")) ti.part_bytes_up = v->as_int();
                    if (const JValue *v = p->find("down")) ti.part_bytes_down = v->as_int();
                }
            }
            tensors_.push_back(ti);
        }
        // Build the name map AFTER the vector is final: pointers into tensors_
        // taken during push_back are invalidated by later reallocations.
        for (const auto & t : tensors_) by_name_[t.name] = &t;
        check_alignment_invariants();
    }

    // Hard load-time assertions (DESIGN §2): gpu/expert tensors must be
    // 2^align-aligned at their ABSOLUTE file offsets.
    void check_alignment_invariants() {
        const int64_t galign = 1LL << hdr_.align_gpu_log2;
        const int64_t ealign = 1LL << hdr_.align_expert_log2;
        const int64_t calign = 1LL << hdr_.align_cpu_log2;
        if (hdr_.blob_start % 65536 != 0)
            throw std::runtime_error("pack: blob_start not 64 KiB aligned");
        for (const auto &t : tensors_) {
            int64_t a = t.region == "cpu" ? calign : (t.region == "expert" ? ealign : galign);
            if ((t.offset % a) != 0)
                throw std::runtime_error("pack: tensor '" + t.name + "' offset " + std::to_string(t.offset) +
                                         " not " + std::to_string(a) + "-aligned (region " + t.region + ")");
            if (t.offset < hdr_.blob_start || t.offset + t.length > hdr_.blob_start + hdr_.blob_bytes)
                throw std::runtime_error("pack: tensor '" + t.name + "' outside blob");
        }
    }

    void verify_all() {
        Xxh64 blob_h;
        for (const auto &t : tensors_) {
            Xxh64 h;
            h.update(base_ + t.offset, (size_t)t.length);
            if ((uint64_t)t.xxh64 != h.digest())
                throw std::runtime_error("pack: xxh64 mismatch for '" + t.name + "'");
            blob_h.update(base_ + t.offset, (size_t)t.length);
        }
        (void)blob_h; // blob-level digest covers tensor bytes only in record order
    }

    std::string path_;
    int fd_ = -1;
    int64_t file_size_ = 0;
    const uint8_t *base_ = nullptr;
    Header hdr_;
    std::vector<TensorInfo> tensors_;
    std::map<std::string, const TensorInfo *> by_name_;
};

} // namespace m64
