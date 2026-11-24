// json_min.hpp — minimal dependency-free JSON parser for .m64pack headers.
// Supports the canonical subset the packer emits: null/bool/number(int|float)/
// string(escaped basic)/array/object. Numbers > 2^53 are rejected as doubles.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>

namespace m64 {

struct JValue;
using JValuePtr = std::shared_ptr<JValue>;

struct JValue {
    enum Type { Null, Bool, Int, Double, Str, Arr, Obj } type = Null;
    bool b = false;
    int64_t i = 0;
    double d = 0;
    std::string s;
    std::vector<JValuePtr> arr;
    std::vector<std::pair<std::string, JValuePtr>> obj; // sorted (canonical)

    const JValue *find(const std::string &key) const {
        for (auto &kv : obj)
            if (kv.first == key) return kv.second.get();
        return nullptr;
    }
    const JValue &at(const std::string &key) const {
        const JValue *v = find(key);
        if (!v) throw std::runtime_error("json: missing key '" + key + "'");
        return *v;
    }
    int64_t as_int() const {
        if (type == Int) return i;
        if (type == Double) return (int64_t)d;
        throw std::runtime_error("json: not a number");
    }
    const std::string &as_str() const {
        if (type != Str) throw std::runtime_error("json: not a string");
        return s;
    }
};

class JsonParser {
public:
    static JValuePtr parse(const std::string &src) {
        JsonParser p(src);
        p.skip_ws();
        auto v = p.value();
        p.skip_ws();
        if (p.p_ != src.size()) throw std::runtime_error("json: trailing content");
        return v;
    }

private:
    explicit JsonParser(const std::string &s) : src_(s) {}
    const std::string &src_;
    size_t p_ = 0;

    [[noreturn]] void fail(const char *msg) {
        throw std::runtime_error(std::string("json: ") + msg + " at " + std::to_string(p_));
    }
    char peek() { return p_ < src_.size() ? src_[p_] : '\0'; }
    void skip_ws() {
        while (p_ < src_.size() && (src_[p_] == ' ' || src_[p_] == '\n' || src_[p_] == '\t' || src_[p_] == '\r'))
            p_++;
    }
    bool lit(const char *l) {
        size_t n = 0;
        while (l[n]) { if (src_.compare(p_, 1, l + n, 1) != 0) return false; n++; p_++; }
        return true;
    }
    JValuePtr value() {
        char c = peek();
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') return string();
        if (c == 't') { if (!lit("true")) fail("bad literal"); return mk(true); }
        if (c == 'f') { if (!lit("false")) fail("bad literal"); return mk(false); }
        if (c == 'n') { if (!lit("null")) fail("bad literal"); return JValuePtr(new JValue()); }
        return number();
    }
    static JValuePtr mk(bool b) { auto v = std::make_shared<JValue>(); v->type = JValue::Bool; v->b = b; return v; }
    JValuePtr number() {
        size_t start = p_;
        if (peek() == '-') p_++;
        bool is_dbl = false;
        while (p_ < src_.size()) {
            char c = src_[p_];
            if (c >= '0' && c <= '9') { p_++; continue; }
            if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') { is_dbl = true; p_++; continue; }
            break;
        }
        if (start == p_) fail("bad number");
        std::string tok = src_.substr(start, p_ - start);
        auto v = std::make_shared<JValue>();
        if (!is_dbl) {
            try { v->type = JValue::Int; v->i = std::stoll(tok); return v; }
            catch (...) { /* fall through to double */ }
        }
        v->type = JValue::Double;
        v->d = std::stod(tok);
        if (v->d == (double)(int64_t)v->d) { v->type = JValue::Int; v->i = (int64_t)v->d; }
        return v;
    }
    JValuePtr string() {
        if (peek() != '"') fail("expected string");
        p_++;
        auto v = std::make_shared<JValue>();
        v->type = JValue::Str;
        while (p_ < src_.size() && src_[p_] != '"') {
            char c = src_[p_++];
            if (c != '\\') { v->s += c; continue; }
            if (p_ >= src_.size()) fail("bad escape");
            char e = src_[p_++];
            switch (e) {
                case '"': v->s += '"'; break;
                case '\\': v->s += '\\'; break;
                case '/': v->s += '/'; break;
                case 'n': v->s += '\n'; break;
                case 't': v->s += '\t'; break;
                case 'r': v->s += '\r'; break;
                case 'b': v->s += '\b'; break;
                case 'f': v->s += '\f'; break;
                case 'u': {
                    unsigned cp = 0;
                    for (int k = 0; k < 4; k++) {
                        char h = src_[p_++];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= h - '0';
                        else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                        else fail("bad \\u");
                    }
                    if (cp < 0x80) v->s += (char)cp;
                    else if (cp < 0x800) { v->s += (char)(0xC0 | cp >> 6); v->s += (char)(0x80 | (cp & 0x3F)); }
                    else { v->s += (char)(0xE0 | cp >> 12); v->s += (char)(0x80 | ((cp >> 6) & 0x3F)); v->s += (char)(0x80 | (cp & 0x3F)); }
                    break;
                }
                default: fail("bad escape");
            }
        }
        if (peek() != '"') fail("unterminated string");
        p_++;
        return v;
    }
    JValuePtr array() {
        p_++; // '['
        auto v = std::make_shared<JValue>();
        v->type = JValue::Arr;
        skip_ws();
        if (peek() == ']') { p_++; return v; }
        while (true) {
            skip_ws();
            v->arr.push_back(value());
            skip_ws();
            char c = peek();
            if (c == ',') { p_++; continue; }
            if (c == ']') { p_++; return v; }
            fail("expected , or ]");
        }
    }
    JValuePtr object() {
        p_++; // '{'
        auto v = std::make_shared<JValue>();
        v->type = JValue::Obj;
        skip_ws();
        if (peek() == '}') { p_++; return v; }
        while (true) {
            skip_ws();
            auto k = string();
            skip_ws();
            if (peek() != ':') fail("expected :");
            p_++;
            skip_ws();
            v->obj.emplace_back(k->s, value());
            skip_ws();
            char c = peek();
            if (c == ',') { p_++; continue; }
            if (c == '}') { p_++; return v; }
            fail("expected , or }");
        }
    }
};

} // namespace m64
