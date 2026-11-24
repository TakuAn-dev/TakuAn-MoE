# Mac64-MoE Engine — POSIX Makefile (Apple clang, zero external deps)
# Amendment A1: no brew/cmake/ninja/xcrun-metal. MSL is compiled at load
# time via newLibraryWithSource: and cached with MTLBinaryArchive.

CXX        := /usr/bin/clang++
CXXFLAGS   := -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -ffp-contract=off
OBJCFLAGS  := -std=c++20 -O2 -Wall -Wextra -Wno-unused-parameter -fobjc-arc -ffp-contract=off
INCLUDES   := -Iengine/src -Iengine/include
LDFLAGS    := -framework Metal -framework Foundation
BUILD      := build

.PHONY: all build test bench clean

all: build

build: $(BUILD)/test_dequant $(BUILD)/test_metal_parity $(BUILD)/test_p2a_kernels $(BUILD)/test_prefill_kernels $(BUILD)/test_arena $(BUILD)/test_prefix_disk $(BUILD)/m64run

# ---- pure-C++ unit test: bit-exact GGML dequant reference vs golden.bin ----
$(BUILD)/test_dequant: engine/tests/test_dequant.cpp engine/src/dequant.hpp \
        engine/src/xxh64.hpp engine/src/pack_reader.hpp engine/src/json_min.hpp | $(BUILD)/.stamp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

# ---- Metal unit parity: MSL kernels vs C++ reference ----
$(BUILD)/test_metal_parity: engine/tests/test_metal_parity.mm engine/src/metal_runtime.mm \
        engine/src/metal_runtime.hpp engine/src/dequant.hpp engine/src/pack_reader.hpp | $(BUILD)/.stamp
	$(CXX) $(OBJCFLAGS) $(INCLUDES) engine/tests/test_metal_parity.mm engine/src/metal_runtime.mm -o $@ $(LDFLAGS)

# ---- resident inference CLI ----
$(BUILD)/m64run: engine/src/main.cpp engine/src/metal_runtime.mm engine/src/forward_metal.mm engine/src/forward_metal.hpp \
        engine/src/expert_arena.mm engine/src/expert_arena.hpp \
        engine/src/forward.cpp engine/src/metal_runtime.hpp engine/src/forward.hpp | $(BUILD)/.stamp
	$(CXX) $(OBJCFLAGS) $(INCLUDES) -DM64_KERNEL_DIR=\"$(CURDIR)/engine/kernels\" -DM64_ARCHIVE_PATH=\"$(CURDIR)/build/m64_p2a.ar\" engine/src/main.cpp engine/src/forward.cpp engine/src/forward_metal.mm engine/src/expert_arena.mm engine/src/metal_runtime.mm -o $@ $(LDFLAGS)

$(BUILD)/.stamp:
	mkdir -p $(BUILD) && touch $@

# ---- Phase-2A forward-kernel parity (all m64_p2a.metal kernels, small buffers) ----
$(BUILD)/test_p2a_kernels: engine/tests/test_p2a_kernels.mm engine/src/metal_runtime.mm \
        engine/src/metal_runtime.hpp engine/kernels/m64_p2a_args.h | $(BUILD)/.stamp
	$(CXX) $(OBJCFLAGS) $(INCLUDES) engine/tests/test_p2a_kernels.mm engine/src/metal_runtime.mm -o $@ $(LDFLAGS)

# ---- Milestone-3A batched-prefill kernel parity (m64_prefill.metal) ----
$(BUILD)/test_prefill_kernels: engine/tests/test_prefill_kernels.mm engine/src/metal_runtime.mm \
        engine/src/metal_runtime.hpp engine/kernels/m64_p2a_args.h engine/kernels/m64_prefill.metal | $(BUILD)/.stamp
	$(CXX) $(OBJCFLAGS) $(INCLUDES) engine/tests/test_prefill_kernels.mm engine/src/metal_runtime.mm -o $@ $(LDFLAGS)

# ---- Phase-2B expert arena: LockedRegion / PreadPool / ExpertArena LRU + MTLIO ----
$(BUILD)/test_arena: engine/tests/test_arena.mm engine/src/expert_arena.mm engine/src/expert_arena.hpp \
        engine/src/metal_runtime.mm engine/src/metal_runtime.hpp | $(BUILD)/.stamp
	$(CXX) $(OBJCFLAGS) $(INCLUDES) engine/tests/test_arena.mm engine/src/expert_arena.mm engine/src/metal_runtime.mm -o $@ $(LDFLAGS)

# ---- SSD persistent prefix cache (.m64kv) format: pure-C++ unit test ----
$(BUILD)/test_prefix_disk: engine/tests/test_prefix_disk.cpp engine/src/prefix_disk.hpp \
        engine/src/xxh64.hpp | $(BUILD)/.stamp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

# Self-contained tiny fixture, generated on demand by tools/make_test_fixture.py
# (Python 3 stdlib only). M64PACK/GOLDEN=/path override to a real model.
FIXTURE       := $(BUILD)/fixture
M64PACK       ?= $(FIXTURE)/tiny.m64pack
GOLDEN        ?= $(FIXTURE)/golden.bin
FIXTURE_STAMP := $(FIXTURE)/.generated

$(FIXTURE_STAMP): tools/make_test_fixture.py
	python3 tools/make_test_fixture.py $(FIXTURE)
	touch $@

test: build $(FIXTURE_STAMP)
	$(BUILD)/test_dequant $(M64PACK) $(GOLDEN)
	$(BUILD)/test_metal_parity $(M64PACK)
	$(BUILD)/test_p2a_kernels
	$(BUILD)/test_prefill_kernels
	$(BUILD)/test_arena
	$(BUILD)/test_prefix_disk

bench: build
	$(BUILD)/m64run $(M64PACK) --bench

clean:
	rm -rf $(BUILD)
