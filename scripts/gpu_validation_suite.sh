#!/bin/bash
# gpu_validation_suite.sh — Phase 2A Metal GPU validation, splasH-SAFE edition.
#
# WHY THIS EXISTS: the full-pack Metal run wraps a 26.5 GB NoCopy buffer; with
# Splash's ~43-45 GiB Metal pool wired, physical RAM (64 GB) overflows and the
# GPU faults every step with MTLCommandBuffer status=5 (Insufficient Memory).
# This suite REFUSES to run while splash is alive, then runs the complete
# Phase 2A Definition-of-Done sequence and logs everything.
#
# Run it from your own terminal AFTER stopping splash:
#   1. stop splash (e.g. Ctrl-C in its terminal)
#   2. bash scripts/gpu_validation_suite.sh /path/to/pack.m64pack
#   3. restart splash
#
# Paths come from env vars (preferred) or positional args — no hardcoded
# model/temp locations. Required:
#   PACK        $M64_PACK or $1        — the .m64pack to validate
#   IDS         $M64_IDS  (default ids.txt)
#   LOGD        $M64_LOG_DIR (default build/gpu_validation)
#   REF_LOGITS  $M64_REF_LOGITS (default ref_smoke_logits.bin)
#   REF_TOKENS  $M64_REF_TOKENS (default ref_smoke_tokens.bin)
set -uo pipefail
cd "$(dirname "$0")/.."
PACK="${M64_PACK:-${1:-}}"
IDS="${M64_IDS:-ids.txt}"
LOGD="${M64_LOG_DIR:-build/gpu_validation}"
REF_LOGITS="${M64_REF_LOGITS:-ref_smoke_logits.bin}"
REF_TOKENS="${M64_REF_TOKENS:-ref_smoke_tokens.bin}"

usage() {
    echo "usage: $0 <pack.m64pack>   (or M64_PACK=/path/to/pack.m64pack $0)"
    echo " env: M64_IDS=$IDS (token-id file)"
    echo "      M64_LOG_DIR=$LOGD"
    echo "      M64_REF_LOGITS=$REF_LOGITS  M64_REF_TOKENS=$REF_TOKENS (llama.cpp reference for --compare)"
    exit 1
}

if [ -z "$PACK" ]; then
    echo "ERROR: no pack given."
    usage
fi
for f in "$IDS" "$REF_LOGITS" "$REF_TOKENS"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required file not found: $f"
        usage
    fi
done
mkdir -p "$LOGD"
FAILED=0

if pgrep -f 'splash.*/engine/splash' >/dev/null; then
    echo "ERROR: splash is still running — the Metal suite would fault with"
    echo "       'Insufficient Memory'. Stop splash first, then re-run this."
    exit 3
fi
WIRED_GB=$(( $(vm_stat | awk '/wired/{gsub("\\.","",$4); print $4}') * 16384 / 1073741824 ))
echo "pre-flight: wired=${WIRED_GB} GiB swap=$(sysctl -n vm.swapusage)"
if [ "$WIRED_GB" -gt 30 ]; then
    echo "WARNING: wired memory ${WIRED_GB} GiB > 30 GiB — something big still holds GPU pool."
fi

step() { # step <name> <logfile> <cmd...>
    local name="$1" log="$2"; shift 2
    echo "=== [$name] $(date '+%H:%M:%S') $* -> $log"
    { echo "date: $(date -Iseconds)"; echo "cmd: $*"; "$@"; } >"$log" 2>&1
    local rc=$?
    echo "=== [$name] exit=$rc"
    [ $rc -ne 0 ] && FAILED=1
    return $rc
}

step build_check "$LOGD/1_check.log" ./scripts/check.sh
step parity8     "$LOGD/2_parity8.log" ./build/m64run "$PACK" --backend metal \
    --ids-file "$IDS" --compare "$REF_LOGITS" \
    "$REF_TOKENS" --steps 8
step bench32     "$LOGD/3_bench32.log" ./build/m64run "$PACK" --backend metal \
    --ids-file "$IDS" --steps 32 --bench
step swapcheck   "$LOGD/4_swap.log" ./scripts/swap_check.sh "$PACK" --backend metal --steps 8 --bench

echo
echo "================ PHASE 2A SUITE SUMMARY ================"
grep -hE "argmax agreement|cosine|tok/s|SWAP CHECK|ALL .* PASSED|GPU fault|status=" "$LOGD"/[1234]*.log | head -40
echo "logs: $LOGD/"
if [ $FAILED -eq 0 ]; then echo "SUITE: ALL PASS"; else
    echo "SUITE: FAILURES PRESENT — tail the logs above before re-running."; fi
exit $FAILED
