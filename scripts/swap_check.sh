#!/bin/bash
# swap_check.sh — run the resident engine on a full pack and assert
# macOS swap usage does NOT grow during the run (Phase 1 DoD: swap delta == 0).
# Usage: scripts/swap_check.sh <PACK> [extra m64run args...]
#        M64_PACK=/path/big.m64pack M64_IDS=/path/ids.txt scripts/swap_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PACK="${1:-${M64_PACK:-}}"
IDS="${M64_IDS:-ids.txt}"
if [[ -z "$PACK" || ! -f "$IDS" ]]; then
    echo "usage: scripts/swap_check.sh <pack.m64pack> [m64run args...]   (or M64_PACK=...)" >&2
    echo "       IDS file must exist: give M64_IDS=<path> (default: ./ids.txt)" >&2
    exit 1
fi
if [[ $# -gt 0 ]]; then shift; fi

swap_used_mb() {
    # vm.swapusage e.g.: vm.swapusage: total = 5120.00M  used = 2678.69M  free = ...
    sysctl -n vm.swapusage | sed -E 's/.*used = ([0-9.]+)M.*/\1/'
}

B=$(swap_used_mb)
echo "swap used before: ${B} MB"
./build/m64run "$PACK" --ids-file "$IDS" "$@"
A=$(swap_used_mb)
echo "swap used after:  ${A} MB"
DELTA=$(echo "$A - $B" | bc -l)
echo "swap delta: ${DELTA} MB"
# assert swap did not grow (delta < 1 MB; negative delta means swap shrank)
if awk "BEGIN{exit !($DELTA < 1.0)}"; then
    echo "SWAP CHECK PASS (delta ${DELTA} MB < 1 MB)"
else
    echo "SWAP CHECK FAIL (delta ${DELTA} MB)"
    exit 1
fi
