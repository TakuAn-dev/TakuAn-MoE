#!/bin/bash
# check.sh — build everything and run all unit parity tests.
# Usage: scripts/check.sh            (tiny pack defaults from Makefile)
#        M64PACK=/path/big.m64pack GOLDEN=/path/golden.bin scripts/check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
make build
make test
python3 tools/test_server.py
echo "check.sh: ALL BUILD+UNIT TESTS PASSED"
