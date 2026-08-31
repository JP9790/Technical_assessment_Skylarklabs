#!/usr/bin/env bash
# Phase 1 §3.4 Dex3 scripted sanity (requires lift > 3cm).
set -euo pipefail
JP_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${JP_TEST_ROOT}/scripts/run_isaac.sh"
exec "$RUN" "${JP_TEST_ROOT}/scripts/phase1/dex3_sanity_test.py" --headless --device cuda --enable_cameras --cycles 1 "$@"
