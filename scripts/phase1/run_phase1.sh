#!/usr/bin/env bash
# Run full Phase 1 pipeline: smoke test → interface extract → sanity test.
set -euo pipefail
JP_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${JP_TEST_ROOT}/scripts/run_isaac.sh"
HEADLESS="${1:---headless}"
EXTRA="${@:2}"
FAIL=0

run_step() {
  local name="$1"; shift
  echo "=== ${name} ==="
  if "$@"; then
    echo "[OK] ${name}"
  else
    echo "[FAIL] ${name}"
    FAIL=1
  fi
}

run_step "Phase 1.1: Smoke test" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase1/smoke_test_env.py" $HEADLESS --device cuda --enable_cameras $EXTRA

run_step "Phase 1.2: Extract robot interface" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase1/extract_robot_interface.py" $HEADLESS --device cuda --enable_cameras $EXTRA

run_step "Phase 1.3: Dex3 sanity tests" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase1/dex3_sanity_test.py" $HEADLESS --device cuda --enable_cameras --cycles 2 $EXTRA

echo "=== Phase 1 finished (failures=$FAIL) ==="
ls -la "${JP_TEST_ROOT}/results/phase1/" 2>/dev/null || true
exit $FAIL
