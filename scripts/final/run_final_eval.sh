#!/usr/bin/env bash
# Phase 8 — Final evaluation: re-run critical GPU steps, verify all phases, produce A/B/C table (guide §10).
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

echo "========== Project 1 Final Evaluation Pipeline =========="

run_step "Phase 0: stack verification" \
  python3 "${JP_TEST_ROOT}/scripts/verify_phase0_stack.py"

run_step "Phase 2.3: Stage A eval (4 configs x 10 trials, guide §4.3 + §10)" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase2/evaluate_stage_a.py" $HEADLESS --device cuda --enable_cameras --trials 10 --mode scripted $EXTRA

run_step "Phase 5: retarget + validate (guide §7)" \
  bash "${JP_TEST_ROOT}/scripts/phase5/run_phase5.sh" $HEADLESS --device cuda --enable_cameras $EXTRA

run_step "Phase 6: Stage B playback (guide §8)" \
  bash "${JP_TEST_ROOT}/scripts/phase6/run_phase6.sh" $HEADLESS --device cuda --enable_cameras $EXTRA

run_step "Phase 7: G1+BrainCo coupling" \
  bash "${JP_TEST_ROOT}/scripts/phase7/run_phase7.sh" $HEADLESS --device cuda --enable_cameras $EXTRA

run_step "Verify phases 0–7 checkpoints" \
  python3 "${JP_TEST_ROOT}/scripts/final/verify_all_phases.py"

run_step "Phase 8: Final A/B/C results table (guide §10)" \
  python3 "${JP_TEST_ROOT}/scripts/final/evaluate_final_abc.py"

echo "========== Final evaluation finished (failures=$FAIL) =========="
echo "Results: ${JP_TEST_ROOT}/results/final/project1_abc_results.md"
exit $FAIL
