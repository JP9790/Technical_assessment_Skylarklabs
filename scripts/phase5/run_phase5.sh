#!/usr/bin/env bash
# Phase 5 — task-space retargeting + offline validation (guide §7).
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

# Prerequisites
for ck in phase0/stack_checkpoint.yaml phase1/phase1_checkpoint.yaml phase2/phase2_checkpoint.yaml phase3/phase3_checkpoint.yaml phase4/phase4_checkpoint.yaml; do
  if [[ ! -f "${JP_TEST_ROOT}/results/${ck}" ]]; then
    echo "Missing prerequisite checkpoint: results/${ck}"
    exit 1
  fi
done

run_step "Phase 5.1: Build Dex3 fingertip FK cache" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase5/build_dex3_tip_cache.py" $HEADLESS --device cuda --enable_cameras $EXTRA

run_step "Phase 5.2a: Task-space IK retarget — right hand" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase5/retarget_side_ik.py" --side right $HEADLESS --device cuda $EXTRA

run_step "Phase 5.2b: Task-space IK retarget — left hand" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase5/retarget_side_ik.py" --side left $HEADLESS --device cuda $EXTRA

run_step "Phase 5.2c: Merge IK partials → Stage B demo" \
  python3 "${JP_TEST_ROOT}/scripts/phase5/retarget_demo.py" --method ik --merge-only || {
    echo "[WARN] IK merge failed — falling back to joint-map v1"
    python3 "${JP_TEST_ROOT}/scripts/phase5/retarget_demo.py" --method joint
  }

run_step "Phase 5.3: YAML validation (limits + smoothness)" \
  python3 "${JP_TEST_ROOT}/scripts/phase5/validate_retarget.py"

run_step "Phase 5.4: Sim sample playback validation" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase5/validate_retarget.py" --sim $HEADLESS --device cuda --enable_cameras $EXTRA

python3 - <<'PY'
import json, yaml
from datetime import datetime, timezone
from pathlib import Path

root = Path("/home/autonomique/jp_test")
offline = json.loads((root / "results/phase5/retarget_offline.json").read_text())
valid = json.loads((root / "results/phase5/retarget_validation.json").read_text())
p2 = yaml.safe_load((root / "results/phase2/phase2_checkpoint.yaml").read_text())
p4 = yaml.safe_load((root / "results/phase4/phase4_checkpoint.yaml").read_text())

ckpt = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "phase": 5,
    "status": "complete" if valid.get("phase5_validation_pass") and offline.get("phase5_retarget_pass") else "partial",
    "guide_section": "7 - Task-space retargeting optimizer",
    "retarget_method": offline.get("method"),
    "source_demo": offline.get("source_demo"),
    "output_demo": offline.get("output_demo"),
    "offline_retarget_pass": offline.get("phase5_retarget_pass"),
    "validation_pass": valid.get("phase5_validation_pass"),
    "prerequisites": {"phase2_status": p2.get("status"), "phase4_status": p4.get("status")},
    "next_phase": "Phase 6 — Stage B BrainCo playback evaluation",
}
out = root / "results/phase5/phase5_checkpoint.yaml"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.dump(ckpt, sort_keys=False))
print(f"Wrote {out}")
PY

echo "=== Phase 5 finished (failures=$FAIL) ==="
exit $FAIL
