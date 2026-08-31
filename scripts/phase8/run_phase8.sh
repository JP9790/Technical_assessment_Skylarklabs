#!/usr/bin/env bash
# Phase 8 / Stage C — residual adapter fine-tuning (assessment §1.5, guide §9).
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

[[ -f "${JP_TEST_ROOT}/results/phase5/ik_partial/right_q.npy" ]] || {
  echo "Missing IK partials — run phase5 IK retarget first"
  exit 1
}

run_step "Stage C.1: Build residual dataset (q_ik - q_joint)" \
  python3 "${JP_TEST_ROOT}/scripts/phase8/build_residual_dataset.py"

run_step "Stage C.2: Train residual adapter (PyTorch CPU/GPU)" \
  python3 "${JP_TEST_ROOT}/scripts/phase8/finetune_residual.py"

run_step "Stage C.3: Apply adapter → Stage C demo" \
  python3 "${JP_TEST_ROOT}/scripts/phase8/apply_residual_demo.py"

run_step "Stage C.4: Evaluate Stage C playback" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase8/evaluate_stage_c.py" $HEADLESS --device cuda --enable_cameras $EXTRA

python3 - <<'PY'
import json, yaml
from datetime import datetime, timezone
from pathlib import Path

root = Path("/home/autonomique/jp_test")
train = json.loads((root / "results/stage_C/finetune_train.json").read_text())
eval_ = json.loads((root / "results/stage_C/stage_c_baseline.json").read_text())
ckpt = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "phase": 8,
    "stage": "C",
    "status": "complete" if eval_.get("stage_c_status") == "pass" else "partial",
    "stage_c_status": eval_.get("stage_c_status"),
    "training": train,
    "evaluation": str(root / "results/stage_C/stage_c_baseline.json"),
    "demo": str(root / "checkpoints/stage_C/demos/pick_place_cylinder/episode_0001/data.json"),
}
(root / "results/stage_C/stage_c_checkpoint.yaml").write_text(yaml.dump(ckpt, sort_keys=False))
print(f"Wrote {root / 'results/stage_C/stage_c_checkpoint.yaml'}")
PY

echo "=== Stage C finished (failures=$FAIL) ==="
exit $FAIL
