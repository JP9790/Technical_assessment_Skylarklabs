#!/usr/bin/env bash
# Phase 3: BrainCo Revo2 bring-up — parse URDF, test hands in Isaac Sim.
set -euo pipefail

JP_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${JP_TEST_ROOT}/scripts/run_isaac.sh"
HEADLESS="${1:---headless}"
EXTRA="${@:2}"
FAIL=0

echo "=== Phase 3.0: Setup BrainCo asset symlink ==="
mkdir -p "${JP_TEST_ROOT}/assets/brainco"
ln -sfn ../../external/brainco-description/revo2_system "${JP_TEST_ROOT}/assets/brainco/revo2_system"

echo "=== Phase 3.1: Parse BrainCo URDF interface ==="
python3 "${JP_TEST_ROOT}/scripts/phase3/parse_brainco_urdf.py"

run_step() {
  local name="$1"; shift
  echo "=== ${name} ==="
  if "$@"; then
    echo "[OK] ${name}"
  else
    echo "[WARN] ${name}"
    FAIL=1
  fi
}

run_step "Phase 3.2: Right hand Isaac test" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase3/test_brainco_hand.py" $HEADLESS --device cuda --hand right $EXTRA

run_step "Phase 3.3: Left hand Isaac test" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase3/test_brainco_hand.py" $HEADLESS --device cuda --hand left $EXTRA

run_step "Phase 3.4: G1 wrist + BrainCo mount smoke (§5.4)" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase3/test_g1_brainco_attach.py" $HEADLESS --device cuda --enable_cameras $EXTRA

python3 "${JP_TEST_ROOT}/scripts/phase3/update_dex3_to_brainco_config.py"

python3 - <<'PY'
import json, yaml
from pathlib import Path
from datetime import datetime, timezone

root = Path("/home/autonomique/jp_test")
right = json.loads((root / "results/phase3/brainco_right_hand_test.json").read_text())
left = json.loads((root / "results/phase3/brainco_left_hand_test.json").read_text())
attach_path = root / "results/phase3/g1_brainco_attachment_smoke.json"
attach = json.loads(attach_path.read_text()) if attach_path.exists() else {}

ckpt = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "phase": 3,
    "status": "complete",
    "target_asset": "BrainCo Revo2 (brainco-description revo2_system)",
    "right_hand_status": right.get("phase3_hand_status"),
    "left_hand_status": left.get("phase3_hand_status"),
    "right_joints": right.get("num_joints"),
    "left_joints": left.get("num_joints"),
    "guide_5_3": right.get("guide_5_3_checklist", {}),
    "guide_5_4_attachment": attach.get("attachment_status", "not_run"),
    "config": "configs/g1_brainco_target.yaml",
    "retargeting_config": "configs/dex3_to_brainco.yaml",
    "g1_attachment_smoke": str(attach_path) if attach_path.exists() else None,
    "next_phase": "Phase 4 — semantic Dex3-to-BrainCo retargeting",
}
out = root / "results/phase3/phase3_checkpoint.yaml"
out.write_text(yaml.dump(ckpt, sort_keys=False))
print(f"Wrote {out}")
PY

echo "=== Phase 3 finished (failures=$FAIL) ==="
exit $FAIL
