#!/usr/bin/env bash
# Phase 6 — Stage B BrainCo playback evaluation (guide §8).
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

p5="${JP_TEST_ROOT}/results/phase5/phase5_checkpoint.yaml"
if [[ ! -f "$p5" ]]; then
  echo "Phase 5 checkpoint missing — run scripts/phase5/run_phase5.sh first"
  exit 1
fi

run_step "Phase 6.1a: Stage B right-hand playback" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase6/evaluate_stage_b.py" --hand right $HEADLESS --device cuda --enable_cameras $EXTRA

run_step "Phase 6.1b: Stage B left-hand playback" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase6/evaluate_stage_b.py" --hand left $HEADLESS --device cuda --enable_cameras $EXTRA

python3 - <<'PY'
import json, yaml
from datetime import datetime, timezone
from pathlib import Path

root = Path("/home/autonomique/jp_test")
right = json.loads((root / "results/stage_B/stage_b_right.json").read_text())
left = json.loads((root / "results/stage_B/stage_b_left.json").read_text())
overall = right["right_hand"]["playback_pass"] and left["left_hand"]["playback_pass"]
merged = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "B",
    "demo": right.get("demo"),
    "subsample": right.get("subsample"),
    "right_hand": right["right_hand"],
    "left_hand": left["left_hand"],
    "overall_success": overall,
    "stage_b_status": "pass" if overall else "fail",
}
(root / "results/stage_B/stage_b_baseline.json").write_text(json.dumps(merged, indent=2))

p5 = yaml.safe_load((root / "results/phase5/phase5_checkpoint.yaml").read_text())
ckpt = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "phase": 6,
    "status": "complete" if merged["stage_b_status"] == "pass" else "partial",
    "guide_section": "8 - Stage B retargeting-only playback",
    "stage_b_status": merged["stage_b_status"],
    "demo": merged.get("demo"),
    "prerequisites": {"phase5_status": p5.get("status")},
    "next_phase": "Phase 7 — Stage C simulation fine-tuning (optional)",
}
out = root / "results/phase6/phase6_checkpoint.yaml"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.dump(ckpt, sort_keys=False))
print(f"Wrote {out}")
PY

echo "=== Phase 6 finished (failures=$FAIL) ==="
exit $FAIL
