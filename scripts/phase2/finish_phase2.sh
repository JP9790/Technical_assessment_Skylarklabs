#!/usr/bin/env bash
# Finish Phase 2 after audit + demo recording (skips long optional DDS build).
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
    echo "[WARN/FAIL] ${name}"
    FAIL=1
  fi
}

run_step "Stage A evaluation (3 configs x 10 trials)" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase2/evaluate_stage_a.py" $HEADLESS --device cuda --trials 10 --mode scripted --step-scale 0.35 $EXTRA

run_step "Replay-json sanity (3 trials)" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase2/evaluate_stage_a.py" $HEADLESS --device cuda --trials 3 --mode replay_json --replay-subsample 4 $EXTRA

python3 - <<'PY'
import json
import yaml
from pathlib import Path
from datetime import datetime, timezone

root = Path("/home/autonomique/jp_test")
audit = json.loads((root / "results/phase2/source_audit.json").read_text())
stage_a_path = root / "results/stage_A/stage_a_baseline.json"
stage_a = json.loads(stage_a_path.read_text()) if stage_a_path.exists() else {}
record = {}
rec_path = root / "results/phase2/recorded_demo.json"
if rec_path.exists():
    record = json.loads(rec_path.read_text())

ckpt = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "phase": 2,
    "status": "complete",
    "selected_source": audit.get("selected_source"),
    "stage_a_overall_success_rate": stage_a.get("overall_success_rate"),
    "stage_a_status": stage_a.get("stage_a_status"),
    "demo_recorded": rec_path.exists(),
    "demo_lift_success": record.get("lift_success"),
    "demo_path": record.get("data_json"),
    "next_phase": "Phase 3 — obtain BrainCo Revo2 Touch asset",
}
out = root / "results/phase2/phase2_checkpoint.yaml"
out.write_text(yaml.dump(ckpt, sort_keys=False))
print(f"Wrote {out}")
PY

echo "=== Phase 2 finish complete (failures=$FAIL) ==="
exit $FAIL
