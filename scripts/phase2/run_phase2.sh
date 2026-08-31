#!/usr/bin/env bash
# Phase 2: source selection audit → record demo → Stage A evaluation.
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

echo "=== Phase 2.0: Source audit ==="
python3 "${JP_TEST_ROOT}/scripts/phase2/audit_sources.py"

echo "=== Phase 2.1: DDS deps (optional) ==="
bash "${JP_TEST_ROOT}/scripts/phase2/setup_dds_deps.sh" || true

run_step "Phase 2.2: Record Stage A demo" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase2/record_demo.py" $HEADLESS --device cuda --enable_cameras $EXTRA

run_step "Phase 2.3: Stage A evaluation (3 configs x 10 trials)" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase2/evaluate_stage_a.py" $HEADLESS --device cuda --enable_cameras --trials 10 --mode scripted --step-scale 1.0 $EXTRA

run_step "Phase 2.4: Replay-json sanity" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase2/evaluate_stage_a.py" $HEADLESS --device cuda --enable_cameras --trials 3 --mode replay_json --replay-subsample 2 --step-scale 1.0 $EXTRA

python3 - <<'PY'
import json
import yaml
from pathlib import Path
from datetime import datetime, timezone

root = Path("/home/autonomique/jp_test")
audit = json.loads((root / "results/phase2/source_audit.json").read_text())
stage_a = json.loads((root / "results/stage_A/stage_a_baseline.json").read_text())
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
    "demo_recorded": rec_path.exists(),
    "demo_lift_success": record.get("lift_success"),
    "next_phase": "Phase 3 — obtain BrainCo Revo2 Touch asset",
}
out = root / "results/phase2/phase2_checkpoint.yaml"
out.write_text(yaml.dump(ckpt, sort_keys=False))
print(f"Wrote {out}")
PY

echo "=== Phase 2 finished (failures=$FAIL) ==="
ls -la "${JP_TEST_ROOT}/results/phase2/" 2>/dev/null || true
ls -la "${JP_TEST_ROOT}/results/stage_A/" 2>/dev/null || true
exit $FAIL
