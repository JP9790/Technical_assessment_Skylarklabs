#!/usr/bin/env bash
# Phase 7 — G1 + BrainCo wrist-coupled validation (overcomes USD-swap deferral).
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

[[ -f "${JP_TEST_ROOT}/results/phase5/phase5_checkpoint.yaml" ]] || {
  echo "Run phase5 first"; exit 1;
}

run_step "Phase 7.1: G1 arm replay + BrainCo wrist mount validation" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase7/evaluate_g1_brainco_coupling.py" $HEADLESS --device cuda --enable_cameras $EXTRA

python3 - <<'PY'
import json, yaml
from datetime import datetime, timezone
from pathlib import Path

root = Path("/home/autonomique/jp_test")
rep = json.loads((root / "results/phase7/g1_brainco_coupling.json").read_text())
ckpt = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "phase": 7,
    "status": "complete" if rep.get("phase7_status") == "pass" else "partial",
    "g1_brainco_coupling": str(root / "results/phase7/g1_brainco_coupling.json"),
    "phase7_status": rep.get("phase7_status"),
    "next_phase": "Stage C fine-tuning (optional)",
}
(root / "results/phase7/phase7_checkpoint.yaml").write_text(yaml.dump(ckpt, sort_keys=False))
print(f"Wrote {root / 'results/phase7/phase7_checkpoint.yaml'}")
PY

echo "=== Phase 7 finished (failures=$FAIL) ==="
exit $FAIL
