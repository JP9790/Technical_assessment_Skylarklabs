#!/usr/bin/env bash
# Phase 4 — define and verify Dex3→BrainCo semantic correspondence (guide §6).
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

echo "=== Phase 4.1: Sync retargeting config from correspondence table ==="
python3 "${JP_TEST_ROOT}/scripts/phase4/sync_retargeting_config.py"

run_step "Phase 4.2: YAML correspondence verification" \
  python3 "${JP_TEST_ROOT}/scripts/phase4/verify_correspondence.py"

run_step "Phase 4.3: Sim frame verification (G1 Dex3 + BrainCo USD)" \
  "$RUN" "${JP_TEST_ROOT}/scripts/phase4/verify_correspondence.py" --sim $HEADLESS --device cuda --enable_cameras $EXTRA

python3 - <<'PY'
import json, yaml
from datetime import datetime, timezone
from pathlib import Path

root = Path("/home/autonomique/jp_test")
ver = json.loads((root / "results/phase4/correspondence_verification.json").read_text())
p2 = yaml.safe_load((root / "results/phase2/phase2_checkpoint.yaml").read_text())
p3 = yaml.safe_load((root / "results/phase3/phase3_checkpoint.yaml").read_text())

ckpt = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "phase": 4,
    "status": "complete" if ver.get("phase4_status") == "pass" else "partial",
    "guide_section": "6 - Define Dex3-to-BrainCo Correspondence",
    "correspondence_table": "configs/correspondence_table.yaml",
    "retargeting_config": "configs/dex3_to_brainco.yaml",
    "verification": str(root / "results/phase4/correspondence_verification.json"),
    "yaml_verification_pass": ver.get("yaml_verification_pass"),
    "sim_verification_pass": ver.get("sim_verification_pass"),
    "prerequisites": {
        "phase2_status": p2.get("status"),
        "phase3_status": p3.get("status"),
    },
    "next_phase": "Phase 5 — implement task-space retargeting optimizer + offline trajectory validation",
}
out = root / "results/phase4/phase4_checkpoint.yaml"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.dump(ckpt, sort_keys=False))
print(f"Wrote {out}")
PY

echo "=== Phase 4 finished (failures=$FAIL) ==="
exit $FAIL
