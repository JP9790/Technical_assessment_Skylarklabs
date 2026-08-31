#!/usr/bin/env python3
"""Verify Phase 0–7 checkpoints and gate criteria (implementation guide)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"


def _load_json(rel: str) -> dict:
    p = ROOT / rel
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def _load_yaml(rel: str) -> dict:
    p = ROOT / rel
    if not p.is_file():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def check_phase0() -> tuple[bool, str]:
    ver = _load_json("results/phase0/stack_verification.json")
    ok = ver.get("phase0_status") == "pass"
    return ok, "stack frozen + verification pass"


def check_phase1() -> tuple[bool, str]:
    ck = _load_yaml("results/phase1/phase1_checkpoint.yaml")
    sanity = _load_json("results/phase1/dex3_sanity_test.json")
    ok = ck.get("status") == "complete" and sanity.get("lift_success", False)
    return ok, "G1+Dex3 sanity + lift > 3cm"


def check_phase2() -> tuple[bool, str]:
    ck = _load_yaml("results/phase2/phase2_checkpoint.yaml")
    stage_a = _load_json("results/stage_A/stage_a_baseline.json")
    rate = float(stage_a.get("overall_success_rate", 0.0))
    ok = ck.get("status") == "complete" and rate >= 0.5
    return ok, f"Stage A success rate {rate*100:.0f}% (min 50%)"


def check_phase3() -> tuple[bool, str]:
    ck = _load_yaml("results/phase3/phase3_checkpoint.yaml")
    right = _load_json("results/phase3/brainco_right_hand_test.json")
    left = _load_json("results/phase3/brainco_left_hand_test.json")
    ok = (
        ck.get("status") == "complete"
        and right.get("phase3_hand_status") == "pass"
        and left.get("phase3_hand_status") == "pass"
    )
    return ok, "BrainCo both hands + mount smoke"


def check_phase4() -> tuple[bool, str]:
    ck = _load_yaml("results/phase4/phase4_checkpoint.yaml")
    ver = _load_json("results/phase4/correspondence_verification.json")
    ok = ck.get("status") == "complete" and ver.get("yaml_verification_pass", False)
    return ok, "correspondence YAML verified"


def check_phase5() -> tuple[bool, str]:
    ck = _load_yaml("results/phase5/phase5_checkpoint.yaml")
    offline = _load_json("results/phase5/retarget_offline.json")
    valid = _load_json("results/phase5/retarget_validation.json")
    ok = (
        ck.get("status") == "complete"
        and offline.get("phase5_retarget_pass", False)
        and valid.get("phase5_validation_pass", False)
    )
    method = offline.get("method", "?")
    return ok, f"retarget ({method}) + validation pass"


def check_phase6() -> tuple[bool, str]:
    ck = _load_yaml("results/phase6/phase6_checkpoint.yaml")
    stage_b = _load_json("results/stage_B/stage_b_baseline.json")
    ok = ck.get("status") == "complete" and stage_b.get("stage_b_status") == "pass"
    return ok, "Stage B playback pass"


def check_phase7() -> tuple[bool, str]:
    ck = _load_yaml("results/phase7/phase7_checkpoint.yaml")
    if not ck:
        rep = _load_json("results/phase7/g1_brainco_coupling.json")
        ok = rep.get("phase7_status") == "pass"
        return ok, "G1+BrainCo coupling (no checkpoint yet)"
    ok = ck.get("status") in ("complete", "partial") and ck.get("phase7_status") == "pass"
    return ok, "G1 wrist + BrainCo mount coupling"


def check_phase8() -> tuple[bool, str]:
    ck = _load_yaml("results/stage_C/stage_c_checkpoint.yaml")
    ev = _load_json("results/stage_C/stage_c_baseline.json")
    ok = ck.get("status") in ("complete", "partial") and ev.get("stage_c_status") in ("pass", "partial")
    return ok, f"Stage C residual adapter ({ev.get('stage_c_status', '?')})"


PHASE_CHECKS = [
    (0, check_phase0),
    (1, check_phase1),
    (2, check_phase2),
    (3, check_phase3),
    (4, check_phase4),
    (5, check_phase5),
    (6, check_phase6),
    (7, check_phase7),
    (8, check_phase8),
]


def main() -> int:
    print("=== Phase verification (guide milestones) ===")
    all_ok = True
    rows = []
    for phase, fn in PHASE_CHECKS:
        ok, detail = fn()
        status = "PASS" if ok else "FAIL"
        print(f"  Phase {phase}: {status} — {detail}")
        rows.append({"phase": phase, "pass": ok, "detail": detail})
        all_ok = all_ok and ok

    out = RESULTS / "final" / "phase_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"all_phases_pass": all_ok, "phases": rows}, indent=2))
    print(f"\nWrote {out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
