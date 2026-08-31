#!/usr/bin/env python3
"""Build Stage C training dataset: residual = q_ik_v2 - q_joint_v1."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(JP_TEST_ROOT / "scripts/phase5"))
from task_space_retarget import retarget_demo_frames

DEFAULT_DEMO = JP_TEST_ROOT / "checkpoints/stage_A/demos/pick_place_cylinder/episode_0001/data.json"
OUT = JP_TEST_ROOT / "results/stage_C/residual_dataset.npz"
CFG = JP_TEST_ROOT / "configs/finetune.yaml"


def main() -> int:
    cfg = yaml.safe_load(CFG.read_text())
    demo = json.loads(Path(cfg["training"]["source_demo"]).read_text())

    joint = retarget_demo_frames(demo, reg_weight=0.05)
    ik_r = np.load(JP_TEST_ROOT / cfg["training"]["teacher_right"])
    ik_l = np.load(JP_TEST_ROOT / cfg["training"]["teacher_left"])

    if ik_r.shape[0] != joint.brainco_right.shape[0]:
        raise ValueError("IK partial frame count mismatch")

    delta_r = ik_r - joint.brainco_right
    delta_l = ik_l - joint.brainco_left

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT,
        q_base_right=joint.brainco_right,
        q_base_left=joint.brainco_left,
        q_teacher_right=ik_r,
        q_teacher_left=ik_l,
        delta_right=delta_r,
        delta_left=delta_l,
    )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(OUT),
        "num_frames": int(joint.brainco_right.shape[0]),
        "mean_abs_delta_right": float(np.mean(np.abs(delta_r))),
        "mean_abs_delta_left": float(np.mean(np.abs(delta_l))),
    }
    (OUT.parent / "residual_dataset.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
