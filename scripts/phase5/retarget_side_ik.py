#!/usr/bin/env python3
"""Retarget one hand side via task-space IK in an isolated Isaac Sim session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(JP_TEST_ROOT / "scripts/phase5"))

from task_space_ik_retarget import CACHE_PATH, _retarget_side_frames

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--side", choices=["right", "left"], required=True)
parser.add_argument(
    "--demo",
    default=str(JP_TEST_ROOT / "checkpoints/stage_A/demos/pick_place_cylinder/episode_0001/data.json"),
)
parser.add_argument("--subsample", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import yaml


def main() -> int:
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"Missing tip cache: {CACHE_PATH}")

    cache = json.loads(CACHE_PATH.read_text())
    demo = json.loads(Path(args.demo).read_text())
    brainco_cfg = yaml.safe_load((JP_TEST_ROOT / "configs/g1_brainco_target.yaml").read_text())
    q, violations = _retarget_side_frames(
        cache["frames"],
        args.side,
        args.device,
        brainco_cfg,
        demo=demo,
        subsample=args.subsample,
    )

    out_dir = JP_TEST_ROOT / "results" / "phase5" / "ik_partial"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.side}_q.npy", q)
    meta = {
        "side": args.side,
        "num_frames": int(q.shape[0]),
        "limit_violations": int(violations),
        "subsample": args.subsample,
    }
    (out_dir / f"{args.side}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
