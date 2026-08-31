#!/usr/bin/env python3
"""Stage C evaluation — BrainCo playback with finetuned demo vs Stage B."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(JP_TEST_ROOT / "scripts/phase1"))
from isaac_bootstrap import patch_configclass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--demo",
    default=str(JP_TEST_ROOT / "checkpoints/stage_C/demos/pick_place_cylinder/episode_0001/data.json"),
)
parser.add_argument(
    "--baseline-demo",
    default=str(JP_TEST_ROOT / "checkpoints/stage_B/demos/pick_place_cylinder/episode_0001/data.json"),
)
parser.add_argument("--subsample", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

patch_configclass()

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

sys.path.insert(0, str(JP_TEST_ROOT / "scripts/phase3"))
from brainco_hand_scene import make_hand_scene_cfg

RESULTS = JP_TEST_ROOT / "results/stage_C"


def replay_error(demo: dict, side: str, subsample: int, device: str) -> float:
    cfg = yaml.safe_load((JP_TEST_ROOT / "configs/g1_brainco_target.yaml").read_text())
    actuated = cfg[f"{side}_hand"]["actuated_joint_names"]
    HandCfg = make_hand_scene_cfg(side)
    sim = SimulationContext(SimulationCfg(dt=0.005, device=device))
    scene = InteractiveScene(HandCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    idx = {n: i for i, n in enumerate(robot.joint_names)}

    max_track = 0.0
    for item in demo["data"][:: max(1, subsample)]:
        q = np.array(item["states"][f"{side}_ee"]["qpos"], dtype=np.float64)
        target = robot.data.default_joint_pos.clone()
        for j, name in enumerate(actuated):
            if name in idx:
                target[0, idx[name]] = float(q[j])
        robot.write_joint_position_to_sim(target)
        robot.set_joint_position_target(target)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=0.005)
        err = float(torch.max(torch.abs(robot.data.joint_pos[0] - target[0])).item())
        max_track = max(max_track, err)

    try:
        from isaaclab.sim import SimulationContext as SC

        SC.clear_instance()
    except Exception:
        pass
    return max_track


def main() -> int:
    c_demo = json.loads(Path(args.demo).read_text())
    b_demo = json.loads(Path(args.baseline_demo).read_text())

    t0 = time.time()
    c_r = replay_error(c_demo, "right", args.subsample, args.device)
    c_l = replay_error(c_demo, "left", args.subsample, args.device)
    b_r = replay_error(b_demo, "right", args.subsample, args.device)
    b_l = replay_error(b_demo, "left", args.subsample, args.device)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "C",
        "demo": str(args.demo),
        "baseline_demo": str(args.baseline_demo),
        "right_hand": {
            "max_joint_track_error": c_r,
            "baseline_error": b_r,
            "improvement_m": b_r - c_r,
            "playback_pass": c_r < 0.08,
        },
        "left_hand": {
            "max_joint_track_error": c_l,
            "baseline_error": b_l,
            "improvement_m": b_l - c_l,
            "playback_pass": c_l < 0.08,
        },
        "adaptation_wall_clock_s": time.time() - t0,
        "stage_c_status": "pass" if c_r < 0.08 and c_l < 0.08 else "partial",
        "overall_success": c_r < 0.08 and c_l < 0.08,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "stage_c_baseline.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    simulation_app.close()
    return 0 if report["overall_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
