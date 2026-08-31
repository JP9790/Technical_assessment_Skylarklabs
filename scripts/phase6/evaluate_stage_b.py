#!/usr/bin/env python3
"""Stage B evaluation — replay retargeted BrainCo demo on standalone hands (guide §8)."""

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
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase1"))
from isaac_bootstrap import patch_configclass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--demo",
    default=str(JP_TEST_ROOT / "checkpoints/stage_B/demos/pick_place_cylinder/episode_0001/data.json"),
)
parser.add_argument("--hand", choices=["right", "left", "both"], default="both")
parser.add_argument("--subsample", type=int, default=2)
parser.add_argument("--max-frame-delta", type=float, default=0.35)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

patch_configclass()

from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext

sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase3"))
from brainco_hand_scene import make_hand_scene_cfg

RESULTS = JP_TEST_ROOT / "results/stage_B"


def replay_hand(demo: dict, side: str, subsample: int, device: str) -> dict:
    cfg = yaml.safe_load((JP_TEST_ROOT / "configs/g1_brainco_target.yaml").read_text())
    actuated = cfg[f"{side}_hand"]["actuated_joint_names"]
    HandCfg = make_hand_scene_cfg(side)
    sim = SimulationContext(SimulationCfg(dt=0.005, device=device))
    scene = InteractiveScene(HandCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    names = list(robot.joint_names)
    idx = {n: i for i, n in enumerate(names)}

    max_track = 0.0
    max_cmd_delta = 0.0
    prev_q = None
    t0 = time.time()
    frames = demo["data"][:: max(1, subsample)]

    for item in frames:
        q = np.array(item["states"][f"{side}_ee"]["qpos"], dtype=np.float64)
        if prev_q is not None:
            max_cmd_delta = max(max_cmd_delta, float(np.max(np.abs(q - prev_q))))
        prev_q = q
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

    elapsed = time.time() - t0
    return {
        "side": side,
        "frames_played": len(frames),
        "max_joint_track_error": max_track,
        "max_cmd_delta": max_cmd_delta,
        "completion_time_s": elapsed,
        "playback_pass": max_track < 0.08 and max_cmd_delta < args.max_frame_delta,
    }


def main() -> int:
    demo_path = Path(args.demo)
    if not demo_path.is_file():
        raise FileNotFoundError(demo_path)
    demo = json.loads(demo_path.read_text())

    sides = ["right", "left"] if args.hand == "both" else [args.hand]
    results = {side: replay_hand(demo, side, args.subsample, args.device) for side in sides}
    overall = all(r["playback_pass"] for r in results.values())

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "B",
        "demo": str(demo_path),
        "subsample": args.subsample,
        "overall_success": overall,
        "stage_b_status": "pass" if overall else "fail",
    }
    for side, res in results.items():
        report[f"{side}_hand"] = res

    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.hand == "both":
        out = RESULTS / "stage_b_baseline.json"
    else:
        out = RESULTS / f"stage_b_{args.hand}.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    simulation_app.close()
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
