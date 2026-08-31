#!/usr/bin/env python3
"""Validate retargeted BrainCo demo — limits, smoothness, sim playback sample (§7.3)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEMO = JP_TEST_ROOT / "checkpoints/stage_B/demos/pick_place_cylinder/episode_0001/data.json"
OUT = JP_TEST_ROOT / "results/phase5" / "retarget_validation.json"

parser = argparse.ArgumentParser()
parser.add_argument("--demo", default=str(DEFAULT_DEMO))
parser.add_argument("--sim-sample-frames", type=int, default=5)
parser.add_argument("--max-frame-delta", type=float, default=0.35)
parser.add_argument("--sim", action="store_true")

simulation_app = None
if "--sim" in sys.argv:
    sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase1"))
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
else:
    args = parser.parse_args()


def validate_yaml(demo_path: Path) -> dict:
    demo = json.loads(demo_path.read_text())
    brainco = yaml.safe_load((JP_TEST_ROOT / "configs/g1_brainco_target.yaml").read_text())
    violations = 0
    frame_deltas: list[float] = []
    prev_r = prev_l = None

    for item in demo["data"]:
        rq = np.array(item["states"]["right_ee"]["qpos"], dtype=np.float64)
        lq = np.array(item["states"]["left_ee"]["qpos"], dtype=np.float64)
        for side, vec in (("right", rq), ("left", lq)):
            hand = brainco[f"{side}_hand"]
            for j, name in enumerate(hand["actuated_joint_names"]):
                lo = float(hand["joint_limits"][name]["lower"])
                hi = float(hand["joint_limits"][name]["upper"])
                if vec[j] < lo - 1e-4 or vec[j] > hi + 1e-4:
                    violations += 1
        if prev_r is not None:
            frame_deltas.append(float(max(np.max(np.abs(rq - prev_r)), np.max(np.abs(lq - prev_l)))))
        prev_r, prev_l = rq, lq

    max_delta = float(max(frame_deltas)) if frame_deltas else 0.0
    return {
        "num_frames": len(demo["data"]),
        "limit_violations": violations,
        "max_frame_delta": max_delta,
        "mean_frame_delta": float(np.mean(frame_deltas)) if frame_deltas else 0.0,
        "yaml_pass": violations == 0 and max_delta <= args.max_frame_delta,
    }


def validate_sim_sample(demo_path: Path, n: int, device: str) -> dict:
    sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase1"))
    sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase3"))
    from brainco_hand_scene import make_hand_scene_cfg
    from isaac_bootstrap import patch_configclass

    patch_configclass()
    import torch
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, SimulationContext

    demo = json.loads(demo_path.read_text())
    indices = np.linspace(0, len(demo["data"]) - 1, num=min(n, len(demo["data"])), dtype=int)
    actuated = yaml.safe_load((JP_TEST_ROOT / "configs/g1_brainco_target.yaml").read_text())[
        "right_hand"
    ]["actuated_joint_names"]

    HandCfg = make_hand_scene_cfg("right")
    sim = SimulationContext(SimulationCfg(dt=0.005, device=device))
    scene = InteractiveScene(HandCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    names = list(robot.joint_names)
    idx = {n: i for i, n in enumerate(names)}

    max_track_err = 0.0
    for fi in indices:
        q = np.array(demo["data"][int(fi)]["states"]["right_ee"]["qpos"], dtype=np.float64)
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
        max_track_err = max(max_track_err, err)

    return {
        "sim_sample_frames": int(len(indices)),
        "max_joint_track_error": max_track_err,
        "sim_pass": max_track_err < 0.15,
    }


def main() -> int:
    demo_path = Path(args.demo)
    if not demo_path.is_file():
        raise FileNotFoundError(demo_path)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "demo": str(demo_path),
        **validate_yaml(demo_path),
    }
    if args.sim:
        report["sim"] = validate_sim_sample(demo_path, args.sim_sample_frames, args.device)
        report["phase5_validation_pass"] = report["yaml_pass"] and report["sim"]["sim_pass"]
    else:
        report["phase5_validation_pass"] = report["yaml_pass"]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    if simulation_app is not None:
        simulation_app.close()
    return 0 if report["phase5_validation_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
