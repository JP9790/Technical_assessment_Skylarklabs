#!/usr/bin/env python3
"""Phase 7 — G1 arm replay + BrainCo wrist-coupled hand validation."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase1"))
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase2"))
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase3"))

from demo_io import load_episode
from isaac_bootstrap import import_unitree_tasks, patch_configclass
from robot_control import RIGHT_HAND_OPEN, apply_hand, build_target, step_hold

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--demo",
    default=str(JP_TEST_ROOT / "checkpoints/stage_B/demos/pick_place_cylinder/episode_0001/data.json"),
)
parser.add_argument("--subsample", type=int, default=4)
parser.add_argument("--ik-samples", type=int, default=8)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()
patch_configclass()

import gymnasium as gym
from brainco_hand_scene import make_hand_scene_cfg
from env_utils import prepare_env_cfg
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"
RESULTS = JP_TEST_ROOT / "results" / "phase7"


def replay_g1_arms(demo: dict, subsample: int, device: str) -> list[dict]:
    env = gym.make(TASK, cfg=prepare_env_cfg(parse_env_cfg(TASK, device=device, num_envs=1))).unwrapped
    env.reset()
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    body_names = list(robot.data.body_names)
    wrist_idx = body_names.index("right_wrist_yaw_link")

    snaps = []
    open_pose = apply_hand(build_target(env), env, RIGHT_HAND_OPEN)
    for item in demo["data"][:: max(1, subsample)]:
        target = open_pose.clone()
        for block in ("left_arm", "right_arm"):
            for j, n in enumerate(demo["info"]["joint_names"][block]):
                if n in idx:
                    target[idx[n]] = float(item["states"][block]["qpos"][j])
        step_hold(env, target, 1)
        pos = robot.data.body_pos_w[0, wrist_idx]
        quat = robot.data.body_quat_w[0, wrist_idx]
        snaps.append(
            {
                "wrist_pos": [float(x) for x in pos.tolist()],
                "wrist_quat_wxyz": [float(x) for x in quat.tolist()],
                "brainco_q": [float(x) for x in item["states"]["right_ee"]["qpos"]],
            }
        )
    env.close()
    return snaps


def validate_mount_samples(snaps: list[dict], n: int, device: str) -> dict:
    import yaml

    indices = np.linspace(0, len(snaps) - 1, num=min(n, len(snaps)), dtype=int)
    actuated = yaml.safe_load((JP_TEST_ROOT / "configs/g1_brainco_target.yaml").read_text())[
        "right_hand"
    ]["actuated_joint_names"]

    HandCfg = make_hand_scene_cfg("right")
    sim = SimulationContext(SimulationCfg(dt=0.005, device=device))
    scene = InteractiveScene(HandCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    robot = scene["robot"]
    idx = {n: i for i, n in enumerate(robot.joint_names)}

    max_err = 0.0
    for fi in indices:
        snap = snaps[int(fi)]
        pos = snap["wrist_pos"]
        quat = snap.get("wrist_quat_wxyz", [1.0, 0.0, 0.0, 0.0])
        root = torch.tensor([[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]], device=robot.device)
        robot.write_root_pose_to_sim(root)
        q = np.array(snap["brainco_q"], dtype=np.float64)
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
        max_err = max(max_err, err)

    return {"samples": int(len(indices)), "max_joint_track_error": max_err, "mount_pass": max_err < 0.08}


def main() -> int:
    demo = load_episode(Path(args.demo))
    snaps = replay_g1_arms(demo, args.subsample, args.device)
    mount = validate_mount_samples(snaps, args.ik_samples, args.device)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "demo": str(args.demo),
        "wrist_frames_captured": len(snaps),
        "brainco_mount_validation": mount,
        "phase7_status": "pass" if mount["mount_pass"] and len(snaps) > 0 else "fail",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "g1_brainco_coupling.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))

    simulation_app.close()
    return 0 if report["phase7_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
