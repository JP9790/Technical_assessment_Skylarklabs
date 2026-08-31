#!/usr/bin/env python3
"""Log palm vs object positions for candidate arm poses."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from action_utils import action_from_target
from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()

import gymnasium as gym
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"


def snapshot(env, label: str, target: torch.Tensor, steps: int = 120) -> dict:
    action = action_from_target(env, target)
    for _ in range(steps):
        env.step(action)
    robot = env.scene["robot"]
    obj = env.scene["object"].data.root_pos_w[0]
    names = list(robot.data.body_names)
    palm = robot.data.body_pos_w[0, names.index("right_hand_palm_link")]
    wrist = robot.data.body_pos_w[0, names.index("right_wrist_yaw_link")]
    return {
        "label": label,
        "object": [float(x) for x in obj.tolist()],
        "palm": [float(x) for x in palm.tolist()],
        "wrist": [float(x) for x in wrist.tolist()],
        "palm_dist": float(torch.linalg.norm(palm - obj).item()),
        "object_z": float(obj[2].item()),
    }


def make_target(env, **joint_vals: float) -> torch.Tensor:
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    vec = robot.data.default_joint_pos[0].clone()
    for name, val in joint_vals.items():
        if name in idx:
            vec[idx[name]] = val
    return vec


def main() -> int:
    env_cfg = prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    base_info = {
        "robot_root": [float(x) for x in robot.data.root_pos_w[0].tolist()],
        "default_joints_right_arm": {
            n: float(robot.data.default_joint_pos[0, i].item())
            for i, n in enumerate(robot.joint_names)
            if n.startswith("right_") and "hand" not in n
        },
    }

    candidates = [
        ("default", {}),
        ("reach_a", dict(
            right_shoulder_pitch_joint=0.35,
            right_shoulder_roll_joint=-0.95,
            right_shoulder_yaw_joint=-0.35,
            right_elbow_joint=1.05,
            right_wrist_pitch_joint=-0.35,
            right_wrist_yaw_joint=0.25,
        )),
        ("reach_b", dict(
            right_shoulder_pitch_joint=0.55,
            right_shoulder_roll_joint=-1.15,
            right_shoulder_yaw_joint=-0.55,
            right_elbow_joint=0.95,
            right_wrist_pitch_joint=-0.55,
            right_wrist_yaw_joint=0.35,
        )),
        ("reach_c", dict(
            right_shoulder_pitch_joint=0.85,
            right_shoulder_roll_joint=-1.25,
            right_shoulder_yaw_joint=-0.75,
            right_elbow_joint=0.75,
            right_wrist_pitch_joint=-0.75,
            right_wrist_yaw_joint=0.45,
        )),
        ("reach_d", dict(
            right_shoulder_pitch_joint=1.05,
            right_shoulder_roll_joint=-0.85,
            right_shoulder_yaw_joint=-0.45,
            right_elbow_joint=1.25,
            right_wrist_pitch_joint=-0.25,
            right_wrist_yaw_joint=0.55,
        )),
        ("reach_e", dict(
            right_shoulder_pitch_joint=-0.25,
            right_shoulder_roll_joint=-0.65,
            right_shoulder_yaw_joint=0.15,
            right_elbow_joint=1.35,
            right_wrist_pitch_joint=0.15,
            right_wrist_yaw_joint=-0.15,
        )),
    ]

    results = []
    for label, joints in candidates:
        env.reset()
        tgt = make_target(env, **joints)
        results.append(snapshot(env, label, tgt))

    out = {
        "base_info": base_info,
        "candidates": results,
        "best": min(results, key=lambda r: r["palm_dist"]),
    }
    path = JP_TEST_ROOT / "results" / "phase2" / "pose_diagnostic.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
