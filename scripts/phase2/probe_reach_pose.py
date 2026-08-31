#!/usr/bin/env python3
"""Grid-search arm joints to minimize palm-to-cylinder distance (headless probe)."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from action_utils import action_from_target
from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PickPlace-Cylinder-G129-Dex3-Joint")
parser.add_argument("--hold-steps", type=int, default=80)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()

import gymnasium as gym
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

RESULTS = JP_TEST_ROOT / "results" / "phase2" / "reach_pose_probe.json"


def palm_object_distance(env) -> float:
    robot = env.scene["robot"]
    obj = env.scene["object"].data.root_pos_w[0]
    body_names = list(robot.data.body_names)
    palm_idx = body_names.index("right_hand_palm_link")
    palm = robot.data.body_pos_w[0, palm_idx]
    return float(torch.linalg.norm(palm - obj).item())


def hold_pose(env, target: torch.Tensor, steps: int) -> None:
    action = action_from_target(env, target)
    for _ in range(steps):
        env.step(action)


def main() -> int:
    env_cfg = prepare_env_cfg(parse_env_cfg(args.task, device=args.device, num_envs=1))
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    base = robot.data.default_joint_pos[0].clone()

    def make(**joint_vals: float) -> torch.Tensor:
        vec = base.clone()
        for name, val in joint_vals.items():
            if name in idx:
                vec[idx[name]] = val
        return vec

    pitch_vals = [0.0, 0.30, 0.60, 0.90, 1.20]
    roll_vals = [-1.0, -0.70, -0.40]
    elbow_vals = [0.55, 0.85, 1.15]
    wrist_pitch_vals = [-0.45, -0.15, 0.15]
    wrist_yaw_vals = [-0.15, 0.20, 0.45]

    best: dict | None = None
    tested = 0
    for pitch, roll, elbow, wp, wy in itertools.product(
        pitch_vals, roll_vals, elbow_vals, wrist_pitch_vals, wrist_yaw_vals
    ):
        env.reset()
        target = make(
            right_shoulder_pitch_joint=pitch,
            right_shoulder_roll_joint=roll,
            right_elbow_joint=elbow,
            right_wrist_pitch_joint=wp,
            right_wrist_yaw_joint=wy,
            right_hand_index_0_joint=0.10,
            right_hand_middle_0_joint=0.10,
            right_hand_thumb_0_joint=0.0,
            right_hand_thumb_1_joint=-0.2,
            right_hand_thumb_2_joint=-0.10,
        )
        hold_pose(env, target, args.hold_steps)
        dist = palm_object_distance(env)
        obj_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
        tested += 1
        if best is None or dist < best["palm_object_dist_m"]:
            best = {
                "palm_object_dist_m": dist,
                "object_z": obj_z,
                "joints": {
                    "right_shoulder_pitch_joint": pitch,
                    "right_shoulder_roll_joint": roll,
                    "right_elbow_joint": elbow,
                    "right_wrist_pitch_joint": wp,
                    "right_wrist_yaw_joint": wy,
                },
            }

    assert best is not None
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"tested": tested, "best": best}, indent=2))
    print(json.dumps({"tested": tested, "best": best}, indent=2))

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
