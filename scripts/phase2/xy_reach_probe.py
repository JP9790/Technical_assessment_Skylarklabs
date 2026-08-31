#!/usr/bin/env python3
"""Find arm pose minimizing horizontal palm-object distance (object stays on table)."""

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
from robot_control import LEFT_HAND_OPEN, RIGHT_HAND_OPEN, build_target

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["right", "left"], default="right")
parser.add_argument("--steps", type=int, default=150)
parser.add_argument("--quick", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()

import gymnasium as gym
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"
OUT = JP_TEST_ROOT / "results" / "phase2" / f"xy_reach_probe_{args.arm}.json"


def palm_xy_dist(env) -> float:
    robot = env.scene["robot"]
    side = args.arm
    palm_idx = list(robot.data.body_names).index(f"{side}_hand_palm_link")
    palm = robot.data.body_pos_w[0, palm_idx]
    obj = env.scene["object"].data.root_pos_w[0]
    return float(torch.linalg.norm(palm[:2] - obj[:2]).item())


def run_pose(env, joints: dict[str, float]) -> tuple[float, float, float]:
    hand = RIGHT_HAND_OPEN if args.arm == "right" else LEFT_HAND_OPEN
    target = build_target(env, **hand, **joints)
    action = action_from_target(env, target)
    for _ in range(args.steps):
        env.step(action)
    init_check = float(env.scene["object"].data.root_pos_w[0, 2].item())
    xy = palm_xy_dist(env)
    obj_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
    robot = env.scene["robot"]
    palm_idx = list(robot.data.body_names).index(f"{args.arm}_hand_palm_link")
    palm_z = float(robot.data.body_pos_w[0, palm_idx, 2].item())
    return xy, obj_z, palm_z


def main() -> int:
    prefix = "right_" if args.arm == "right" else "left_"
    env = gym.make(TASK, cfg=prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))).unwrapped

    if args.quick:
        waist_vals = [-1.0, 0.0, 1.0]
        pitch_vals = [0.0, 0.8, 1.5]
        roll_vals = [-1.2, -0.6]
        elbow_vals = [0.8, 1.4]
        yaw_vals = [-0.3, 0.3]
    else:
        waist_vals = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
        pitch_vals = [-0.5, 0.0, 0.5, 1.0, 1.5]
        roll_vals = [-1.5, -1.0, -0.5, 0.0]
        elbow_vals = [0.5, 1.0, 1.5]
        yaw_vals = [-0.5, 0.0, 0.5]

    best = None
    tested = 0
    for wy, pitch, roll, elbow, yaw in itertools.product(
        waist_vals, pitch_vals, roll_vals, elbow_vals, yaw_vals
    ):
        env.reset()
        init_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
        joints = {
            "waist_yaw_joint": wy,
            f"{prefix}shoulder_pitch_joint": pitch,
            f"{prefix}shoulder_roll_joint": roll,
            f"{prefix}shoulder_yaw_joint": yaw,
            f"{prefix}elbow_joint": elbow,
            f"{prefix}wrist_pitch_joint": -0.3,
            f"{prefix}wrist_yaw_joint": 0.2,
        }
        xy, obj_z, palm_z = run_pose(env, joints)
        tested += 1
        if obj_z < init_z - 0.015:
            continue
        if best is None or xy < best["xy_dist_m"]:
            best = {
                "xy_dist_m": xy,
                "palm_z": palm_z,
                "object_z": obj_z,
                "joints": joints,
            }

    result = {"arm": args.arm, "tested": tested, "best": best}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
