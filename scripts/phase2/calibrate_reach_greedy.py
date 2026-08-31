#!/usr/bin/env python3
"""Greedy arm joint calibration — palm near cylinder without knocking it off."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks
from robot_control import (
    RIGHT_HAND_CLOSED,
    RIGHT_HAND_OPEN,
    apply_hand,
    build_target,
    palm_object_distance,
    step_hold,
    step_hold_kinematic,
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--iters", type=int, default=250)
parser.add_argument("--delta", type=float, default=0.05)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()

import gymnasium as gym
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"
OUT = JP_TEST_ROOT / "results" / "phase2" / "calibrated_reach.json"

SEARCH_JOINTS = [
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
TARGET_DIST = 0.042


def score(palm_dist: float, obj_z: float, init_z: float) -> float:
    if obj_z < init_z - 0.012:
        return 999.0
    return abs(palm_dist - TARGET_DIST)


def get_joint_limits(env, name: str) -> tuple[float, float]:
    robot = env.scene["robot"]
    i = list(robot.joint_names).index(name)
    lo = float(robot.data.soft_joint_pos_limits[0, i, 0].item())
    hi = float(robot.data.soft_joint_pos_limits[0, i, 1].item())
    return lo, hi


def apply_pose(env, arm_joints: dict[str, float], steps: int = 4) -> None:
    tgt = build_target(env, **RIGHT_HAND_OPEN, **arm_joints)
    step_hold_kinematic(env, tgt, steps)


def restore_if_needed(env, init_z: float, arm_joints: dict[str, float]) -> float:
    obj_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
    if obj_z < init_z - 0.012:
        env.reset()
        apply_pose(env, arm_joints)
        obj_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
    return obj_z


def main() -> int:
    env = gym.make(TASK, cfg=prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))).unwrapped
    env.reset()
    init_z = float(env.scene["object"].data.root_pos_w[0, 2].item())

    arm_joints = {n: 0.0 for n in SEARCH_JOINTS}
    apply_pose(env, arm_joints)
    best_score = score(palm_object_distance(env), init_z, init_z)
    delta = args.delta

    for it in range(args.iters):
        improved = False
        for jname in SEARCH_JOINTS:
            lo, hi = get_joint_limits(env, jname)
            for sign in (1.0, -1.0):
                trial = dict(arm_joints)
                trial[jname] = max(lo, min(hi, trial[jname] + sign * delta))
                apply_pose(env, trial, steps=3)
                obj_z = restore_if_needed(env, init_z, arm_joints)
                dist = palm_object_distance(env)
                s = score(dist, obj_z, init_z)
                if s < best_score:
                    best_score = s
                    arm_joints = trial
                    improved = True
                else:
                    apply_pose(env, arm_joints, steps=2)
        if not improved:
            delta *= 0.65
            if delta < 0.008:
                break

    apply_pose(env, arm_joints, steps=10)
    pre_dist = palm_object_distance(env)
    pre_z = float(env.scene["object"].data.root_pos_w[0, 2].item())

    grasp_tgt = apply_hand(build_target(env, **RIGHT_HAND_OPEN, **arm_joints), env, RIGHT_HAND_CLOSED)
    step_hold(env, grasp_tgt, 140)

    lift_j = dict(arm_joints)
    lift_j["right_shoulder_pitch_joint"] -= 0.11
    lift_j["right_elbow_joint"] -= 0.09
    lift_tgt = apply_hand(build_target(env, **RIGHT_HAND_CLOSED, **lift_j), env, RIGHT_HAND_CLOSED)
    step_hold(env, lift_tgt, 180)
    final_z = float(env.scene["object"].data.root_pos_w[0, 2].item())

    result = {
        "joints": arm_joints,
        "palm_object_dist_m": pre_dist,
        "calibration_score": best_score,
        "object_init_z": init_z,
        "object_final_z": final_z,
        "lift_delta_m": final_z - init_z,
        "lift_success": (final_z - init_z) > 0.03 and pre_z > init_z - 0.02,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    env.close()
    simulation_app.close()
    return 0 if result["lift_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
