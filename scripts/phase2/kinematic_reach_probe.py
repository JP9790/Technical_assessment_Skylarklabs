#!/usr/bin/env python3
"""Kinematic grid search with stable standing pose (find reach joints)."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks
from robot_control import RIGHT_HAND_OPEN, build_target, palm_object_distance, step_hold

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hold-steps", type=int, default=100)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()

import gymnasium as gym
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"
OUT = JP_TEST_ROOT / "results" / "phase2" / "kinematic_reach_probe.json"


def main() -> int:
    env_cfg = prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))
    env = gym.make(TASK, cfg=env_cfg).unwrapped

    pitch_vals = [-1.6, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2]
    roll_vals = [-1.2, -0.8, -0.4, 0.0]
    elbow_vals = [0.6, 1.0, 1.4]
    yaw_vals = [-0.6, 0.0, 0.6]
    wp_vals = [-0.8, -0.2, 0.4]

    best = None
    tested = 0
    for pitch, roll, elbow, yaw, wp in itertools.product(
        pitch_vals, roll_vals, elbow_vals, yaw_vals, wp_vals
    ):
        env.reset()
        target = build_target(
            env,
            right_shoulder_pitch_joint=pitch,
            right_shoulder_roll_joint=roll,
            right_shoulder_yaw_joint=yaw,
            right_elbow_joint=elbow,
            right_wrist_pitch_joint=wp,
            right_wrist_yaw_joint=0.15,
            **RIGHT_HAND_OPEN,
        )
        step_hold(env, target, args.hold_steps)
        dist = palm_object_distance(env)
        obj_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
        root_z = float(env.scene["robot"].data.root_pos_w[0, 2].item())
        tested += 1
        if best is None or dist < best["palm_object_dist_m"]:
            best = {
                "palm_object_dist_m": dist,
                "object_z": obj_z,
                "root_z": root_z,
                "joints": {
                    "right_shoulder_pitch_joint": pitch,
                    "right_shoulder_roll_joint": roll,
                    "right_shoulder_yaw_joint": yaw,
                    "right_elbow_joint": elbow,
                    "right_wrist_pitch_joint": wp,
                },
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"tested": tested, "best": best}, indent=2))
    print(json.dumps({"tested": tested, "best": best}, indent=2))
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
