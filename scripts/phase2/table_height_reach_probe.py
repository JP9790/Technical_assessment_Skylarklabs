#!/usr/bin/env python3
"""Probe reach poses keeping wrist height near table level."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks
from robot_control import RIGHT_HAND_OPEN, build_target, step_hold

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
OUT = JP_TEST_ROOT / "results" / "phase2" / "table_height_reach_probe.json"


def main() -> int:
    env = gym.make(TASK, cfg=prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))).unwrapped
    pitch_vals = [0.2, 0.5, 0.8, 1.1, 1.4]
    roll_vals = [-1.3, -1.0, -0.7, -0.4]
    elbow_vals = [0.5, 0.8, 1.1]
    yaw_vals = [-0.6, -0.2, 0.2, 0.6]

    best = None
    for pitch, roll, elbow, yaw in itertools.product(pitch_vals, roll_vals, elbow_vals, yaw_vals):
        env.reset()
        init_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
        target = build_target(
            env,
            right_shoulder_pitch_joint=pitch,
            right_shoulder_roll_joint=roll,
            right_shoulder_yaw_joint=yaw,
            right_elbow_joint=elbow,
            right_wrist_pitch_joint=-0.35,
            right_wrist_yaw_joint=0.15,
            **RIGHT_HAND_OPEN,
        )
        step_hold(env, target, 90)
        robot = env.scene["robot"]
        names = list(robot.data.body_names)
        wrist = robot.data.body_pos_w[0, names.index("right_wrist_yaw_link")]
        palm = robot.data.body_pos_w[0, names.index("right_hand_palm_link")]
        obj = env.scene["object"].data.root_pos_w[0]
        obj_z = float(obj[2].item())
        wrist_z = float(wrist[2].item())
        dist = float(torch.linalg.norm(palm - obj).item())
        if obj_z < init_z - 0.02 or wrist_z < 0.65:
            continue
        if best is None or dist < best["palm_object_dist_m"]:
            best = {
                "palm_object_dist_m": dist,
                "wrist_z": wrist_z,
                "object_z": obj_z,
                "palm": [float(x) for x in palm.tolist()],
                "object": [float(x) for x in obj.tolist()],
                "joints": {
                    "right_shoulder_pitch_joint": pitch,
                    "right_shoulder_roll_joint": roll,
                    "right_shoulder_yaw_joint": yaw,
                    "right_elbow_joint": elbow,
                    "right_wrist_pitch_joint": -0.35,
                    "right_wrist_yaw_joint": 0.15,
                },
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"best": best}, indent=2))
    print(json.dumps({"best": best}, indent=2))
    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
