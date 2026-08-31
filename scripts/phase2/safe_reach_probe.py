#!/usr/bin/env python3
"""Find reach pose that minimizes palm distance while keeping object on table."""

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
parser.add_argument("--hold-steps", type=int, default=90)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()
import gymnasium as gym
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"
OUT = JP_TEST_ROOT / "results" / "phase2" / "safe_reach_probe.json"


def score(dist: float, obj_z: float, init_z: float) -> float:
    if obj_z < init_z - 0.02:
        return 999.0
    return dist


def main() -> int:
    env = gym.make(TASK, cfg=prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))).unwrapped
    pitch_vals = [-1.8, -1.4, -1.0, -0.6, 0.0, 0.4]
    roll_vals = [-1.4, -1.0, -0.6, -0.2]
    elbow_vals = [0.9, 1.3, 1.7]
    yaw_vals = [-0.8, -0.2, 0.4]
    wp_vals = [-1.0, -0.4, 0.2]

    best = None
    tested = 0
    for pitch, roll, elbow, yaw, wp in itertools.product(
        pitch_vals, roll_vals, elbow_vals, yaw_vals, wp_vals
    ):
        env.reset()
        init_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
        target = build_target(
            env,
            right_shoulder_pitch_joint=pitch,
            right_shoulder_roll_joint=roll,
            right_shoulder_yaw_joint=yaw,
            right_elbow_joint=elbow,
            right_wrist_pitch_joint=wp,
            right_wrist_yaw_joint=0.2,
            **RIGHT_HAND_OPEN,
        )
        step_hold(env, target, args.hold_steps)
        dist = palm_object_distance(env)
        obj_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
        s = score(dist, obj_z, init_z)
        tested += 1
        if best is None or s < best["score"]:
            robot = env.scene["robot"]
            names = list(robot.data.body_names)
            palm = robot.data.body_pos_w[0, names.index("right_hand_palm_link")]
            obj = env.scene["object"].data.root_pos_w[0]
            best = {
                "score": s,
                "palm_object_dist_m": dist,
                "object_z": obj_z,
                "object_on_table": obj_z > init_z - 0.02,
                "palm": [float(x) for x in palm.tolist()],
                "object": [float(x) for x in obj.tolist()],
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
