#!/usr/bin/env python3
"""Numerical IK calibration (sim FK) for table-safe reach pose."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks
from robot_control import RIGHT_HAND_OPEN, build_target, palm_object_distance, step_hold_kinematic

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--restarts", type=int, default=40)
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

JOINTS = [
    "waist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


def bounds_for(env) -> list[tuple[float, float]]:
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    out = []
    for j in JOINTS:
        i = names.index(j)
        lo = float(robot.data.soft_joint_pos_limits[0, i, 0].item())
        hi = float(robot.data.soft_joint_pos_limits[0, i, 1].item())
        out.append((lo, hi))
    return out


def eval_pose(env, q: np.ndarray, init_z: float) -> tuple[float, float, float]:
    joints = {j: float(q[i]) for i, j in enumerate(JOINTS)}
    tgt = build_target(env, **RIGHT_HAND_OPEN, **joints)
    step_hold_kinematic(env, tgt, 4)
    dist = palm_object_distance(env)
    obj_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
    penalty = 0.0 if obj_z >= init_z - 0.01 else 50.0 + 100.0 * (init_z - obj_z)
    return dist + penalty, dist, obj_z


def main() -> int:
    from scipy.optimize import minimize

    env = gym.make(TASK, cfg=prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))).unwrapped
    bnds = bounds_for(env)
    rng = random.Random(42)
    best_q = None
    best_dist = 999.0
    init_z_ref = 0.84

    for r in range(args.restarts):
        env.reset()
        init_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
        init_z_ref = init_z
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bnds])

        def objective(q: np.ndarray) -> float:
            env.reset()
            val, _, _ = eval_pose(env, q, init_z)
            return val

        res = minimize(objective, x0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 60})
        env.reset()
        total, dist, obj_z = eval_pose(env, res.x, init_z)
        if obj_z >= init_z - 0.01 and dist < best_dist:
            best_dist = dist
            best_q = res.x.copy()
        print(f"restart {r}: dist={dist:.3f} obj_z={obj_z:.3f} ok={obj_z >= init_z - 0.01}")

    if best_q is None:
        print("No valid pose found")
        env.close()
        simulation_app.close()
        return 1

    joints = {j: float(best_q[i]) for i, j in enumerate(JOINTS)}
    env.reset()
    init_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
    _, dist, _ = eval_pose(env, best_q, init_z)

    result = {
        "method": "scipy_lbfgsb",
        "joints": joints,
        "palm_object_dist_m": dist,
        "object_init_z": init_z,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
