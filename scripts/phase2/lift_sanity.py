#!/usr/bin/env python3
"""Fast lift sanity check — reports best palm distance and lift delta."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2"))

from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks
from pick_place_trajectory import build_keyframes, run_keyframe
from robot_control import palm_object_distance, step_hold

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


def run_once(env) -> dict:
    env.reset()
    obj0 = float(env.scene["object"].data.root_pos_w[0, 2].item())
    for label, target, hold in build_keyframes(env):
        run_keyframe(env, label, target, hold)
    obj1 = float(env.scene["object"].data.root_pos_w[0, 2].item())
    return {
        "object_init_z": obj0,
        "object_final_z": obj1,
        "lift_delta_m": obj1 - obj0,
        "palm_object_dist_m": palm_object_distance(env),
        "root_z": float(env.scene["robot"].data.root_pos_w[0, 2].item()),
    }


def main() -> int:
    env = gym.make(TASK, cfg=prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))).unwrapped
    result = run_once(env)
    result["lift_success"] = result["lift_delta_m"] > 0.03 and result["object_final_z"] > 0.5
    out = JP_TEST_ROOT / "results" / "phase2" / "lift_sanity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    env.close()
    simulation_app.close()
    return 0 if result["lift_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
