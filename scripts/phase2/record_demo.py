#!/usr/bin/env python3
"""Record a Stage A demonstration in xr_teleoperate data.json format."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))

from demo_io import make_item, split_arm_hand, write_episode
from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks
from pick_place_trajectory import build_keyframes, expand_keyframes, run_keyframe
from robot_control import step_hold

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PickPlace-Cylinder-G129-Dex3-Joint")
parser.add_argument("--episode-dir", default=str(JP_TEST_ROOT / "checkpoints/stage_A/demos/pick_place_cylinder/episode_0001"))
parser.add_argument("--no-cameras", action="store_true", help="Disable scene cameras (stable headless record)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()

import gymnasium as gym
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from tools.data_json_load import sim_state_to_json

RESULTS_DIR = JP_TEST_ROOT / "results" / "phase2"
TASK = args.task


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    env_cfg = prepare_env_cfg(
        parse_env_cfg(TASK, device=args.device, num_envs=1),
        use_cameras=not args.no_cameras,
    )
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    robot = env.scene["robot"]
    joint_index = {n: i for i, n in enumerate(robot.joint_names)}
    obj_init_z = float(env.scene["object"].data.root_pos_w[0, 2].item())

    keyframes = build_keyframes(env)
    steps = expand_keyframes(keyframes)
    items = []

    for i, (label, target) in enumerate(steps):
        run_keyframe(env, label.split("_")[0], target, 1)
        q = robot.data.joint_pos[0]
        robot_action, hand_action = split_arm_hand(q, joint_index)
        env_state = env.scene.get_state()
        sim_state = {
            "init_state": sim_state_to_json(env_state),
            "task_name": TASK,
        }
        items.append(make_item(i, q, robot_action, hand_action, sim_state))

    obj_final_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
    lift_delta = obj_final_z - obj_init_z

    episode_dir = Path(args.episode_dir)
    json_path = write_episode(episode_dir, TASK, items)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "episode_dir": str(episode_dir),
        "data_json": str(json_path),
        "num_steps": len(items),
        "object_init_z": obj_init_z,
        "object_final_z": obj_final_z,
        "lift_delta_m": lift_delta,
        "lift_success": lift_delta > 0.03,
        "source_type": "sim_scripted_demo",
        "format": "xr_teleoperate_data.json",
    }
    meta_path = RESULTS_DIR / "recorded_demo.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Recorded {len(items)} steps -> {json_path}")
    print(f"Lift delta: {lift_delta * 100:.1f} cm success={meta['lift_success']}")

    env.close()
    simulation_app.close()
    # Demo saved successfully; exit non-zero if lift failed (§4.3 gate).
    return 0 if meta["lift_success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
