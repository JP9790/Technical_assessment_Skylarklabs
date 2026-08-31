#!/usr/bin/env python3
"""Build Dex3 fingertip cache (wrist-relative) from Stage A demo playback in G1 sim."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase1"))
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase2"))

from demo_io import load_episode
from isaac_bootstrap import import_unitree_tasks
from robot_control import step_hold

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--demo",
    default=str(JP_TEST_ROOT / "checkpoints/stage_A/demos/pick_place_cylinder/episode_0001/data.json"),
)
parser.add_argument("--subsample", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()
import gymnasium as gym
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"
OUT = JP_TEST_ROOT / "results" / "phase5" / "dex3_tip_cache.json"

FINGERS = {
    "right": {
        "thumb": "right_hand_thumb_2_link",
        "index": "right_hand_index_1_link",
        "middle": "right_hand_middle_1_link",
    },
    "left": {
        "thumb": "left_hand_thumb_2_link",
        "index": "left_hand_index_1_link",
        "middle": "left_hand_middle_1_link",
    },
}
WRIST = {"right": "right_wrist_yaw_link", "left": "left_wrist_yaw_link"}


def tips_in_wrist_frame(robot, side: str) -> dict[str, list[float]]:
    names = list(robot.data.body_names)
    wrist = robot.data.body_pos_w[0, names.index(WRIST[side])].cpu().numpy()
    out: dict[str, list[float]] = {}
    for finger, link in FINGERS[side].items():
        pos = robot.data.body_pos_w[0, names.index(link)].cpu().numpy()
        out[finger] = (pos - wrist).tolist()
    return out


def main() -> int:
    demo = load_episode(Path(args.demo))
    env = gym.make(TASK, cfg=prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))).unwrapped
    env.reset()
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    idx = {n: i for i, n in enumerate(names)}

    frames = []
    for i, item in enumerate(demo["data"][:: max(1, args.subsample)]):
        target = robot.data.default_joint_pos[0].clone()
        for block, prefix in (("left_arm", "left"), ("right_arm", "right")):
            arm = item["states"][block]["qpos"]
            arm_names = demo["info"]["joint_names"][block]
            for j, n in enumerate(arm_names):
                if n in idx:
                    target[idx[n]] = float(arm[j])
        for block, prefix in (("left_ee", "left"), ("right_ee", "right")):
            hand = item["states"][block]["qpos"]
            hand_names = demo["info"]["joint_names"][block]
            for j, n in enumerate(hand_names):
                if n in idx:
                    target[idx[n]] = float(hand[j])
        step_hold(env, target, 1)
        frames.append(
            {
                "idx": i,
                "right_tips_wrist": tips_in_wrist_frame(robot, "right"),
                "left_tips_wrist": tips_in_wrist_frame(robot, "left"),
            }
        )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_demo": str(args.demo),
        "num_frames": len(frames),
        "frames": frames,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))
    print(f"Wrote {OUT} ({len(frames)} frames)")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
