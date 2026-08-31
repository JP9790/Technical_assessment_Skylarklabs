#!/usr/bin/env python3
"""Quick headless smoke test: launch G1+Dex3 pick-place env and run 50 steps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# --- bootstrap paths only (no isaac imports yet) ---
sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks  # noqa: E402

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PickPlace-Cylinder-G129-Dex3-Joint")
parser.add_argument("--steps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()

import gymnasium as gym
import torch
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = args.task
print(f"Smoke test: {TASK}")
env_cfg = prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))
env = gym.make(TASK, cfg=env_cfg).unwrapped
obs, _ = env.reset()
zero_action = torch.zeros(env.action_space.shape, device=env.device)

for i in range(args.steps):
    action = zero_action
    obs, rew, term, trunc, info = env.step(action)
    if term or trunc:
        obs, _ = env.reset()

robot = env.scene["robot"]
obj = env.scene["object"]
print(f"Steps OK: {args.steps}")
print(f"Robot joints: {robot.num_joints}, bodies: {robot.num_bodies}")
print(f"Object pos: {obj.data.root_pos_w[0].tolist()}")
print(f"Max joint vel: {float(torch.max(torch.abs(robot.data.joint_vel)).item()):.4f}")

env.close()
simulation_app.close()
print("SMOKE TEST PASSED")
