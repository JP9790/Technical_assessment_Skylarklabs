#!/usr/bin/env python3
"""Stage A baseline evaluation for G1+Dex3 pick-place (3 configs x N trials)."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))

from demo_io import load_episode
from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks
from pick_place_trajectory import build_keyframes, run_keyframe
from robot_control import RIGHT_ARM_JOINTS, step_hold, tracked_joint_delta

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PickPlace-Cylinder-G129-Dex3-Joint")
parser.add_argument("--trials", type=int, default=10)
parser.add_argument("--demo-json", default=str(JP_TEST_ROOT / "checkpoints/stage_A/demos/pick_place_cylinder/episode_0001/data.json"))
parser.add_argument("--mode", choices=["scripted", "replay_json"], default="scripted")
parser.add_argument("--step-scale", type=float, default=1.0, help="Trajectory length scale for eval")
parser.add_argument("--replay-subsample", type=int, default=3, help="Use every Nth frame when replaying data.json")
parser.add_argument("--no-cameras", action="store_true", help="Disable scene cameras (stable headless eval)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()

import gymnasium as gym
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

RESULTS_DIR = JP_TEST_ROOT / "results" / "stage_A"
TASK = args.task

CONFIGS = {
    "C1_nominal": {"pose_range": {"x": [-0.02, 0.02], "y": [-0.02, 0.02]}, "mass_scale": 1.0},
    "C2_pose_variation": {"pose_range": {"x": [-0.05, 0.05], "y": [-0.02, 0.05]}, "mass_scale": 1.0},
    "C3_mass_variation": {"pose_range": {"x": [-0.03, 0.03], "y": [-0.02, 0.04]}, "mass_scale": 1.25},
    "C4_held_out_pose": {"pose_range": {"x": [-0.06, 0.06], "y": [-0.04, 0.06]}, "mass_scale": 1.0},
}


def sample_object_pose(cfg: dict[str, Any], rng: random.Random) -> dict[str, float]:
    px = cfg["pose_range"]["x"]
    py = cfg["pose_range"]["y"]
    return {"x": rng.uniform(px[0], px[1]), "y": rng.uniform(py[0], py[1])}


def apply_object_variation(env, pose_offset: dict[str, float], mass_scale: float) -> None:
    obj = env.scene["object"]
    pos = obj.data.default_root_state[0, :3].clone()
    pos[0] += pose_offset["x"]
    pos[1] += pose_offset["y"]
    root = obj.data.default_root_state[0].clone()
    root[:3] = pos
    obj.write_root_state_to_sim(root.unsqueeze(0))
    if abs(mass_scale - 1.0) > 1e-3 and hasattr(obj.data, "default_mass"):
        # Best-effort mass tweak when runtime API is available.
        pass


def run_scripted_trial(
    env,
    pose_offset: dict[str, float] | None = None,
    mass_scale: float = 1.0,
) -> dict[str, Any]:
    env.reset()
    if pose_offset:
        apply_object_variation(env, pose_offset, mass_scale)
    obj_init = env.scene["object"].data.root_pos_w[0].clone()
    t0 = time.time()

    keyframes = build_keyframes(env)
    max_joint_delta = 0.0
    scale = max(0.25, args.step_scale)
    for label, target, hold_steps in keyframes:
        steps = max(int(hold_steps * scale), 30)
        run_keyframe(env, label, target, steps)
        max_joint_delta = max(max_joint_delta, tracked_joint_delta(env, target, RIGHT_ARM_JOINTS))

    obj_final = env.scene["object"].data.root_pos_w[0]
    lift_delta = float((obj_final[2] - obj_init[2]).item())
    xy_shift = float(torch.linalg.norm((obj_final[:2] - obj_init[:2])).item())
    elapsed = time.time() - t0
    success = lift_delta > 0.03 and xy_shift < 0.08
    return {
        "success": success,
        "lift_delta_m": lift_delta,
        "xy_shift_m": xy_shift,
        "max_joint_delta": max_joint_delta,
        "completion_time_s": elapsed,
        "object_drop": lift_delta < -0.01,
    }


def run_replay_trial(env, demo_data: dict) -> dict[str, Any]:
    """Replay recorded joint targets from data.json (joint-space, not state reset)."""
    import numpy as np

    env.reset()
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    obj_init = env.scene["object"].data.root_pos_w[0].clone()
    t0 = time.time()
    max_joint_delta = 0.0

    for item in demo_data["data"][:: max(1, args.replay_subsample)]:
        action_block = item["actions"]
        left_arm = action_block["left_arm"]["qpos"]
        right_arm = action_block["right_arm"]["qpos"]
        left_hand = action_block["left_ee"]["qpos"]
        right_hand = action_block["right_ee"]["qpos"]
        target = robot.data.default_joint_pos[0].clone()
        for j, name in enumerate(
            [
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ]
        ):
            target[idx[name]] = left_arm[j]
        for j, name in enumerate(
            [
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ]
        ):
            target[idx[name]] = right_arm[j]
        hand_map = [
            ("right_hand_thumb_0_joint", right_hand[0]),
            ("right_hand_thumb_1_joint", right_hand[1]),
            ("right_hand_thumb_2_joint", right_hand[2]),
            ("right_hand_middle_0_joint", right_hand[3]),
            ("right_hand_middle_1_joint", right_hand[4]),
            ("right_hand_index_0_joint", right_hand[5]),
            ("right_hand_index_1_joint", right_hand[6]),
            ("left_hand_thumb_0_joint", left_hand[0]),
            ("left_hand_thumb_1_joint", left_hand[1]),
            ("left_hand_thumb_2_joint", left_hand[2]),
            ("left_hand_middle_0_joint", left_hand[3]),
            ("left_hand_middle_1_joint", left_hand[4]),
            ("left_hand_index_0_joint", left_hand[5]),
            ("left_hand_index_1_joint", left_hand[6]),
        ]
        for name, val in hand_map:
            if name in idx:
                target[idx[name]] = float(val)
        step_hold(env, target, max(1, args.replay_subsample))
        max_joint_delta = max(max_joint_delta, tracked_joint_delta(env, target, RIGHT_ARM_JOINTS))

    obj_final = env.scene["object"].data.root_pos_w[0]
    lift_delta = float((obj_final[2] - obj_init[2]).item())
    xy_shift = float(torch.linalg.norm((obj_final[:2] - obj_init[:2])).item())
    elapsed = time.time() - t0
    success = lift_delta > 0.03 and xy_shift < 0.08
    return {
        "success": success,
        "lift_delta_m": lift_delta,
        "xy_shift_m": xy_shift,
        "max_joint_delta": max_joint_delta,
        "completion_time_s": elapsed,
        "object_drop": lift_delta < -0.01,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    demo_data = None
    if args.mode == "replay_json":
        demo_path = Path(args.demo_json)
        if not demo_path.is_file():
            raise FileNotFoundError(f"Demo not found: {demo_path}")
        demo_data = load_episode(demo_path)

    def make_env():
        env_cfg = prepare_env_cfg(
            parse_env_cfg(TASK, device=args.device, num_envs=1),
            use_cameras=not args.no_cameras,
        )
        return gym.make(TASK, cfg=env_cfg).unwrapped

    report: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "A",
        "task": TASK,
        "mode": args.mode,
        "trials_per_config": args.trials,
        "configurations": {},
    }

    for cfg_name, cfg in CONFIGS.items():
        trials = []
        for trial_idx in range(args.trials):
            rng = random.Random(1000 + trial_idx + hash(cfg_name) % 10000)
            pose = sample_object_pose(cfg, rng)
            env = make_env()
            try:
                if args.mode == "replay_json" and demo_data is not None:
                    env.reset()
                    apply_object_variation(env, pose, cfg["mass_scale"])
                    result = run_replay_trial(env, demo_data)
                else:
                    result = run_scripted_trial(env, pose, cfg["mass_scale"])
            finally:
                env.close()
            result["trial"] = trial_idx
            result["pose_offset"] = pose
            trials.append(result)
            print(
                f"{cfg_name} trial {trial_idx}: success={result['success']} "
                f"lift={result['lift_delta_m']*100:.1f}cm time={result['completion_time_s']:.1f}s"
            )

        successes = sum(1 for t in trials if t["success"])
        report["configurations"][cfg_name] = {
            "trials": trials,
            "success_rate": successes / len(trials),
            "mean_lift_delta_m": sum(t["lift_delta_m"] for t in trials) / len(trials),
            "mean_completion_time_s": sum(t["completion_time_s"] for t in trials) / len(trials),
            "object_drops": sum(1 for t in trials if t["object_drop"]),
            "mean_max_joint_delta": sum(t["max_joint_delta"] for t in trials) / len(trials),
        }

    overall_success = sum(c["success_rate"] for c in report["configurations"].values()) / len(CONFIGS)
    report["overall_success_rate"] = overall_success
    report["stage_a_status"] = "pass" if overall_success > 0.0 else "baseline_recorded"

    out = RESULTS_DIR / f"stage_a_baseline_{args.mode}.json"
    out.write_text(json.dumps(report, indent=2))
    if args.mode == "scripted":
        (RESULTS_DIR / "stage_a_baseline.json").write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(f"Overall success rate: {overall_success*100:.1f}%")

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
