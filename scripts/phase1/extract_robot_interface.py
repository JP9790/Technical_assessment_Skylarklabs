#!/usr/bin/env python3
"""Phase 1 — extract G1+Dex3 robot interface from Isaac Sim and save to configs/."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks  # noqa: E402

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PickPlace-Cylinder-G129-Dex3-Joint")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()

import gymnasium as gym
import yaml
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = args.task
RESULTS_DIR = JP_TEST_ROOT / "results" / "phase1"
CONFIG_OUT = JP_TEST_ROOT / "configs" / "g1_dex3_source.yaml"


def _tensor_list(t) -> list:
    return t[0].detach().cpu().tolist()


def extract(env) -> dict:
    robot = env.scene["robot"]
    data = robot.data

    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)

    # Index maps
    name_to_idx = {n: i for i, n in enumerate(joint_names)}

    dex3_joint_names = []
    try:
        from tasks.common_observations.dex3_state import get_robot_girl_joint_names

        dex3_joint_names = get_robot_girl_joint_names()
    except Exception:
        dex3_joint_names = [n for n in joint_names if "_hand_" in n]

    arm_joint_names = []
    try:
        from tasks.common_observations.g1_29dof_state import (
            get_robot_arm_joint_names,
            get_robot_boy_joint_names,
        )

        arm_joint_names = get_robot_arm_joint_names()
        body_joint_names = get_robot_boy_joint_names()
    except Exception:
        body_joint_names = [n for n in joint_names if "_hand_" not in n]

    dex3_indices = [name_to_idx[n] for n in dex3_joint_names if n in name_to_idx]

    # Heuristic link identification
    def find_links(keywords: list[str]) -> list[str]:
        out = []
        for bn in body_names:
            low = bn.lower()
            if any(k in low for k in keywords):
                out.append(bn)
        return out

    wrist_links = find_links(["wrist"])
    palm_links = find_links(["palm", "hand_palm"])
    tip_links = find_links(["tip", "distal"])

    limits = {
        "position_lower": dict(zip(joint_names, _tensor_list(data.soft_joint_pos_limits[..., 0]))),
        "position_upper": dict(zip(joint_names, _tensor_list(data.soft_joint_pos_limits[..., 1]))),
    }
    if hasattr(data, "soft_joint_vel_limits"):
        limits["velocity"] = dict(zip(joint_names, _tensor_list(data.soft_joint_vel_limits)))
    if hasattr(data, "soft_joint_effort_limits"):
        limits["effort"] = dict(zip(joint_names, _tensor_list(data.soft_joint_effort_limits)))
    elif hasattr(data, "joint_effort_limits"):
        limits["effort"] = dict(zip(joint_names, _tensor_list(data.joint_effort_limits)))

    default_pos = dict(zip(joint_names, _tensor_list(data.default_joint_pos)))
    default_vel = dict(zip(joint_names, _tensor_list(data.default_joint_vel)))

    cfg = env.cfg
    control_frequency_hz = 1.0 / (cfg.sim.dt * cfg.decimation)

    return {
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "embodiment": "Unitree G1 29DoF + Dex3",
        "sim": {
            "physics_dt": cfg.sim.dt,
            "decimation": cfg.decimation,
            "control_frequency_hz": control_frequency_hz,
            "episode_length_s": cfg.episode_length_s,
        },
        "joints": {
            "all_names": joint_names,
            "body_dof_names": body_joint_names,
            "arm_names": arm_joint_names,
            "dex3_names": dex3_joint_names,
            "dex3_indices_in_robot_state": dex3_indices,
            "count": len(joint_names),
        },
        "links": {
            "all_names": body_names,
            "wrist_candidates": wrist_links,
            "palm_candidates": palm_links,
            "fingertip_candidates": tip_links,
        },
        "limits": limits,
        "default_joint_pos": default_pos,
        "default_joint_vel": default_vel,
        "actuator_groups_from_unitree_cfg": {
            "legs": ".*_hip_yaw_joint, .*_hip_roll_joint, .*_hip_pitch_joint, .*_knee_joint",
            "waist": "waist_yaw_joint, waist_roll_joint, waist_pitch_joint (locked in base_fix preset)",
            "feet": ".*_ankle_pitch_joint, .*_ankle_roll_joint",
            "arms": ".*_shoulder_.*_joint, .*_elbow_joint, .*_wrist_.*_joint",
            "hands": ".*_hand_index_.*_joint, .*_hand_middle_.*_joint, .*_hand_thumb_.*_joint",
        },
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Creating env: {TASK}")
    env_cfg = prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    report = extract(env)
    CONFIG_OUT.write_text(yaml.dump(report, sort_keys=False, default_flow_style=False))
    json_out = RESULTS_DIR / "g1_dex3_interface.json"
    json_out.write_text(json.dumps(report, indent=2))

    print(f"Wrote {CONFIG_OUT}")
    print(f"Wrote {json_out}")
    print(f"Joints: {report['joints']['count']}, bodies: {len(report['links']['all_names'])}")
    print(f"Control frequency: {report['sim']['control_frequency_hz']:.1f} Hz")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
