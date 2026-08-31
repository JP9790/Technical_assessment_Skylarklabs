"""Read/write xr_teleoperate-compatible data.json demonstrations."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

LEFT_ARM = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
RIGHT_HAND = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]
LEFT_HAND = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]


def _tolist(x: Any) -> list:
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, list):
        return x
    return [float(x)]


def split_arm_hand(joint_pos, joint_index: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    left_arm = np.array([float(joint_pos[joint_index[n]]) for n in LEFT_ARM], dtype=np.float64)
    right_arm = np.array([float(joint_pos[joint_index[n]]) for n in RIGHT_ARM], dtype=np.float64)
    left_hand = np.array([float(joint_pos[joint_index[n]]) for n in LEFT_HAND], dtype=np.float64)
    right_hand = np.array([float(joint_pos[joint_index[n]]) for n in RIGHT_HAND], dtype=np.float64)
    robot_action = np.concatenate([left_arm, right_arm])
    hand_action = np.concatenate([right_hand, left_hand])
    return robot_action, hand_action


def make_item(
    idx: int,
    joint_pos,
    robot_action: np.ndarray,
    hand_action: np.ndarray,
    sim_state: dict,
) -> dict:
    left_arm = robot_action[:7].tolist()
    right_arm = robot_action[7:].tolist()
    right_hand = hand_action[:7].tolist()
    left_hand = hand_action[7:].tolist()
    return {
        "idx": idx,
        "colors": {},
        "depths": {},
        "states": {
            "left_arm": {"qpos": left_arm, "qvel": [], "torque": []},
            "right_arm": {"qpos": right_arm, "qvel": [], "torque": []},
            "left_ee": {"qpos": left_hand, "qvel": [], "torque": []},
            "right_ee": {"qpos": right_hand, "qvel": [], "torque": []},
            "body": {"qpos": []},
        },
        "actions": {
            "left_arm": {"qpos": left_arm, "qvel": [], "torque": []},
            "right_arm": {"qpos": right_arm, "qvel": [], "torque": []},
            "left_ee": {"qpos": left_hand, "qvel": [], "torque": []},
            "right_ee": {"qpos": right_hand, "qvel": [], "torque": []},
            "body": {"qpos": []},
        },
        "sim_state": sim_state,
    }


def write_episode(
    out_dir: Path,
    task_name: str,
    items: list[dict],
    goal: str = "Pick and lift cylinder.",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "data.json"
    info = {
        "version": "1.0.0",
        "date": date.today().isoformat(),
        "author": "jp_test_phase2",
        "image": {"width": 640, "height": 480, "fps": 30},
        "depth": {"width": 640, "height": 480, "fps": 30},
        "audio": {"sample_rate": 16000, "channels": 1, "format": "PCM", "bits": 16},
        "joint_names": {
            "left_arm": LEFT_ARM,
            "left_ee": LEFT_HAND,
            "right_arm": RIGHT_ARM,
            "right_ee": RIGHT_HAND,
            "body": [],
        },
        "tactile_names": {"left_ee": [], "right_ee": []},
        "sim_state": "",
    }
    text = {
        "goal": goal,
        "desc": "Sim-recorded scripted pick-place trajectory (Stage A baseline).",
        "steps": "approach -> grasp -> lift",
    }
    payload = {"info": info, "text": text, "data": items}
    json_path.write_text(json.dumps(payload, indent=2))
    return json_path


def load_episode(json_path: Path) -> dict:
    return json.loads(json_path.read_text())
