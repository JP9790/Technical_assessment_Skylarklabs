#!/usr/bin/env python3
"""Update dex3_to_brainco.yaml with verified BrainCo frame names from Phase 3."""

from __future__ import annotations

from pathlib import Path

import yaml

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
TARGET = JP_TEST_ROOT / "configs" / "g1_brainco_target.yaml"
OUT = JP_TEST_ROOT / "configs" / "dex3_to_brainco.yaml"
DEX3 = JP_TEST_ROOT / "configs" / "dex3_joint_map.yaml"


def main() -> None:
    brainco = yaml.safe_load(TARGET.read_text())
    dex3 = yaml.safe_load(DEX3.read_text())

    right = brainco["right_hand"]
    left = brainco["left_hand"]
    dex_frames = dex3.get("frames", {})

    mapping = {
        "status": "frames_verified_phase3",
        "source_embodiment": "Unitree G1 29DoF + Dex3",
        "target_embodiment": "Unitree G1 29DoF + BrainCo Revo2 Touch",
        "target_asset": {
            "repository": brainco["source_repository"],
            "commit": brainco["source_commit"],
            "usd_right": right["usd_path"],
            "usd_left": left["usd_path"],
        },
        "retargeting": {
            "method": "task_space_ik",
            "wrist": {
                "source_frame_right": dex_frames.get("right_wrist", "right_wrist_yaw_link"),
                "target_frame_right": right["wrist_mount_link"],
                "source_frame_left": dex_frames.get("left_wrist", "left_wrist_yaw_link"),
                "target_frame_left": left["wrist_mount_link"],
                "weight_position": 1.0,
                "weight_orientation": 1.0,
            },
            "palm": {
                "source_frame_right": dex_frames.get("right_palm", "right_hand_palm_link"),
                "target_frame_right": right["palm_link"],
                "source_frame_left": dex_frames.get("left_palm", "left_hand_palm_link"),
                "target_frame_left": left["palm_link"],
                "weight_position": 0.8,
                "weight_orientation": 0.8,
            },
            "fingertips": {
                "thumb": {
                    "source_right": dex_frames.get("right_thumb_tip", "right_hand_thumb_2_link"),
                    "target_right": right["fingertip_links"]["thumb"],
                    "source_left": dex_frames.get("left_thumb_tip", "left_hand_thumb_2_link"),
                    "target_left": left["fingertip_links"]["thumb"],
                    "weight": 3.0,
                },
                "index": {
                    "source_right": dex_frames.get("right_index_tip", "right_hand_index_1_link"),
                    "target_right": right["fingertip_links"]["index"],
                    "source_left": dex_frames.get("left_index_tip", "left_hand_index_1_link"),
                    "target_left": left["fingertip_links"]["index"],
                    "weight": 2.5,
                },
                "middle": {
                    "source_right": dex_frames.get("right_middle_tip", "right_hand_middle_1_link"),
                    "target_right": right["fingertip_links"]["middle"],
                    "source_left": dex_frames.get("left_middle_tip", "left_hand_middle_1_link"),
                    "target_left": left["fingertip_links"]["middle"],
                    "weight": 1.5,
                },
            },
            "joint_regularization": {
                "weight": 0.05,
                "neutral_pose": "brainco_default_open",
            },
            "constraints": {
                "joint_limits": True,
                "max_iterations": 100,
                "tolerance": 1.0e-4,
            },
            "action_mapping": {
                "source_dof": 14,
                "target_dof_actuated": 6,
                "source_type": "joint_position",
                "target_type": "joint_position",
                "note": "BrainCo distal joints are mimic-coupled; command 6 actuated joints per hand.",
            },
        },
        "dof_summary": {
            "dex3_per_hand": 7,
            "brainco_actuated_per_hand": 6,
            "brainco_revolute_per_hand": len(right["all_revolute_joints"]),
        },
    }

    OUT.write_text(yaml.dump(mapping, sort_keys=False))
    print(f"Updated {OUT}")


if __name__ == "__main__":
    main()
