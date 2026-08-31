#!/usr/bin/env python3
"""Merge correspondence_table.yaml into dex3_to_brainco.yaml (Phase 4)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

JP = Path(__file__).resolve().parents[2]
CORR = JP / "configs" / "correspondence_table.yaml"
DEX3_MAP = JP / "configs" / "dex3_joint_map.yaml"
BRAINCO = JP / "configs" / "g1_brainco_target.yaml"
OUT = JP / "configs" / "dex3_to_brainco.yaml"

# Neutral open poses for regularization (Phase 5 optimizer).
DEX3_NEUTRAL_OPEN = {
    "right_hand_index_0_joint": 0.10,
    "right_hand_index_1_joint": 0.10,
    "right_hand_middle_0_joint": 0.10,
    "right_hand_middle_1_joint": 0.10,
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": -0.2,
    "right_hand_thumb_2_joint": -0.10,
    "left_hand_index_0_joint": 0.10,
    "left_hand_index_1_joint": 0.10,
    "left_hand_middle_0_joint": 0.10,
    "left_hand_middle_1_joint": 0.10,
    "left_hand_thumb_0_joint": 0.0,
    "left_hand_thumb_1_joint": -0.2,
    "left_hand_thumb_2_joint": -0.10,
}


def brainco_neutral_open(brainco_cfg: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for side in ("right_hand", "left_hand"):
        for j in brainco_cfg[side]["actuated_joint_names"]:
            lo = brainco_cfg[side]["joint_limits"][j]["lower"]
            out[j] = float(lo)
    return out


def main() -> None:
    corr = yaml.safe_load(CORR.read_text())
    dex3 = yaml.safe_load(DEX3_MAP.read_text())
    brainco = yaml.safe_load(BRAINCO.read_text())
    frames = dex3.get("frames", {})
    rh, lh = brainco["right_hand"], brainco["left_hand"]

    merged = {
        "status": "phase4_complete",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_embodiment": corr["source_embodiment"],
        "target_embodiment": corr["target_embodiment"],
        "correspondence_table": "configs/correspondence_table.yaml",
        "target_asset": {
            "repository": brainco["source_repository"],
            "commit": brainco["source_commit"],
            "usd_right": rh["usd_path"],
            "usd_left": lh["usd_path"],
        },
        "mount_transform": corr["mount_transform"],
        "retargeting": {
            "method": "task_space_ik",
            "wrist": {
                "source_frame_right": frames["right_wrist"],
                "target_frame_right": rh["wrist_mount_link"],
                "source_frame_left": frames["left_wrist"],
                "target_frame_left": lh["wrist_mount_link"],
                "weight_position": 1.0,
                "weight_orientation": 1.0,
            },
            "palm": {
                "source_frame_right": frames["right_palm"],
                "target_frame_right": rh["palm_link"],
                "source_frame_left": frames["left_palm"],
                "target_frame_left": lh["palm_link"],
                "weight_position": 0.8,
                "weight_orientation": 0.8,
            },
            "fingertips": {
                "thumb": {
                    "source_right": frames["right_thumb_tip"],
                    "target_right": rh["fingertip_links"]["thumb"],
                    "source_left": frames["left_thumb_tip"],
                    "target_left": lh["fingertip_links"]["thumb"],
                    "weight": 3.0,
                },
                "index": {
                    "source_right": frames["right_index_tip"],
                    "target_right": rh["fingertip_links"]["index"],
                    "source_left": frames["left_index_tip"],
                    "target_left": lh["fingertip_links"]["index"],
                    "weight": 2.5,
                },
                "middle": {
                    "source_right": frames["right_middle_tip"],
                    "target_right": rh["fingertip_links"]["middle"],
                    "source_left": frames["left_middle_tip"],
                    "target_left": lh["fingertip_links"]["middle"],
                    "weight": 1.5,
                },
                "ring": {
                    "source_right": None,
                    "target_right": rh["fingertip_links"]["ring"],
                    "source_left": None,
                    "target_left": lh["fingertip_links"]["ring"],
                    "weight": 0.0,
                },
                "pinky": {
                    "source_right": None,
                    "target_right": rh["fingertip_links"]["pinky"],
                    "source_left": None,
                    "target_left": lh["fingertip_links"]["pinky"],
                    "weight": 0.0,
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
                "target_dof_actuated": 12,
                "source_type": "joint_position",
                "target_type": "joint_position",
                "retargeting_function": "task_space_ik_then_mimic_expand",
                "note": "Map 14-D Dex3 demo/policy hand slice → 12-D BrainCo actuated; mimic joints expanded in sim.",
            },
            "contacts": {
                "enabled": False,
                "source": "none_on_dex3_usd",
                "target_touch_links": [],
                "mapping": "deferred_phase5",
            },
        },
        "neutral_poses": {
            "dex3_open": DEX3_NEUTRAL_OPEN,
            "brainco_open": brainco_neutral_open(brainco),
        },
        "dof_summary": {
            "dex3_per_hand": 7,
            "brainco_actuated_per_hand": 6,
            "brainco_revolute_per_hand": len(rh["all_revolute_joints"]),
        },
    }

    OUT.write_text(yaml.dump(merged, sort_keys=False))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
