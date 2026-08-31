#!/usr/bin/env python3
"""Parse BrainCo Revo2 URDF and write target interface YAML (no Isaac required)."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
BRAINCO_ROOT = JP_TEST_ROOT / "external" / "brainco-description" / "revo2_system"
URDF_DIR = BRAINCO_ROOT / "urdf"
RESULTS = JP_TEST_ROOT / "results" / "phase3"
CONFIG_OUT = JP_TEST_ROOT / "configs" / "g1_brainco_target.yaml"

# Hardware API joints (6 DoF per hand) per xr_teleoperate / BrainCo docs.
ACTUATED_JOINTS = {
    "right": [
        "right_thumb_metacarpal_joint",
        "right_thumb_proximal_joint",
        "right_index_proximal_joint",
        "right_middle_proximal_joint",
        "right_ring_proximal_joint",
        "right_pinky_proximal_joint",
    ],
    "left": [
        "left_thumb_metacarpal_joint",
        "left_thumb_proximal_joint",
        "left_index_proximal_joint",
        "left_middle_proximal_joint",
        "left_ring_proximal_joint",
        "left_pinky_proximal_joint",
    ],
}


def parse_urdf(path: Path) -> dict:
    root = ET.parse(path).getroot()
    joints: dict[str, dict] = {}
    links: list[str] = []
    mimic: dict[str, dict] = {}

    for link in root.findall("joint/.."):
        pass

    for link in root.findall("link"):
        name = link.get("name")
        if name:
            links.append(name)

    for joint in root.findall("joint"):
        jname = joint.get("name")
        jtype = joint.get("type")
        parent = child = None
        limit = {"lower": 0.0, "upper": 0.0, "effort": 0.0, "velocity": 0.0}
        origin = {"xyz": [0, 0, 0], "rpy": [0, 0, 0]}
        axis = [0, 0, 1]

        for child_el in joint:
            tag = child_el.tag
            if tag == "parent":
                parent = child_el.get("link")
            elif tag == "child":
                child = child_el.get("link")
            elif tag == "limit":
                limit = {
                    "lower": float(child_el.get("lower", 0)),
                    "upper": float(child_el.get("upper", 0)),
                    "effort": float(child_el.get("effort", 0)),
                    "velocity": float(child_el.get("velocity", 0)),
                }
            elif tag == "origin":
                origin = {
                    "xyz": [float(x) for x in child_el.get("xyz", "0 0 0").split()],
                    "rpy": [float(x) for x in child_el.get("rpy", "0 0 0").split()],
                }
            elif tag == "axis":
                axis = [float(x) for x in child_el.get("xyz", "0 0 1").split()]
            elif tag == "mimic":
                mimic[jname] = {
                    "joint": child_el.get("joint"),
                    "multiplier": float(child_el.get("multiplier", 1)),
                    "offset": float(child_el.get("offset", 0)),
                }

        joints[jname] = {
            "type": jtype,
            "parent": parent,
            "child": child,
            "limit": limit,
            "origin": origin,
            "axis": axis,
        }

    revolute = [n for n, j in joints.items() if j["type"] == "revolute"]
    fixed = [n for n, j in joints.items() if j["type"] == "fixed"]

    return {
        "links": links,
        "joints": joints,
        "revolute_joint_names": revolute,
        "fixed_joint_names": fixed,
        "mimic_joints": mimic,
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    right = parse_urdf(URDF_DIR / "revo2_right.urdf")
    left = parse_urdf(URDF_DIR / "revo2_left.urdf")

    def side_summary(side: str, parsed: dict) -> dict:
        actuated = ACTUATED_JOINTS[side]
        limits = {n: parsed["joints"][n]["limit"] for n in actuated if n in parsed["joints"]}
        tips = [l for l in parsed["links"] if l.endswith("_tip")]
        touch = [l for l in parsed["links"] if l.endswith("_touch")]
        return {
            "usd_path": f"assets/brainco/revo2_system/usd/revo2_{side}.usd",
            "urdf_path": f"external/brainco-description/revo2_system/urdf/revo2_{side}.urdf",
            "base_link": f"{side}_hand_base_link",
            "wrist_mount_link": f"{side}_hand_base_link",
            "palm_link": f"{side}_hand_base_link",
            "fingertip_links": {
                "thumb": f"{side}_thumb_tip",
                "index": f"{side}_index_tip",
                "middle": f"{side}_middle_tip",
                "ring": f"{side}_ring_tip",
                "pinky": f"{side}_pinky_tip",
            },
            "touch_links": touch,
            "actuated_joint_names": actuated,
            "all_revolute_joints": parsed["revolute_joint_names"],
            "joint_limits": limits,
            "mimic_joints": parsed["mimic_joints"],
            "total_links": len(parsed["links"]),
            "hand_dof_actuated": len(actuated),
        }

    repo_commit = ""
    git_dir = JP_TEST_ROOT / "external" / "brainco-description" / ".git"
    if git_dir.exists():
        import subprocess

        repo_commit = subprocess.check_output(
            ["git", "-C", str(BRAINCO_ROOT.parent), "rev-parse", "HEAD"], text=True
        ).strip()

    interface = {
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "embodiment": "BrainCo Revo2 Touch (official brainco-description)",
        "source_repository": "https://github.com/BrainCoTech/brainco-description",
        "source_commit": repo_commit,
        "license": "Check repository LICENSE at clone time",
        "deviation_note": (
            "Assessment vendor bundle not found on legion; using BrainCoTech public "
            "revo2_system USD/URDF (Revo2 dexterous hand). Touch sensor links present in URDF."
        ),
        "control": {
            "hand_dof_per_side": 6,
            "total_hand_dof_bimanual": 12,
            "distal_joints": "mimic_coupled_to_proximal",
            "control_frequency_hz": 100.0,
        },
        "right_hand": side_summary("right", right),
        "left_hand": side_summary("left", left),
        "g1_attachment": {
            "status": "planned_phase3b",
            "replace_links": ["right_hand_*", "left_hand_*"],
            "mount_parent_links": ["right_wrist_yaw_link", "left_wrist_yaw_link"],
            "notes": "Swap Dex3 hand USD subtree on G1 29DoF; keep arm/body actuators unchanged.",
        },
    }

    import yaml

    CONFIG_OUT.write_text(yaml.dump(interface, sort_keys=False, default_flow_style=False))

    json_out = RESULTS / "brainco_interface.json"
    json_out.write_text(json.dumps(interface, indent=2))

    print(yaml.dump({"right_dof": interface["right_hand"]["hand_dof_actuated"],
                     "right_joints": interface["right_hand"]["actuated_joint_names"],
                     "config": str(CONFIG_OUT)}, sort_keys=False))
    print(f"Wrote {CONFIG_OUT}")


if __name__ == "__main__":
    main()
