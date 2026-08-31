#!/usr/bin/env python3
"""Audit Stage A source-policy candidates and write Phase 2 selection record."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = JP_TEST_ROOT / "results" / "phase2"
EXTERNAL = JP_TEST_ROOT / "external"


def git_head(repo: Path) -> str | None:
    if not (repo / ".git").exists():
        return None
    try:
        return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError:
        return None


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    xr = EXTERNAL / "xr_teleoperate"
    dex3_rl = EXTERNAL / "dex3_rl_manipulation"

    audit = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selected_source": "xr_teleoperate_sim_scripted_demos",
        "selection_status": "confirmed_provisional",
        "candidates": [
            {
                "name": "xr_teleoperate",
                "url": "https://github.com/unitreerobotics/xr_teleoperate",
                "local_path": str(xr) if xr.exists() else None,
                "commit": git_head(xr),
                "license": "Apache-2.0",
                "artifact_type": "demonstration_trajectory",
                "artifact_available": xr.exists(),
                "checkpoint_verified": False,
                "notes": "Cloned; public repo has no bundled pick-place episodes. Stage A uses sim-recorded data.json in compatible format.",
                "verdict": "selected",
            },
            {
                "name": "dex3_rl_manipulation",
                "url": "https://github.com/PabloKevin/dex3_rl_manipulation",
                "local_path": str(dex3_rl) if dex3_rl.exists() else None,
                "commit": git_head(dex3_rl),
                "artifact_type": "rl_policy_checkpoint",
                "artifact_available": False,
                "checkpoint_verified": False,
                "notes": "Repository not found (404) as of Phase 2 audit.",
                "verdict": "unavailable",
            },
            {
                "name": "G1_Dex3_ObjectPlacement_Dataset",
                "url": "https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ObjectPlacement_Dataset",
                "artifact_type": "lerobot_parquet_dataset",
                "artifact_available": True,
                "checkpoint_verified": True,
                "notes": "210 teleop episodes, 28-dim arm+hand state/action; different task (object placement) — reference only.",
                "verdict": "reference_motion_prior",
            },
        ],
        "stage_a_interface": {
            "embodiment": "Unitree G1 29DoF + Dex3",
            "task": "Isaac-PickPlace-Cylinder-G129-Dex3-Joint",
            "observation_space": "joint_state (43) + camera_image (optional)",
            "action_space": "joint_position_offset (43,) via JointPositionAction",
            "control_frequency_hz": 100.0,
            "demo_format": "xr_teleoperate episode_xxxx/data.json",
        },
    }

    out = RESULTS_DIR / "source_audit.json"
    out.write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
