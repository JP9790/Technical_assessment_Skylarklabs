#!/usr/bin/env python3
"""Audit source-policy candidates for Project 1 Stage A."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

JP_TEST_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = JP_TEST_ROOT / "results" / "phase0"

CANDIDATES = [
    {
        "name": "xr_teleoperate",
        "url": "https://github.com/unitreerobotics/xr_teleoperate",
        "artifact": "demonstration_dataset",
        "checkpoint_verified": False,
        "sim_compatible": "unitree_sim_isaaclab replay",
        "recommended": True,
        "blocker": None,
    },
    {
        "name": "dex3_rl_manipulation",
        "url": "https://github.com/PabloKevin/dex3_rl_manipulation",
        "artifact": "rl_checkpoint",
        "checkpoint_verified": False,
        "sim_compatible": "Isaac Lab G1 Dex3",
        "recommended": False,
        "blocker": "Public checkpoint link not confirmed",
    },
    {
        "name": "unitree_sim_isaaclab",
        "url": "https://github.com/unitreerobotics/unitree_sim_isaaclab",
        "artifact": "sim_environment",
        "checkpoint_verified": False,
        "sim_compatible": "self",
        "recommended": False,
        "blocker": "Not a pretrained policy",
    },
]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "provisional_selection": "xr_teleoperate_demos",
        "candidates": CANDIDATES,
        "next_steps": [
            "Clone xr_teleoperate and obtain or record a Dex3 pick-place demo",
            "Verify dex3_rl_manipulation for downloadable checkpoint",
            "Record exact observation/action dims after first sim launch (Phase 1)",
        ],
    }
    out = RESULTS_DIR / "source_policy_audit.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
