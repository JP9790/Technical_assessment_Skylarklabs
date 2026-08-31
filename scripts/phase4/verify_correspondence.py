#!/usr/bin/env python3
"""Phase 4 — verify Dex3↔BrainCo correspondence frames exist on both models."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
CORR_PATH = JP_TEST_ROOT / "configs" / "correspondence_table.yaml"
DEX3_CFG = JP_TEST_ROOT / "configs" / "g1_dex3_source.yaml"
BRAINCO_CFG = JP_TEST_ROOT / "configs" / "g1_brainco_target.yaml"
OUT = JP_TEST_ROOT / "results" / "phase4" / "correspondence_verification.json"

parser = argparse.ArgumentParser()
parser.add_argument("--sim", action="store_true", help="Also verify frames in Isaac Sim")
args_pre, _ = parser.parse_known_args()

simulation_app = None
if args_pre.sim:
    sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase1"))
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
else:
    args = parser.parse_args()


def load_link_sets() -> tuple[set[str], set[str]]:
    dex3 = yaml.safe_load(DEX3_CFG.read_text())
    brainco = yaml.safe_load(BRAINCO_CFG.read_text())
    dex_links = set(dex3.get("links", {}).get("all_names", []))
    bc_links: set[str] = set()
    for side in ("right_hand", "left_hand"):
        hand = brainco[side]
        bc_links.add(hand["base_link"])
        bc_links.add(hand["palm_link"])
        bc_links.update(hand.get("fingertip_links", {}).values())
        bc_links.update(hand.get("touch_links", []) or [])
    return dex_links, bc_links


def _is_link_name(name: str) -> bool:
    return name.endswith("_link") or name.endswith("_tip")


def collect_required_frames(corr: dict) -> tuple[set[str], set[str]]:
    dex_req: set[str] = set()
    bc_req: set[str] = set()
    mount = corr.get("mount_transform", {})
    for key in ("source_mount_link_right", "source_mount_link_left"):
        if mount.get(key):
            dex_req.add(mount[key])
    for key in ("target_mount_link_right", "target_mount_link_left"):
        if mount.get(key):
            bc_req.add(mount[key])
    for row in corr.get("rows", []):
        for k, v in row.items():
            if k.startswith("source_dex3") and isinstance(v, str) and _is_link_name(v):
                dex_req.add(v)
        for k, v in row.items():
            if k.startswith("target_brainco") and isinstance(v, str) and _is_link_name(v):
                bc_req.add(v)
    return dex_req, bc_req


def verify_sim_frames(device: str) -> dict:
    from isaac_bootstrap import import_unitree_tasks, patch_configclass

    import_unitree_tasks()
    patch_configclass()

    import gymnasium as gym
    from env_utils import prepare_env_cfg
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"
    env = gym.make(TASK, cfg=prepare_env_cfg(parse_env_cfg(TASK, device=device, num_envs=1))).unwrapped
    env.reset()
    dex_bodies = set(env.scene["robot"].data.body_names)
    env.close()

    sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase3"))
    from brainco_hand_scene import make_hand_scene_cfg

    HandCfg = make_hand_scene_cfg("right")
    sim = SimulationContext(SimulationCfg(dt=0.005, device=device))
    scene = InteractiveScene(HandCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    bc_bodies = set(scene["robot"].data.body_names)
    return {"dex3_bodies_in_sim": sorted(dex_bodies), "brainco_bodies_in_sim": sorted(bc_bodies)}


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    corr = yaml.safe_load(CORR_PATH.read_text())
    dex_links, bc_links = load_link_sets()
    dex_req, bc_req = collect_required_frames(corr)

    dex_missing = sorted(dex_req - dex_links)
    bc_missing = sorted(bc_req - bc_links)

    report: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "correspondence_file": str(CORR_PATH),
        "dex3_required_frames": sorted(dex_req),
        "brainco_required_frames": sorted(bc_req),
        "dex3_missing": dex_missing,
        "brainco_missing": bc_missing,
        "yaml_verification_pass": not dex_missing and not bc_missing,
    }

    if args.sim:
        sim_data = verify_sim_frames(args.device)
        report["sim_verification"] = sim_data
        dex_bodies = set(sim_data["dex3_bodies_in_sim"])
        bc_bodies = set(sim_data["brainco_bodies_in_sim"])
        report["sim_dex3_missing"] = sorted(dex_req - dex_bodies)
        report["sim_brainco_missing"] = sorted(bc_req - bc_bodies)
        report["sim_verification_pass"] = not report["sim_dex3_missing"] and not report["sim_brainco_missing"]
        report["phase4_status"] = "pass" if report["yaml_verification_pass"] and report["sim_verification_pass"] else "fail"
    else:
        report["phase4_status"] = "pass" if report["yaml_verification_pass"] else "fail"

    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps({"phase4_status": report["phase4_status"], "dex3_missing": dex_missing, "brainco_missing": bc_missing}, indent=2))

    if simulation_app is not None:
        simulation_app.close()
    return 0 if report["phase4_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
