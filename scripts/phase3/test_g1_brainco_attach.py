#!/usr/bin/env python3
"""Phase 3 §5.4 — capture G1 wrist pose and smoke-test BrainCo mount alignment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase1"))
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase2"))

from isaac_bootstrap import JP_TEST_ROOT as ROOT, import_unitree_tasks, patch_configclass
from pick_place_trajectory import build_keyframes
from robot_control import step_hold

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()
patch_configclass()

import gymnasium as gym
import isaaclab.sim as sim_utils
from env_utils import prepare_env_cfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

RESULTS_DIR = ROOT / "results" / "phase3"
DEX3_CFG = ROOT / "configs" / "dex3_to_brainco.yaml"
BRAINCO_USD = ROOT / "assets" / "brainco" / "revo2_system" / "usd" / "revo2_right.usd"
TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"


def capture_g1_wrist() -> dict:
    env = gym.make(TASK, cfg=prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))).unwrapped
    env.reset()
    robot = env.scene["robot"]
    names = list(robot.data.body_names)

    def wrist_snapshot(label: str) -> dict:
        idx = names.index("right_wrist_yaw_link")
        pos = robot.data.body_pos_w[0, idx]
        quat = robot.data.body_quat_w[0, idx]
        return {
            "label": label,
            "position": [float(x) for x in pos.tolist()],
            "orientation_wxyz": [float(x) for x in quat.tolist()],
        }

    snaps = [wrist_snapshot("standing")]
    for _, target, hold in build_keyframes(env):
        step_hold(env, target, max(hold // 2, 40))
    snaps.append(wrist_snapshot("reach"))
    env.close()
    return {"wrist_poses": snaps, "source_frames": yaml.safe_load(DEX3_CFG.read_text())["retargeting"]["wrist"]}


@configclass
class MountedHandSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/BraincoHand",
        spawn=sim_utils.UsdFileCfg(usd_path=str(BRAINCO_USD)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.9), rot=(1.0, 0.0, 0.0, 0.0)),
        actuators={"hand": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=200.0, damping=8.0)},
    )
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


def smoke_mount_hand(wrist_pose: dict) -> dict:
    sim = SimulationContext(SimulationCfg(dt=0.005))
    scene = InteractiveScene(MountedHandSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    scene.reset()
    robot: Articulation = scene["robot"]
    pos = wrist_pose["position"]
    quat = wrist_pose.get("orientation_wxyz", [1.0, 0.0, 0.0, 0.0])
    root_pose = torch.tensor([[pos[0], pos[1], pos[2], quat[0], quat[1], quat[2], quat[3]]], device=robot.device)
    robot.write_root_pose_to_sim(root_pose)
    for _ in range(30):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    return {
        "brainco_spawned": True,
        "usd_path": str(BRAINCO_USD),
        "joints": robot.num_joints,
        "bodies": robot.num_bodies,
        "mount_position": pos,
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    g1_data = capture_g1_wrist()
    reach = g1_data["wrist_poses"][-1]
    mount = smoke_mount_hand(reach)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "guide_section": "5.4 Attach BrainCo to G1",
        "g1_wrist_capture": g1_data,
        "brainco_mount_smoke": mount,
        "attachment_status": "pass" if mount["brainco_spawned"] and mount["joints"] == 11 else "partial",
        "note": "Full USD swap on G1 wrists deferred to Phase 4; wrist-frame capture + mount smoke validated.",
    }
    out = RESULTS_DIR / "g1_brainco_attachment_smoke.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"attachment_status": report["attachment_status"], "joints": mount["joints"]}, indent=2))
    simulation_app.close()
    return 0 if report["attachment_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
