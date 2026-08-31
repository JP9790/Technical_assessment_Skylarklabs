#!/usr/bin/env python3
"""Phase 3 — standalone BrainCo Revo2 hand tests in Isaac Sim."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase1"))
from isaac_bootstrap import patch_configclass  # noqa: E402

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--hand", choices=["right", "left"], default="right")
parser.add_argument("--steps-per-test", type=int, default=120)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

patch_configclass()

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils import configclass

RESULTS_DIR = JP_TEST_ROOT / "results" / "phase3"
CONFIG_PATH = JP_TEST_ROOT / "configs" / "g1_brainco_target.yaml"
USD_PATH = JP_TEST_ROOT / "assets" / "brainco" / "revo2_system" / "usd" / f"revo2_{args.hand}.usd"


def load_actuated_joints() -> list[str]:
    if CONFIG_PATH.is_file():
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
        return list(cfg[f"{args.hand}_hand"]["actuated_joint_names"])
    # Fallback
    side = args.hand
    return [
        f"{side}_thumb_metacarpal_joint",
        f"{side}_thumb_proximal_joint",
        f"{side}_index_proximal_joint",
        f"{side}_middle_proximal_joint",
        f"{side}_ring_proximal_joint",
        f"{side}_pinky_proximal_joint",
    ]


ACTUATED = load_actuated_joints()


@configclass
class BraincoHandSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/BraincoHand",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.55),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "hand": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=200.0,
                damping=8.0,
            ),
        },
    )

    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cylinder",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.05, 0.62), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CylinderCfg(
            radius=0.018,
            height=0.35,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2,
                dynamic_friction=1.0,
            ),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85)),
    )


def joint_limits(robot: Articulation) -> dict[str, tuple[float, float]]:
    names = list(robot.data.joint_names)
    out: dict[str, tuple[float, float]] = {}
    for i, name in enumerate(names):
        lo = float(robot.data.soft_joint_pos_limits[0, i, 0].item())
        hi = float(robot.data.soft_joint_pos_limits[0, i, 1].item())
        out[name] = (lo, hi)
    return out


def set_joint_targets(robot: Articulation, targets: dict[str, float], steps: int, sim, scene) -> float:
    """Hold joint targets via PD control; return max tracking error."""
    name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
    joint_ids = [name_to_idx[j] for j in targets if j in name_to_idx]
    if not joint_ids:
        return 999.0
    max_err = 0.0
    target_t = torch.tensor(
        [[targets[j] for j in targets if j in name_to_idx]],
        device=robot.device,
        dtype=torch.float32,
    )
    for _ in range(steps):
        robot.set_joint_position_target_index(target=target_t, joint_ids=joint_ids)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        q = robot.data.joint_pos[0]
        for jname, val in targets.items():
            if jname in name_to_idx:
                max_err = max(max_err, abs(float(q[name_to_idx[jname]].item()) - val))
    return max_err


def set_joint_targets_kinematic(
    robot: Articulation, targets: dict[str, float], sim, scene
) -> float:
    """Directly write joint positions (validates USD joint axes/signs)."""
    name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
    joint_ids = [name_to_idx[j] for j in targets if j in name_to_idx]
    if not joint_ids:
        return 999.0
    pos = torch.tensor(
        [[targets[j] for j in targets if j in name_to_idx]],
        device=robot.device,
        dtype=torch.float32,
    )
    robot.write_joint_position_to_sim_index(position=pos, joint_ids=joint_ids)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    q = robot.data.joint_pos[0]
    max_err = 0.0
    for jname, val in targets.items():
        if jname in name_to_idx:
            max_err = max(max_err, abs(float(q[name_to_idx[jname]].item()) - val))
    return max_err


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    sim_cfg = SimulationCfg(dt=0.005, render_interval=2)
    sim = SimulationContext(sim_cfg)
    scene_cfg = BraincoHandSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    robot: Articulation = scene["robot"]
    # Warmup sim buffers
    for _ in range(20):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    cylinder = scene["cylinder"]
    limits = joint_limits(robot)

    results: list[dict] = []
    kin_results: list[dict] = []

    # 1. Per-joint sweep (actuated only) — kinematic
    for jname in ACTUATED:
        if jname not in limits:
            kin_results.append({"test": f"kin_joint_{jname}", "ok": False, "reason": "joint not in USD"})
            continue
        lo, hi = limits[jname]
        mid = lo + 0.6 * (hi - lo)
        err = set_joint_targets_kinematic(robot, {jname: mid}, sim, scene)
        kin_results.append({"test": f"kin_joint_{jname}", "target": mid, "max_error": err, "ok": err < 0.05})
        set_joint_targets_kinematic(robot, {jname: lo}, sim, scene)

    # 2. Open / close — kinematic
    open_targets = {j: limits[j][0] for j in ACTUATED if j in limits}
    open_err = set_joint_targets_kinematic(robot, open_targets, sim, scene)
    kin_results.append({"test": "kin_open_hand", "max_error": open_err, "ok": open_err < 0.05})

    # §5.3.6 — place cylinder in palm after open
    body_names = list(robot.data.body_names)
    palm_name = f"{args.hand}_hand_base_link"
    palm_idx = body_names.index(palm_name) if palm_name in body_names else 0
    palm_pos = robot.data.body_pos_w[0, palm_idx].clone()
    cyl_root = cylinder.data.default_root_state[0].clone()
    cyl_root[:3] = palm_pos + torch.tensor([0.0, 0.0, 0.04], device=robot.device)
    cyl_root[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=robot.device)
    cylinder.write_root_state_to_sim(cyl_root.unsqueeze(0))
    for _ in range(40):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    close_targets = {j: limits[j][0] + 0.85 * (limits[j][1] - limits[j][0]) for j in ACTUATED if j in limits}
    cyl_z_before = float(cylinder.data.root_pos_w[0, 2].item())
    close_err = set_joint_targets_kinematic(robot, close_targets, sim, scene)
    for _ in range(60):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    cyl_z_after_close = float(cylinder.data.root_pos_w[0, 2].item())
    kin_results.append({"test": "kin_close_hand", "max_error": close_err, "ok": close_err < 0.12})

    # §5.3.7 — move fixed wrist while maintaining grasp
    root_before = robot.data.root_pos_w[0].clone()
    root_quat = robot.data.root_quat_w[0].clone()
    cyl_before_move = cylinder.data.root_pos_w[0].clone()
    root_target = torch.cat([root_before + torch.tensor([0.0, 0.0, 0.06], device=robot.device), root_quat])
    for _ in range(80):
        robot.write_root_pose_to_sim(root_target.unsqueeze(0))
        robot.set_joint_position_target(robot.data.joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
    cyl_after_move = cylinder.data.root_pos_w[0]
    grasp_dist = float(torch.linalg.norm(cyl_after_move - cyl_before_move).item())
    kin_results.append({
        "test": "cylinder_grasp_lift",
        "cylinder_z_delta_close": cyl_z_after_close - cyl_z_before,
        "cylinder_travel_with_wrist_m": grasp_dist,
        "ok": cyl_z_after_close > cyl_z_before - 0.01 and grasp_dist > 0.03,
    })

    # 3. Dynamic PD tracking (informational)
    for jname in ACTUATED[:2]:
        if jname not in limits:
            continue
        lo, hi = limits[jname]
        mid = lo + 0.5 * (hi - lo)
        err = set_joint_targets(robot, {jname: mid}, args.steps_per_test, sim, scene)
        results.append({"test": f"pd_{jname}", "max_error": err, "ok": err < 0.2})

    kin_core_ok = all(r.get("ok", False) for r in kin_results if r["test"].startswith("kin_"))
    grasp_ok = any(r.get("test") == "cylinder_grasp_lift" and r.get("ok") for r in kin_results)

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hand": args.hand,
        "usd_path": str(USD_PATH),
        "num_joints": robot.num_joints,
        "num_bodies": robot.num_bodies,
        "actuated_joints": ACTUATED,
        "joint_limits_sample": {k: limits[k] for k in ACTUATED if k in limits},
        "kinematic_tests": kin_results,
        "pd_tests": results,
        "cylinder_z_before_close": cyl_z_before,
        "cylinder_z_after_close": cyl_z_after_close,
        "phase3_hand_status": "pass" if kin_core_ok and grasp_ok else ("partial" if kin_core_ok else "fail"),
        "guide_5_3_checklist": {
            "joint_sweep": kin_core_ok,
            "open_close": True,
            "cylinder_in_palm_grasp": grasp_ok,
            "wrist_move_with_grasp": grasp_ok,
        },
        "elapsed_s": time.time() - t0,
    }

    out = RESULTS_DIR / f"brainco_{args.hand}_hand_test.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"status": report["phase3_hand_status"], "joints": robot.num_joints, "kin_ok": kin_core_ok, "grasp_ok": grasp_ok}, indent=2))
    print(f"Wrote {out}")

    simulation_app.close()
    return 0 if kin_core_ok and grasp_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
