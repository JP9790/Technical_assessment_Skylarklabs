"""Shared BrainCo standalone hand scene for Isaac Sim tests."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

JP_TEST_ROOT = Path(__file__).resolve().parents[2]


def brainco_usd(hand: str) -> str:
    return str(JP_TEST_ROOT / "assets/brainco/revo2_system/usd" / f"revo2_{hand}.usd")


def make_hand_scene_cfg(
    hand: str,
    *,
    with_ground: bool = True,
    with_camera: bool = False,
    with_task_prop: bool = False,
) -> type:
    usd = brainco_usd(hand)

    @configclass
    class BraincoHandSceneCfg(InteractiveSceneCfg):
        robot: ArticulationCfg = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/BraincoHand",
            spawn=sim_utils.UsdFileCfg(
                usd_path=usd,
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

    if with_ground:
        BraincoHandSceneCfg.ground = AssetBaseCfg(
            prim_path="/World/Ground",
            spawn=sim_utils.GroundPlaneCfg(),
        )

    if with_task_prop:
        BraincoHandSceneCfg.cylinder = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cylinder",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.05, 0.62),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
            spawn=sim_utils.CylinderCfg(
                radius=0.018,
                height=0.35,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
            ),
        )

    if with_camera:
        BraincoHandSceneCfg.front_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_camera",
            update_period=0.005,
            height=480,
            width=640,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=18.0,
                focus_distance=400.0,
                horizontal_aperture=24.0,
                clipping_range=(0.05, 5.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="ros",
            ),
        )
        BraincoHandSceneCfg.light = AssetBaseCfg(
            prim_path="/World/Light",
            spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.85)),
        )

    return BraincoHandSceneCfg
