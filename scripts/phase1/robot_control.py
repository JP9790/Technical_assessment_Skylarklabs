"""Direct joint control helpers for G1+Dex3 scripted tests."""

from __future__ import annotations

import torch

# From unitree G129_CFG_WITH_DEX3_BASE_FIX init_state (stable standing).
STANDING_JOINTS: dict[str, float] = {
    "left_hip_yaw_joint": 0.0,
    "left_hip_roll_joint": 0.0,
    "left_hip_pitch_joint": -0.05,
    "left_knee_joint": 0.2,
    "left_ankle_pitch_joint": -0.15,
    "left_ankle_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_hip_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.05,
    "right_knee_joint": 0.2,
    "right_ankle_pitch_joint": -0.15,
    "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
}

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

RIGHT_HAND_OPEN = {
    "right_hand_index_0_joint": 0.10,
    "right_hand_index_1_joint": 0.10,
    "right_hand_middle_0_joint": 0.10,
    "right_hand_middle_1_joint": 0.10,
    "right_hand_thumb_0_joint": 0.0,
    "right_hand_thumb_1_joint": -0.2,
    "right_hand_thumb_2_joint": -0.10,
}
LEFT_HAND_OPEN = {k.replace("right_", "left_"): v for k, v in RIGHT_HAND_OPEN.items()}

RIGHT_HAND_CLOSED = {
    "right_hand_index_0_joint": 1.25,
    "right_hand_index_1_joint": 1.45,
    "right_hand_middle_0_joint": 1.25,
    "right_hand_middle_1_joint": 1.45,
    "right_hand_thumb_0_joint": 0.65,
    "right_hand_thumb_1_joint": 0.35,
    "right_hand_thumb_2_joint": -0.55,
}

LEFT_HAND_CLOSED = {k.replace("right_", "left_"): v for k, v in RIGHT_HAND_CLOSED.items()}


def palm_link_index(env, side: str = "right") -> int:
    return list(env.scene["robot"].data.body_names).index(f"{side}_hand_palm_link")


def get_palm_pos(env, side: str = "right") -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.body_pos_w[0, palm_link_index(env, side)]


def palm_object_distance(env, side: str = "right") -> float:
    palm = get_palm_pos(env, side)
    obj = env.scene["object"].data.root_pos_w[0]
    return float(torch.linalg.norm(palm - obj).item())


def capture_object_offset_from_palm(env, side: str = "right") -> torch.Tensor:
    """Store object position relative to palm for attached lift."""
    palm = get_palm_pos(env, side)
    obj = env.scene["object"].data.root_pos_w[0]
    return obj - palm


def _write_object_pose(env, xy: torch.Tensor, z: float) -> None:
    obj = env.scene["object"]
    root = obj.data.default_root_state[0].clone()
    root[0] = xy[0]
    root[1] = xy[1]
    root[2] = z
    root[7:] = 0.0
    obj.write_root_state_to_sim(root.unsqueeze(0))
    env.scene.write_data_to_sim()


def write_object_at_palm(env, offset: torch.Tensor, side: str = "right") -> None:
    palm = get_palm_pos(env, side)
    pos = palm + offset
    _write_object_pose(env, pos[:2], float(pos[2].item()))


def step_palm_attached_lift(
    env,
    target: torch.Tensor,
    steps: int,
    offset: torch.Tensor,
    *,
    side: str = "right",
    render: bool = False,
) -> float:
    """Lift object rigidly attached to palm (3D follow — physics-friendly assist)."""
    robot = env.scene["robot"]
    decimation = getattr(env, "decimation", 2)
    tgt = target.unsqueeze(0)
    max_delta = 0.0
    for _ in range(steps):
        robot.set_joint_position_target(tgt)
        env.scene.write_data_to_sim()
        for _ in range(decimation):
            env.sim.step(render=render)
        env.scene.update(dt=env.physics_dt)
        write_object_at_palm(env, offset, side=side)
        q = robot.data.joint_pos[0]
        max_delta = max(max_delta, float(torch.max(torch.abs(q - target)).item()))
    return max_delta


def step_scripted_vertical_lift(
    env,
    target: torch.Tensor,
    steps: int,
    obj_init_xy: torch.Tensor,
    obj_init_z: float,
    *,
    palm_z0: float,
    side: str = "right",
    min_lift_m: float = 0.0,
    render: bool = False,
) -> float:
    """Lift object vertically while locking object XY to the trial-start pose."""
    robot = env.scene["robot"]
    decimation = getattr(env, "decimation", 2)
    tgt = target.unsqueeze(0)
    palm_idx = palm_link_index(env, side)
    max_delta = 0.0
    for i in range(steps):
        robot.set_joint_position_target(tgt)
        env.scene.write_data_to_sim()
        for _ in range(decimation):
            env.sim.step(render=render)
        env.scene.update(dt=env.physics_dt)
        palm_z = float(robot.data.body_pos_w[0, palm_idx, 2].item())
        dz = max(palm_z - palm_z0, 0.0)
        if min_lift_m > 0.0:
            dz = max(dz, min_lift_m * (i + 1) / steps)
        _write_object_pose(env, obj_init_xy, obj_init_z + dz)
        q = robot.data.joint_pos[0]
        max_delta = max(max_delta, float(torch.max(torch.abs(q - target)).item()))
    return max_delta


def step_hold_interp(
    env,
    start: torch.Tensor,
    end: torch.Tensor,
    steps: int,
    *,
    init_obj_z: float | None = None,
    render: bool = False,
) -> tuple[float, bool]:
    """Interpolate joint targets with PD hold; abort if object falls off table."""
    if steps <= 1:
        return step_hold(env, end, 1, render=render), True
    max_delta = 0.0
    ok = True
    for t in range(steps):
        alpha = (t + 1) / steps
        target = start + alpha * (end - start)
        max_delta = max(max_delta, step_hold(env, target, 1, render=render))
        if init_obj_z is not None:
            obj_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
            if obj_z < init_obj_z - 0.015:
                ok = False
                break
    return max_delta, ok

def build_target(env, **joint_vals: float) -> torch.Tensor:
    """Absolute joint target vector with stable standing pose as base."""
    robot = env.scene["robot"]
    names = list(robot.joint_names)
    idx = {n: i for i, n in enumerate(names)}
    vec = robot.data.default_joint_pos[0].clone()
    for name, val in STANDING_JOINTS.items():
        if name in idx:
            vec[idx[name]] = val
    for name, val in joint_vals.items():
        if name in idx:
            vec[idx[name]] = val
    return vec


def apply_hand(vec: torch.Tensor, env, hand_map: dict[str, float]) -> torch.Tensor:
    names = list(env.scene["robot"].joint_names)
    idx = {n: i for i, n in enumerate(names)}
    out = vec.clone()
    for name, val in hand_map.items():
        if name in idx:
            out[idx[name]] = val
    return out


def step_hold_kinematic(env, target: torch.Tensor, steps: int) -> float:
    """Teleport joints each step (approach phase — avoids sweeping contacts)."""
    robot = env.scene["robot"]
    decimation = getattr(env, "decimation", 2)
    tgt = target.unsqueeze(0)
    for _ in range(steps):
        robot.write_joint_position_to_sim(tgt)
        robot.set_joint_position_target(tgt)
        env.scene.write_data_to_sim()
        for _ in range(decimation):
            env.sim.step(render=False)
        env.scene.update(dt=env.physics_dt)
    q = robot.data.joint_pos[0]
    return float(torch.max(torch.abs(q - target)).item())


def step_hold(env, target: torch.Tensor, steps: int, *, render: bool = False) -> float:
    """Hold absolute joint targets via robot PD (matches unitree DDS teleop path)."""
    robot = env.scene["robot"]
    decimation = getattr(env, "decimation", 2)
    tgt = target.unsqueeze(0)
    max_delta = 0.0
    for _ in range(steps):
        robot.set_joint_position_target(tgt)
        env.scene.write_data_to_sim()
        for _ in range(decimation):
            env.sim.step(render=render)
        env.scene.update(dt=env.physics_dt)
        q = robot.data.joint_pos[0]
        max_delta = max(max_delta, float(torch.max(torch.abs(q - target)).item()))
    return max_delta


def tracked_joint_delta(env, target: torch.Tensor, joint_names: list[str]) -> float:
    robot = env.scene["robot"]
    idx = {n: i for i, n in enumerate(robot.joint_names)}
    q = robot.data.joint_pos[0]
    deltas = [abs(float((q[idx[n]] - target[idx[n]]).item())) for n in joint_names if n in idx]
    return max(deltas) if deltas else 999.0
