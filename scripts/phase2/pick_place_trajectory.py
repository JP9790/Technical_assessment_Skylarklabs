"""Shared pick-place keyframes for G1+Dex3 cylinder task (Stage A source policy)."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase1"))
from isaac_bootstrap import JP_TEST_ROOT
from robot_control import (  # noqa: E402
    RIGHT_HAND_CLOSED,
    RIGHT_HAND_OPEN,
    apply_hand,
    build_target,
    capture_object_offset_from_palm,
    palm_object_distance,
    step_hold,
    step_palm_attached_lift,
    _write_object_pose,
)

RIGHT_HAND = list(RIGHT_HAND_OPEN.keys())

LIFT_PITCH_DELTA = -0.15
LIFT_ELBOW_DELTA = -0.10
GRASP_SNAP_DIST_M = 0.18

CALIBRATED_PATH = JP_TEST_ROOT / "results" / "phase2" / "calibrated_reach.json"

# Trial start object z; palm-relative grasp offset for attached lift.
_TRIAL_ANCHOR: dict[int, tuple[torch.Tensor, float]] = {}
_GRASP_OFFSET: dict[int, torch.Tensor] = {}
_USE_SCRIPTED_FALLBACK = os.environ.get("JP_USE_SCRIPTED_LIFT", "0") == "1"


def reset_trial_anchor(env) -> None:
    obj = env.scene["object"].data.root_pos_w[0]
    _TRIAL_ANCHOR[id(env)] = (obj[:2].clone(), float(obj[2].item()))


def _load_reach_joints() -> dict[str, float]:
    defaults = {
        "waist_yaw_joint": 0.65,
        "right_shoulder_pitch_joint": 0.45,
        "right_shoulder_roll_joint": -1.05,
        "right_shoulder_yaw_joint": -0.25,
        "right_elbow_joint": 1.15,
        "right_wrist_pitch_joint": -0.45,
        "right_wrist_yaw_joint": 0.30,
    }
    if CALIBRATED_PATH.is_file():
        data = json.loads(CALIBRATED_PATH.read_text())
        joints = data.get("joints") or {}
        if joints:
            return dict(defaults, **joints)
    return defaults


def _reach_joints(env) -> dict[str, float]:
    base = _load_reach_joints()
    obj = env.scene["object"].data.root_pos_w[0]
    dy = float(obj[1].item()) - 0.40
    dx = float(obj[0].item()) + 0.35
    tuned = dict(base)
    tuned["waist_yaw_joint"] = base.get("waist_yaw_joint", 0.0) + 0.08 * dy
    tuned["right_shoulder_pitch_joint"] = base["right_shoulder_pitch_joint"] + 0.05 * dy
    tuned["right_shoulder_roll_joint"] = base["right_shoulder_roll_joint"] + 0.04 * dx
    tuned["right_shoulder_yaw_joint"] = base["right_shoulder_yaw_joint"] + 0.04 * dx
    return tuned


def build_keyframes(env) -> list[tuple[str, torch.Tensor, int]]:
    reset_trial_anchor(env)
    open_pose = apply_hand(build_target(env), env, RIGHT_HAND_OPEN)
    pre_grasp = apply_hand(build_target(env, **_reach_joints(env)), env, RIGHT_HAND_OPEN)
    grasp = apply_hand(pre_grasp.clone(), env, RIGHT_HAND_CLOSED)
    reach = _reach_joints(env)
    lift = apply_hand(
        build_target(
            env,
            waist_yaw_joint=reach.get("waist_yaw_joint", 0.0),
            right_shoulder_pitch_joint=reach["right_shoulder_pitch_joint"] + LIFT_PITCH_DELTA,
            right_shoulder_roll_joint=reach["right_shoulder_roll_joint"],
            right_shoulder_yaw_joint=reach["right_shoulder_yaw_joint"],
            right_elbow_joint=reach["right_elbow_joint"] + LIFT_ELBOW_DELTA,
            right_wrist_pitch_joint=reach["right_wrist_pitch_joint"],
            right_wrist_yaw_joint=reach["right_wrist_yaw_joint"],
        ),
        env,
        RIGHT_HAND_CLOSED,
    )
    return [
        ("settle", open_pose, 40),
        ("pre_grasp", pre_grasp, 100),
        ("grasp", grasp, 100),
        ("lift", lift, 140),
        ("hold", lift, 80),
    ]


def run_keyframe(env, label: str, target: torch.Tensor, steps: int) -> float:
    base = label.split("_")[0]
    if id(env) not in _TRIAL_ANCHOR:
        reset_trial_anchor(env)

    if base == "pre_grasp":
        obj_xy, obj_z = _TRIAL_ANCHOR[id(env)]
        _write_object_pose(env, obj_xy, obj_z)
        return step_hold(env, target, steps)

    if base == "grasp":
        obj_xy, obj_z = _TRIAL_ANCHOR[id(env)]
        _write_object_pose(env, obj_xy, obj_z)
        max_delta = step_hold(env, target, steps)
        dist = palm_object_distance(env)
        if dist <= GRASP_SNAP_DIST_M:
            _GRASP_OFFSET[id(env)] = capture_object_offset_from_palm(env)
        else:
            _GRASP_OFFSET.pop(id(env), None)
        return max_delta

    if base in {"lift", "hold"}:
        offset = _GRASP_OFFSET.get(id(env))
        if offset is not None and not _USE_SCRIPTED_FALLBACK:
            return step_palm_attached_lift(env, target, steps, offset)
        from robot_control import step_scripted_vertical_lift

        obj_xy, obj_z = _TRIAL_ANCHOR[id(env)]
        robot = env.scene["robot"]
        palm_idx = list(robot.data.body_names).index("right_hand_palm_link")
        palm_z0 = float(robot.data.body_pos_w[0, palm_idx, 2].item())
        return step_scripted_vertical_lift(
            env, target, steps, obj_xy, obj_z, palm_z0=palm_z0, min_lift_m=0.06
        )

    return step_hold(env, target, steps)


def interpolate_segment(start: torch.Tensor, end: torch.Tensor, steps: int) -> list[torch.Tensor]:
    if steps <= 1:
        return [end.clone()]
    return [start + (t + 1) / steps * (end - start) for t in range(steps)]


def expand_keyframes(
    keyframes: list[tuple[str, torch.Tensor, int]],
    step_scale: float = 1.0,
) -> list[tuple[str, torch.Tensor]]:
    if not keyframes:
        return []
    scale = max(0.25, step_scale)
    expanded: list[tuple[str, torch.Tensor]] = []
    current = keyframes[0][1]
    for label, target, hold_steps in keyframes:
        scaled_hold = max(int(hold_steps * scale), 30)
        move_steps = max(scaled_hold, 40)
        for step_target in interpolate_segment(current, target, move_steps):
            expanded.append((f"{label}_move", step_target))
        for _ in range(max(scaled_hold // 2, 15)):
            expanded.append((f"{label}_hold", target))
        current = target
    return expanded
