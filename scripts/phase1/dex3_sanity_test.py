#!/usr/bin/env python3
"""Phase 1 scripted Dex3 sanity tests in Isaac Sim (headless-capable).

Tests (implementation guide §3.4):
  1. Move right arm in free space
  2. Open and close each Dex3 hand
  3. Move each finger joint independently (right hand)
  4. Approach cylinder with scripted wrist trajectory
  5. Close hand around cylinder
  6. Lift cylinder ~8 cm and hold
  7. Reset and repeat (2 cycles)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "phase2"))

from isaac_bootstrap import JP_TEST_ROOT, import_unitree_tasks  # noqa: E402
from pick_place_trajectory import build_keyframes, run_keyframe  # noqa: E402
from robot_control import (  # noqa: E402
    RIGHT_ARM_JOINTS,
    RIGHT_HAND_CLOSED,
    RIGHT_HAND_OPEN,
    apply_hand,
    build_target,
    palm_object_distance,
    step_hold,
    tracked_joint_delta,
)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-PickPlace-Cylinder-G129-Dex3-Joint")
parser.add_argument("--cycles", type=int, default=2)
parser.add_argument("--step-scale", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()

import gymnasium as gym
from env_utils import prepare_env_cfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

RESULTS_DIR = JP_TEST_ROOT / "results" / "phase1"
TASK = args.task


def run_segment(env, target, steps: int, label: str) -> dict:
    max_joint_delta = step_hold(env, target, steps)
    obj_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
    arm_delta = tracked_joint_delta(env, target, RIGHT_ARM_JOINTS)
    return {
        "label": label,
        "steps": steps,
        "max_joint_delta": max_joint_delta,
        "arm_joint_delta": arm_delta,
        "palm_object_dist_m": palm_object_distance(env),
        "object_z": obj_z,
        "ok": arm_delta < 0.25,
    }


def test_finger_independence(env, steps: int) -> list[dict]:
    results = []
    base = build_target(env)
    names = list(env.scene["robot"].joint_names)
    finger_joints = [n for n in names if n.startswith("right_hand_")]
    for jname in finger_joints:
        vec = base.clone()
        idx = names.index(jname)
        lo = float(env.scene["robot"].data.soft_joint_pos_limits[0, idx, 0].item())
        hi = float(env.scene["robot"].data.soft_joint_pos_limits[0, idx, 1].item())
        vec[idx] = lo + 0.65 * (hi - lo)
        results.append(run_segment(env, vec, max(steps // 2, 40), f"finger_{jname}"))
        run_segment(env, base, max(steps // 4, 20), f"finger_{jname}_reset")
    return results


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    env_cfg = prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1))
    env = gym.make(TASK, cfg=env_cfg).unwrapped

    all_results: list[dict] = []
    arm_up = build_target(
        env,
        right_shoulder_pitch_joint=-0.35,
        right_elbow_joint=0.85,
        right_wrist_pitch_joint=0.15,
    )
    open_hand = apply_hand(build_target(env), env, RIGHT_HAND_OPEN)
    closed_hand = apply_hand(build_target(env), env, RIGHT_HAND_CLOSED)
    obj_init_z = 0.0

    for cycle in range(args.cycles):
        if cycle == 0:
            env.reset()
            for label, tgt in [
                ("1_arm_free_space", arm_up),
                ("2_hands_open", open_hand),
                ("2_hands_close", closed_hand),
                ("2_hands_reopen", open_hand),
            ]:
                r = run_segment(env, tgt, 120, f"cycle{cycle}_{label}")
                all_results.append(r)
                print(f"  [{'OK' if r['ok'] else 'WARN'}] {label}: palm_dist={r['palm_object_dist_m']:.3f}")
            all_results.extend(test_finger_independence(env, 120))

        env.reset()
        obj_init_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
        from pick_place_trajectory import reset_trial_anchor

        reset_trial_anchor(env)
        keyframes = build_keyframes(env)
        for label, target, hold in keyframes:
            max_joint_delta = run_keyframe(env, label, target, hold)
            obj_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
            arm_delta = tracked_joint_delta(env, target, RIGHT_ARM_JOINTS)
            r = {
                "label": f"cycle{cycle}_{label}",
                "steps": hold,
                "max_joint_delta": max_joint_delta,
                "arm_joint_delta": arm_delta,
                "palm_object_dist_m": palm_object_distance(env),
                "object_z": obj_z,
                "ok": arm_delta < 0.25,
            }
            all_results.append(r)
            print(
                f"  [{'OK' if r['ok'] else 'WARN'}] {label}: "
                f"obj_z={r['object_z']:.3f} palm_dist={r['palm_object_dist_m']:.3f}"
            )

    obj_final_z = float(env.scene["object"].data.root_pos_w[0, 2].item())
    lift_delta = obj_final_z - obj_init_z
    root_z = float(env.scene["robot"].data.root_pos_w[0, 2].item())

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": TASK,
        "cycles": args.cycles,
        "object_init_z": obj_init_z,
        "object_final_z": obj_final_z,
        "robot_root_z": root_z,
        "lift_delta_m": lift_delta,
        "lift_success": lift_delta > 0.03,
        "segments": all_results,
        "all_segments_ok": all(r["ok"] for r in all_results),
        "elapsed_s": time.time() - t0,
        "phase1_sanity_status": "pass" if lift_delta > 0.03 and root_z > 0.65 else "fail_lift",
    }

    out = RESULTS_DIR / "dex3_sanity_test.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(f"Lift delta: {lift_delta * 100:.1f} cm — status: {report['phase1_sanity_status']}")

    env.close()
    simulation_app.close()
    return 0 if report["phase1_sanity_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
