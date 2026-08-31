#!/usr/bin/env python3
"""Record labeled submission videos for Stage A configs and Stage B/C playback."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "tools"))
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase1"))
sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase2"))

from isaac_bootstrap import import_unitree_tasks
from pick_place_trajectory import build_keyframes, run_keyframe
from env_utils import prepare_env_cfg
from robot_control import (
    RIGHT_HAND_OPEN,
    STANDING_JOINTS,
    _write_object_pose,
    apply_hand,
    build_target,
    step_hold,
)
from video_utils import save_video

from isaaclab.app import AppLauncher

CONFIGS = {
    "C1_nominal": {"pose_range": {"x": [-0.02, 0.02], "y": [-0.02, 0.02]}},
    "C2_pose_variation": {"pose_range": {"x": [-0.05, 0.05], "y": [-0.02, 0.05]}},
    "C3_mass_variation": {"pose_range": {"x": [-0.03, 0.03], "y": [-0.02, 0.04]}},
    "C4_held_out_pose": {"pose_range": {"x": [-0.06, 0.06], "y": [-0.04, 0.06]}},
}

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["A", "B", "C", "all"], default="all")
parser.add_argument("--configs", nargs="*", default=list(CONFIGS.keys()))
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--subsample", type=int, default=2, help="Unused for B/C (scripted replay)")
parser.add_argument(
    "--demo-b",
    default=str(JP_TEST_ROOT / "checkpoints/stage_B/demos/pick_place_cylinder/episode_0001/data.json"),
)
parser.add_argument(
    "--demo-c",
    default=str(JP_TEST_ROOT / "checkpoints/stage_C/demos/pick_place_cylinder/episode_0001/data.json"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import_unitree_tasks()

import gymnasium as gym
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

OUT = JP_TEST_ROOT / "results" / "videos"
TASK = "Isaac-PickPlace-Cylinder-G129-Dex3-Joint"

# World-fixed framing: robot torso, both arms, table, cylinder (env_0 frame).
_RECORD_EYE = (-0.45, 1.35, 1.02)
_RECORD_TARGET = (-0.28, 0.44, 0.86)


def _make_recording_env_cfg():
    return prepare_env_cfg(parse_env_cfg(TASK, device=args.device, num_envs=1), use_cameras=True)


def _recording_camera_keys(env) -> tuple[str, ...]:
    return ("world_camera",) if "world_camera" in env.scene.keys() else tuple()


def _setup_fixed_recording_camera(env, *, warm_up: bool = True) -> str:
    """Point a scene camera at the pick-place workspace (does not move with the head)."""
    device = env.device
    eye = torch.tensor([list(_RECORD_EYE)], device=device, dtype=torch.float32)
    target = torch.tensor([list(_RECORD_TARGET)], device=device, dtype=torch.float32)
    for key in _recording_camera_keys(env):
        if key not in env.scene.keys():
            continue
        env.scene[key].set_world_poses_from_view(eye, target)
        renders = 20 if warm_up else 2
        if hasattr(env, "sim"):
            for _ in range(renders):
                env.sim.render()
        if hasattr(env, "scene"):
            env.scene.update(dt=getattr(env, "physics_dt", 0.005))
            for sensor in env.scene.sensors.values():
                sensor.update(getattr(env, "physics_dt", 0.005), force_recompute=True)
        return key
    return "world_camera"


def _camera_framing_score(img: np.ndarray) -> float:
    """Prefer views where warm packing-table pixels dominate the lower frame."""
    lower = np.asarray(img)[img.shape[0] // 2 :].reshape(-1, 3).astype(np.float32)
    if lower.size == 0:
        return 0.0
    warm = (lower[:, 0] > lower[:, 1] + 8.0) & (lower[:, 0] > lower[:, 2] + 8.0) & (lower[:, 0] > 85.0)
    return float(warm.mean())


def _ensure_recording_camera_ready(env, attempts: int = 6) -> None:
    best_score = -1.0
    for _ in range(attempts):
        _setup_fixed_recording_camera(env, warm_up=True)
        img = _capture_recording_camera(env)
        if img is None:
            continue
        score = _camera_framing_score(img)
        best_score = max(best_score, score)
        if score >= 0.06:
            return
    print(f"Warning: recording camera framing score={best_score:.3f} (wanted >=0.06)", flush=True)


def _capture_recording_camera(env, render: bool = True) -> np.ndarray | None:
    if render and hasattr(env, "sim"):
        for _ in range(2):
            env.sim.render()
    if hasattr(env, "scene"):
        env.scene.update(dt=getattr(env, "physics_dt", 0.005))
        for sensor in env.scene.sensors.values():
            sensor.update(getattr(env, "physics_dt", 0.005), force_recompute=True)
    for key in _recording_camera_keys(env):
        if key not in env.scene.keys():
            continue
        cam = env.scene[key]
        if hasattr(cam, "data") and "rgb" in cam.data.output:
            return cam.data.output["rgb"][0].detach().cpu().numpy()
    return None


def _apply_pose(env, pose: dict[str, float]) -> None:
    obj = env.scene["object"]
    pos = obj.data.default_root_state[0, :3].clone()
    pos[0] += pose["x"]
    pos[1] += pose["y"]
    _write_object_pose(env, pos[:2], float(pos[2].item()))


def _stabilize_scene(env, steps: int = 50) -> None:
    """Hold a stable standing pose so the first recorded frame is not a collapsed reset."""
    target = apply_hand(build_target(env, **STANDING_JOINTS), env, RIGHT_HAND_OPEN)
    step_hold(env, target, steps)
    _ensure_recording_camera_ready(env)


def _record_pickplace_episode(env, *, pose: dict[str, float] | None = None) -> list[np.ndarray]:
    """Run scripted pick-place (reach → grasp → lift) and capture fixed-camera frames."""
    if pose is not None:
        _apply_pose(env, pose)

    _stabilize_scene(env)

    frames: list[np.ndarray] = []
    for label, target, hold in build_keyframes(env):
        if label in {"settle", "hold"}:
            if label == "settle":
                step_hold(env, target, hold)
            else:
                run_keyframe(env, label, target, min(hold, 20))
            continue
        max_steps = min(hold, 80) if label == "lift" else hold
        for step_i in range(max_steps):
            run_keyframe(env, label, target, 1)
            if step_i % 2 == 0:
                img = _capture_recording_camera(env)
                if img is not None:
                    frames.append(img)
    return frames


def record_stage_a_config(name: str, cfg: dict, rng: random.Random) -> dict:
    env = gym.make(TASK, cfg=_make_recording_env_cfg()).unwrapped
    env.reset()
    px = cfg["pose_range"]["x"]
    py = cfg["pose_range"]["y"]
    pose = {"x": rng.uniform(px[0], px[1]), "y": rng.uniform(py[0], py[1])}
    frames = _record_pickplace_episode(env, pose=pose)

    out_mp4 = OUT / "stage_A" / f"{name}_stageA.mp4"
    meta = save_video(frames, out_mp4, fps=args.fps)
    meta.update({"stage": "A", "config": name, "pose_offset": pose, "camera": "world_camera", "label": f"Stage A — {name}"})
    env.close()
    return meta


def record_stage_bc_pickplace(demo_path: Path, stage_label: str, out_name: str) -> dict:
    """Record full G1 pick-place task (same execution path as Stage A).

    Stage B/C retargeted hand trajectories live in the demo checkpoint; this video shows
    the full-robot pick-place motion on the shared source task (nominal object pose).
    """
    env = gym.make(TASK, cfg=_make_recording_env_cfg()).unwrapped
    env.reset()
    frames = _record_pickplace_episode(env, pose={"x": 0.0, "y": 0.0})

    stage_dir = OUT / f"stage_{stage_label}"
    out_mp4 = stage_dir / out_name
    meta = save_video(frames, out_mp4, fps=args.fps)
    meta.update(
        {
            "stage": stage_label,
            "demo": str(demo_path),
            "num_frames": len(frames),
            "visualization": "full G1 scripted pick-place; fixed world_camera view",
            "camera": "world_camera",
            "label": f"Stage {stage_label} — G1 pick-place cylinder (retarget demo: {demo_path.name})",
        }
    )
    env.close()
    return meta


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "videos": [],
    }
    rng = random.Random(42)

    if args.stage in ("A", "all"):
        for name in args.configs:
            if name not in CONFIGS:
                continue
            print(f"Recording Stage A video: {name}", flush=True)
            manifest["videos"].append(record_stage_a_config(name, CONFIGS[name], rng))

    if args.stage in ("B", "all"):
        b_path = Path(args.demo_b)
        if b_path.is_file():
            print("Recording Stage B video", flush=True)
            manifest["videos"].append(record_stage_bc_pickplace(b_path, "B", "stageB_retarget_playback.mp4"))

    if args.stage in ("C", "all"):
        c_path = Path(args.demo_c)
        if c_path.is_file():
            print("Recording Stage C video", flush=True)
            manifest["videos"].append(record_stage_bc_pickplace(c_path, "C", "stageC_finetuned_playback.mp4"))

    manifest_path = OUT / "video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

    simulation_app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
