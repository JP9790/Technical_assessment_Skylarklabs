"""Task-space IK retargeting Dex3 tip cache → BrainCo actuated joints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from task_space_retarget import (
    BRAINCO_ACTUATED_ORDER,
    RetargetResult,
    retarget_hand_joint_space,
    write_retargeted_demo,
)

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = JP_TEST_ROOT / "results" / "phase5" / "dex3_tip_cache.json"
WEIGHTS = {"thumb": 3.0, "index": 2.5, "middle": 1.5, "ring": 0.35, "pinky": 0.25}


def _load_cache() -> dict[str, Any]:
    import json

    return json.loads(CACHE_PATH.read_text())


def retarget_hand_ik(
    tips_wrist: dict[str, list[float]],
    side: str,
    fk,
    *,
    q0: np.ndarray | None = None,
    reg_weight: float = 0.03,
) -> np.ndarray:
    from scipy.optimize import minimize

    targets = {k: np.array(v, dtype=np.float64) for k, v in tips_wrist.items()}
    middle_u = float(np.linalg.norm(targets.get("middle", np.zeros(3))))
    if middle_u > 1e-6:
        spread = middle_u / 0.12
        targets["ring"] = targets.get("middle", np.zeros(3)) * 0.85 * min(spread, 1.0)
        targets["pinky"] = targets.get("middle", np.zeros(3)) * 0.65 * min(spread, 1.0)

    if q0 is None:
        q0 = retarget_hand_joint_space(
            np.zeros(7), side, reg_weight=reg_weight
        )

    bnds = fk.bounds()

    def loss(q: np.ndarray) -> float:
        tips = fk.tips_in_base(q)
        total = 0.0
        for finger, w in WEIGHTS.items():
            if finger not in targets or finger not in tips:
                continue
            total += w * float(np.sum((tips[finger] - targets[finger]) ** 2))
        neutral = np.array(q0)
        total += reg_weight * float(np.sum((q - neutral) ** 2))
        return total

    res = minimize(loss, q0, method="L-BFGS-B", bounds=bnds, options={"maxiter": 20})
    return res.x.astype(np.float64)


def _interpolate_frames(key_indices: list[int], key_q: np.ndarray, total: int) -> np.ndarray:
    """Linear interpolation between IK keyframes to full trajectory length."""
    if len(key_indices) == 0:
        raise ValueError("empty keyframes")
    if len(key_indices) == 1:
        return np.repeat(key_q, total, axis=0)
    out = np.zeros((total, key_q.shape[1]), dtype=np.float64)
    for t in range(total):
        if t <= key_indices[0]:
            out[t] = key_q[0]
            continue
        if t >= key_indices[-1]:
            out[t] = key_q[-1]
            continue
        for k in range(len(key_indices) - 1):
            i0, i1 = key_indices[k], key_indices[k + 1]
            if i0 <= t <= i1:
                alpha = (t - i0) / max(i1 - i0, 1)
                out[t] = (1.0 - alpha) * key_q[k] + alpha * key_q[k + 1]
                break
    return out


def _retarget_side_frames(
    cache_frames: list[dict],
    side: str,
    device: str,
    brainco_cfg: dict,
    demo: dict | None = None,
    *,
    reg_weight: float = 0.03,
    subsample: int = 4,
) -> tuple[np.ndarray, int]:
    import sys

    sys.path.insert(0, str(JP_TEST_ROOT / "scripts" / "phase5"))
    from brainco_hand_fk import BraincoHandFK
    from task_space_retarget import retarget_hand_joint_space

    tip_key = f"{side}_tips_wrist"
    ee_key = f"{side}_ee"
    fk = BraincoHandFK(side, device)
    step = max(1, subsample)
    key_indices = list(range(0, len(cache_frames), step))
    if key_indices[-1] != len(cache_frames) - 1:
        key_indices.append(len(cache_frames) - 1)

    key_out: list[np.ndarray] = []
    violations = 0
    hand = brainco_cfg[f"{side}_hand"]

    for ki, fi in enumerate(key_indices):
        if ki % 5 == 0:
            print(f"  IK retarget {side}: keyframe {ki}/{len(key_indices)}", flush=True)
        frame = cache_frames[fi]
        dex3_q = None
        if demo is not None and fi < len(demo["data"]):
            dex3_q = np.array(demo["data"][fi]["states"][ee_key]["qpos"], dtype=np.float64)
        q0 = retarget_hand_joint_space(dex3_q if dex3_q is not None else np.zeros(7), side, reg_weight=reg_weight)
        q = retarget_hand_ik(frame[tip_key], side, fk, q0=q0, reg_weight=reg_weight)
        key_out.append(q)
        for j, name in enumerate(hand["actuated_joint_names"]):
            lo = float(hand["joint_limits"][name]["lower"])
            hi = float(hand["joint_limits"][name]["upper"])
            if q[j] < lo - 1e-4 or q[j] > hi + 1e-4:
                violations += 1

    del fk
    try:
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
    except Exception:
        pass
    full = _interpolate_frames(key_indices, np.stack(key_out), len(cache_frames))
    return full, violations


def retarget_demo_ik_merged(demo: dict, *, subsample: int = 4) -> "RetargetResult":
    """Merge per-side IK outputs written by retarget_side_ik.py (separate Isaac sessions)."""
    import json

    partial = JP_TEST_ROOT / "results" / "phase5" / "ik_partial"
    right = np.load(partial / "right_q.npy")
    left = np.load(partial / "left_q.npy")
    meta_r = json.loads((partial / "right_meta.json").read_text())
    meta_l = json.loads((partial / "left_meta.json").read_text())
    if right.shape[0] != len(demo["data"]) or left.shape[0] != len(demo["data"]):
        raise ValueError("Partial IK frame count does not match demo")

    deltas: list[float] = []
    for i in range(1, len(right)):
        deltas.append(float(max(np.max(np.abs(right[i] - right[i - 1])), np.max(np.abs(left[i] - left[i - 1])))))

    return RetargetResult(
        brainco_right=right,
        brainco_left=left,
        method="task_space_ik_v2",
        per_frame_max_delta=np.array(deltas, dtype=np.float64),
        limit_violations=int(meta_r["limit_violations"]) + int(meta_l["limit_violations"]),
    )


def retarget_demo_ik(demo: dict, device: str = "cuda", *, subsample: int = 4) -> RetargetResult:
    if not CACHE_PATH.is_file():
        raise FileNotFoundError(f"Missing tip cache: {CACHE_PATH}. Run build_dex3_tip_cache.py first.")

    cache = _load_cache()
    if len(cache["frames"]) != len(demo["data"]):
        raise ValueError("Tip cache frame count does not match demo")

    brainco_cfg = yaml.safe_load((JP_TEST_ROOT / "configs/g1_brainco_target.yaml").read_text())
    right, viol_r = _retarget_side_frames(cache["frames"], "right", device, brainco_cfg, subsample=subsample)
    left, viol_l = _retarget_side_frames(cache["frames"], "left", device, brainco_cfg, subsample=subsample)

    deltas: list[float] = []
    for i in range(1, len(right)):
        deltas.append(float(max(np.max(np.abs(right[i] - right[i - 1])), np.max(np.abs(left[i] - left[i - 1])))))

    return RetargetResult(
        brainco_right=right,
        brainco_left=left,
        method="task_space_ik_v2",
        per_frame_max_delta=np.array(deltas, dtype=np.float64),
        limit_violations=viol_r + viol_l,
    )
