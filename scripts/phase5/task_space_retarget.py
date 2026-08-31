"""Task-space / semantic joint retargeting Dex3 → BrainCo (guide §7)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = JP_TEST_ROOT / "configs" / "dex3_to_brainco.yaml"
DEFAULT_BRAINCO = JP_TEST_ROOT / "configs" / "g1_brainco_target.yaml"

# xr_teleoperate hand qpos order (per side)
DEX3_HAND_ORDER = [
    "thumb_0",
    "thumb_1",
    "thumb_2",
    "middle_0",
    "middle_1",
    "index_0",
    "index_1",
]

BRAINCO_ACTUATED_ORDER = [
    "thumb_metacarpal",
    "thumb_proximal",
    "index_proximal",
    "middle_proximal",
    "ring_proximal",
    "pinky_proximal",
]


@dataclass
class RetargetResult:
    brainco_right: np.ndarray
    brainco_left: np.ndarray
    method: str
    per_frame_max_delta: np.ndarray
    limit_violations: int


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _side_prefix(side: str) -> str:
    return "right" if side == "right" else "left"


def _dex3_joint_name(side: str, short: str) -> str:
    return f"{_side_prefix(side)}_hand_{short}_joint"


def _dex3_limits(cfg: dict, side: str) -> dict[str, tuple[float, float]]:
    lo_map = cfg["limits"]["position_lower"]
    hi_map = cfg["limits"]["position_upper"]
    out: dict[str, tuple[float, float]] = {}
    for short in DEX3_HAND_ORDER:
        j = _dex3_joint_name(side, short)
        out[short] = (float(lo_map[j]), float(hi_map[j]))
    return out


def _brainco_limits(cfg: dict, side: str) -> dict[str, tuple[float, float]]:
    hand = cfg[f"{side}_hand"]
    limits = hand["joint_limits"]
    out: dict[str, tuple[float, float]] = {}
    for j in hand["actuated_joint_names"]:
        short = j.replace(f"{side}_", "")
        lo = float(limits[j]["lower"])
        hi = float(limits[j]["upper"])
        out[short] = (lo, hi)
    return out


def _normalize(val: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))


def _denormalize(u: float, lo: float, hi: float) -> float:
    return float(lo + np.clip(u, 0.0, 1.0) * (hi - lo))


def retarget_hand_joint_space(
    dex3_q: np.ndarray,
    side: str,
    *,
    dex3_cfg: dict | None = None,
    brainco_cfg: dict | None = None,
    neutral: dict[str, float] | None = None,
    reg_weight: float = 0.05,
) -> np.ndarray:
    """Map 7-D Dex3 hand qpos → 6-D BrainCo actuated joints."""
    dex3_cfg = dex3_cfg or _load_yaml(JP_TEST_ROOT / "configs" / "g1_dex3_source.yaml")
    brainco_cfg = brainco_cfg or _load_yaml(DEFAULT_BRAINCO)
    neutral = neutral or (_load_yaml(DEFAULT_CFG).get("neutral_poses", {}).get("brainco_open", {}))

    prefix = _side_prefix(side)
    dlims = _dex3_limits(dex3_cfg, side)
    blims = _brainco_limits(brainco_cfg, side)

    q = {name: float(dex3_q[i]) for i, name in enumerate(DEX3_HAND_ORDER)}

    def map_pair(src_keys: list[str], dst: str) -> float:
        vals = [_normalize(q[k], *dlims[k]) for k in src_keys if k in dlims]
        u = float(np.mean(vals)) if vals else 0.0
        lo, hi = blims[dst]
        neutral_key = f"{prefix}_{dst}_joint"
        u_neutral = _normalize(float(neutral.get(neutral_key, lo)), lo, hi)
        u = (1.0 - reg_weight) * u + reg_weight * u_neutral
        return _denormalize(u, lo, hi)

    middle_val = map_pair(["middle_0", "middle_1"], "middle_proximal_joint")
    ring_lo, ring_hi = blims["ring_proximal_joint"]
    pinky_lo, pinky_hi = blims["pinky_proximal_joint"]
    ring_val = float(np.clip(middle_val * 0.85, ring_lo, ring_hi))
    pinky_val = float(np.clip(middle_val * 0.65, pinky_lo, pinky_hi))

    out = np.array(
        [
            map_pair(["thumb_0"], "thumb_metacarpal_joint"),
            map_pair(["thumb_1", "thumb_2"], "thumb_proximal_joint"),
            map_pair(["index_0", "index_1"], "index_proximal_joint"),
            middle_val,
            ring_val,
            pinky_val,
        ],
        dtype=np.float64,
    )
    return out


def retarget_demo_frames(
    demo: dict,
    *,
    reg_weight: float = 0.05,
) -> RetargetResult:
    """Retarget all frames in xr_teleoperate data.json."""
    dex3_cfg = _load_yaml(JP_TEST_ROOT / "configs" / "g1_dex3_source.yaml")
    brainco_cfg = _load_yaml(DEFAULT_BRAINCO)
    retarget_cfg = _load_yaml(DEFAULT_CFG)
    neutral = retarget_cfg.get("neutral_poses", {}).get("brainco_open", {})

    right_out: list[np.ndarray] = []
    left_out: list[np.ndarray] = []
    deltas: list[float] = []
    violations = 0

    prev_r = None
    prev_l = None
    for item in demo["data"]:
        rq = np.array(item["states"]["right_ee"]["qpos"], dtype=np.float64)
        lq = np.array(item["states"]["left_ee"]["qpos"], dtype=np.float64)
        br = retarget_hand_joint_space(rq, "right", dex3_cfg=dex3_cfg, brainco_cfg=brainco_cfg, neutral=neutral, reg_weight=reg_weight)
        bl = retarget_hand_joint_space(lq, "left", dex3_cfg=dex3_cfg, brainco_cfg=brainco_cfg, neutral=neutral, reg_weight=reg_weight)
        right_out.append(br)
        left_out.append(bl)
        if prev_r is not None:
            deltas.append(float(max(np.max(np.abs(br - prev_r)), np.max(np.abs(bl - prev_l)))))
        prev_r, prev_l = br, bl

        for side, vec in (("right", br), ("left", bl)):
            blims = _brainco_limits(brainco_cfg, side)
            for j, key in enumerate(BRAINCO_ACTUATED_ORDER):
                lo, hi = blims[f"{key}_joint"]
                if vec[j] < lo - 1e-4 or vec[j] > hi + 1e-4:
                    violations += 1

    return RetargetResult(
        brainco_right=np.stack(right_out),
        brainco_left=np.stack(left_out),
        method="semantic_joint_map_v1",
        per_frame_max_delta=np.array(deltas, dtype=np.float64),
        limit_violations=violations,
    )


def write_retargeted_demo(
    source_demo: dict,
    result: RetargetResult,
    out_path: Path,
    *,
    source_path: str,
) -> Path:
    """Write retargeted demo preserving arms/body; replace hand qpos with BrainCo."""
    import copy

    out = copy.deepcopy(source_demo)
    out["info"]["author"] = "jp_test_phase5"
    out["info"]["joint_names"]["right_ee"] = [
        f"right_{n}_joint" for n in BRAINCO_ACTUATED_ORDER
    ]
    out["info"]["joint_names"]["left_ee"] = [
        f"left_{n}_joint" for n in BRAINCO_ACTUATED_ORDER
    ]
    out["text"]["desc"] = (
        "Stage A Dex3 demo retargeted to BrainCo Revo2 actuated hand joints (Phase 5)."
    )
    for i, item in enumerate(out["data"]):
        br = result.brainco_right[i].tolist()
        bl = result.brainco_left[i].tolist()
        for block in ("states", "actions"):
            item[block]["right_ee"]["qpos"] = br
            item[block]["left_ee"]["qpos"] = bl
    meta = out.setdefault("retargeting_meta", {})
    meta.update(
        {
            "source_demo": source_path,
            "method": result.method,
            "target_embodiment": "BrainCo Revo2 Touch",
            "limit_violations": result.limit_violations,
            "mean_frame_delta": float(np.mean(result.per_frame_max_delta)) if len(result.per_frame_max_delta) else 0.0,
            "max_frame_delta": float(np.max(result.per_frame_max_delta)) if len(result.per_frame_max_delta) else 0.0,
        }
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    out_path.write_text(json.dumps(out, indent=2))
    return out_path
