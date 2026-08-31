#!/usr/bin/env python3
"""Offline retarget Stage A Dex3 demo → BrainCo hand trajectory (guide §7.2)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_space_retarget import retarget_demo_frames, write_retargeted_demo

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = JP_TEST_ROOT / "checkpoints/stage_A/demos/pick_place_cylinder/episode_0001/data.json"
DEFAULT_OUT = JP_TEST_ROOT / "checkpoints/stage_B/demos/pick_place_cylinder/episode_0001/data.json"
RESULTS = JP_TEST_ROOT / "results" / "phase5"
RUN_ISAAC = JP_TEST_ROOT / "scripts/run_isaac.sh"


def _run_ik_isolated(device: str, extra_argv: list[str]) -> None:
    """Run each hand in a fresh Isaac session (avoids /World/Ground and articulation conflicts)."""
    script = JP_TEST_ROOT / "scripts/phase5/retarget_side_ik.py"
    for side in ("right", "left"):
        cmd = [str(RUN_ISAAC), str(script), "--side", side, "--headless", "--device", device, *extra_argv]
        print(f"Running: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_IN))
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--reg-weight", type=float, default=0.05)
    parser.add_argument("--method", choices=["joint", "ik"], default="ik")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ik-extra", nargs=argparse.REMAINDER, help="Extra args forwarded to retarget_side_ik.py")
    parser.add_argument("--merge-only", action="store_true", help="Only merge ik_partial/*.npy (skip Isaac subprocess)")
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(in_path)

    demo = json.loads(in_path.read_text())
    ik_extra = [a for a in (args.ik_extra or []) if a != "--"]

    if args.method == "ik":
        from task_space_ik_retarget import retarget_demo_ik_merged

        if not args.merge_only:
            _run_ik_isolated(args.device, ik_extra)
        result = retarget_demo_ik_merged(demo)
    else:
        result = retarget_demo_frames(demo, reg_weight=args.reg_weight)

    out_path = write_retargeted_demo(demo, result, Path(args.output), source_path=str(in_path))

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_demo": str(in_path),
        "output_demo": str(out_path),
        "num_frames": len(demo["data"]),
        "method": result.method,
        "limit_violations": result.limit_violations,
        "mean_frame_delta": float(result.per_frame_max_delta.mean()) if len(result.per_frame_max_delta) else 0.0,
        "max_frame_delta": float(result.per_frame_max_delta.max()) if len(result.per_frame_max_delta) else 0.0,
        "phase5_retarget_pass": result.limit_violations == 0 and len(demo["data"]) > 0,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "retarget_offline.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["phase5_retarget_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
