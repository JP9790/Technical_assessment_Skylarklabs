#!/usr/bin/env python3
"""Phase 0 stack verification — records system info without launching Isaac Sim GUI."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

JP_TEST_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = JP_TEST_ROOT / "results" / "phase0"
STACK_YAML = JP_TEST_ROOT / "configs" / "stack.yaml"


def run_cmd(cmd: list[str], timeout: int = 30) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except Exception as exc:
        return -1, str(exc)


def read_version_file(path: Path) -> str | None:
    if path.is_file():
        return path.read_text().strip()
    return None


def check_path(label: str, path: Path) -> dict:
    return {
        "label": label,
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
    }


def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    isaac_sim = Path("/home/autonomique/AVSR/isaac_sim")
    isaac_lab = Path("/home/autonomique/AVSR/IsaacLab")
    unitree_sim = JP_TEST_ROOT / "external" / "unitree_sim_isaaclab"
    gmr = JP_TEST_ROOT / "external" / "GMR"

    report: dict = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "python": sys.version,
        "jp_test_root": str(JP_TEST_ROOT),
        "paths": [
            check_path("isaac_sim", isaac_sim),
            check_path("isaac_lab", isaac_lab),
            check_path("unitree_sim_isaaclab", unitree_sim),
            check_path("GMR", gmr),
            check_path("sim_main.py", unitree_sim / "sim_main.py"),
            check_path("fetch_assets.sh", unitree_sim / "fetch_assets.sh"),
        ],
        "versions": {
            "isaac_sim": read_version_file(isaac_sim / "VERSION"),
            "isaac_lab_git": None,
            "unitree_sim_git": None,
            "gmr_git": None,
        },
        "gpu": None,
        "imports": {},
        "phase0_status": "pass",
        "warnings": [],
        "errors": [],
    }

    for name, repo in [
        ("isaac_lab_git", isaac_lab),
        ("unitree_sim_git", unitree_sim),
        ("gmr_git", gmr),
    ]:
        code, out = run_cmd(["git", "-C", str(repo), "log", "-1", "--format=%H %s"])
        report["versions"][name] = out if code == 0 else f"git failed: {out}"

    code, gpu_out = run_cmd(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    report["gpu"] = gpu_out if code == 0 else f"nvidia-smi unavailable: {gpu_out}"

    # Isaac Lab import test (uses env_legion PYTHONPATH if sourced)
    isaaclab_src = isaac_lab / "source" / "isaaclab"
    if isaaclab_src.is_dir():
        sys.path.insert(0, str(isaaclab_src))
    try:
        import isaaclab  # noqa: F401

        report["imports"]["isaaclab"] = "ok"
    except Exception as exc:
        report["imports"]["isaaclab"] = f"fail: {exc}"
        report["warnings"].append("isaaclab import failed — run: source scripts/env_legion.sh")

    # Asset download status
    assets_candidates = [
        unitree_sim / "unitree_sim_isaaclab_usds" / "assets",
        unitree_sim / "assets",
    ]
    report["unitree_assets_downloaded"] = any(p.is_dir() for p in assets_candidates)
    if not report["unitree_assets_downloaded"]:
        report["warnings"].append(
            "Unitree sim USD assets not downloaded — run fetch_assets.sh in Phase 1"
        )

    # BrainCo asset search (jp_test only)
    brainco_in_jp = list(JP_TEST_ROOT.rglob("*brainco*")) + list(JP_TEST_ROOT.rglob("*revo2*"))
    report["brainco_asset_in_jp_test"] = [str(p) for p in brainco_in_jp]
    if not brainco_in_jp:
        report["warnings"].append("BrainCo Revo2 Touch asset not found in jp_test (expected for Phase 0)")

    if report["errors"]:
        report["phase0_status"] = "fail"
    elif report["warnings"]:
        report["phase0_status"] = "pass_with_warnings"

    out_json = RESULTS_DIR / "stack_verification.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_json}")

    if STACK_YAML.is_file() and yaml is not None:
        stack = yaml.safe_load(STACK_YAML.read_text())
        stack["_verification"] = {
            "timestamp_utc": report["timestamp_utc"],
            "status": report["phase0_status"],
        }
        checkpoint = RESULTS_DIR / "stack_checkpoint.yaml"
        checkpoint.write_text(yaml.dump(stack, sort_keys=False, default_flow_style=False))
        print(f"Wrote {checkpoint}")

    return 0 if report["phase0_status"] != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
