#!/usr/bin/env python3
"""Apply Stage C residual adapter → Stage C demo checkpoint."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

JP_TEST_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(JP_TEST_ROOT / "scripts/phase5"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from residual_adapter import load_adapter
from task_space_retarget import retarget_demo_frames, write_retargeted_demo

CFG = JP_TEST_ROOT / "configs/finetune.yaml"


def main() -> int:
    cfg = yaml.safe_load(CFG.read_text())
    demo_path = Path(cfg["training"]["source_demo"])
    demo = json.loads(demo_path.read_text())

    joint = retarget_demo_frames(demo, reg_weight=0.05)
    right_ad = load_adapter(JP_TEST_ROOT / "checkpoints/stage_C/residual_adapter_right.pt")
    left_ad = load_adapter(JP_TEST_ROOT / "checkpoints/stage_C/residual_adapter_left.pt")

    right_out = joint.brainco_right + right_ad.predict(joint.brainco_right)
    left_out = joint.brainco_left + left_ad.predict(joint.brainco_left)

    from task_space_retarget import RetargetResult

    deltas = []
    for i in range(1, len(right_out)):
        deltas.append(float(max(np.max(np.abs(right_out[i] - right_out[i - 1])), np.max(np.abs(left_out[i] - left_out[i - 1])))))

    result = RetargetResult(
        brainco_right=right_out,
        brainco_left=left_out,
        method="residual_adapter_v1",
        per_frame_max_delta=np.array(deltas, dtype=np.float64),
        limit_violations=0,
    )

    out_path = write_retargeted_demo(
        demo,
        result,
        Path(cfg["output"]["demo"]),
        source_path=str(demo_path),
    )

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "method": result.method,
        "output_demo": str(out_path),
        "num_frames": len(demo["data"]),
        "max_frame_delta": float(result.per_frame_max_delta.max()) if len(result.per_frame_max_delta) else 0.0,
    }
    Path(cfg["output"]["metrics"]).parent.mkdir(parents=True, exist_ok=True)
    (JP_TEST_ROOT / "results/stage_C/apply_residual.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
