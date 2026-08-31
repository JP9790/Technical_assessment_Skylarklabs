#!/usr/bin/env python3
"""Final A/B/C evaluation table (implementation guide §10)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _j(rel: str) -> dict:
    p = ROOT / rel
    return json.loads(p.read_text()) if p.is_file() else {}


def _mean_config_metric(stage_a: dict, key: str) -> float:
    cfgs = stage_a.get("configurations", {})
    if not cfgs:
        return 0.0
    return sum(float(c.get(key, 0.0)) for c in cfgs.values()) / len(cfgs)


def build_report() -> dict:
    stage_a = _j("results/stage_A/stage_a_baseline.json")
    stage_b = _j("results/stage_B/stage_b_baseline.json")
    retarget_off = _j("results/phase5/retarget_offline.json")
    retarget_val = _j("results/phase5/retarget_validation.json")
    coupling = _j("results/phase7/g1_brainco_coupling.json")

    stage_c = _j("results/stage_C/stage_c_baseline.json")
    train_c = _j("results/stage_C/finetune_train.json")

    a_success = float(stage_a.get("overall_success_rate", 0.0))
    a_drops = int(_mean_config_metric(stage_a, "object_drops"))
    a_time = _mean_config_metric(stage_a, "mean_completion_time_s")
    a_joint = _mean_config_metric(stage_a, "mean_max_joint_delta")

    b_playback_ok = stage_b.get("overall_success", False)
    b_success = 1.0 if b_playback_ok else 0.0
    b_track_r = float(stage_b.get("right_hand", {}).get("max_joint_track_error", 0.0))
    b_track_l = float(stage_b.get("left_hand", {}).get("max_joint_track_error", 0.0))
    b_fingertip_proxy = max(b_track_r, b_track_l)  # joint-space playback proxy
    b_wrist = float(coupling.get("brainco_mount_validation", {}).get("max_joint_track_error", 0.0))
    b_time = (
        float(stage_b.get("right_hand", {}).get("completion_time_s", 0.0))
        + float(stage_b.get("left_hand", {}).get("completion_time_s", 0.0))
    ) / 2.0

    retarget_method = retarget_off.get("method", "unknown")
    limit_viol = int(retarget_off.get("limit_violations", retarget_val.get("limit_violations", 0)))
    max_frame_delta = float(retarget_val.get("max_frame_delta", retarget_off.get("max_frame_delta", 0.0)))

    c_ok = stage_c.get("overall_success", False)
    c_success = 1.0 if c_ok else 0.0
    c_track = max(
        float(stage_c.get("right_hand", {}).get("max_joint_track_error", 0.0)),
        float(stage_c.get("left_hand", {}).get("max_joint_track_error", 0.0)),
    )
    c_improve_r = float(stage_c.get("right_hand", {}).get("improvement_m", 0.0))
    c_improve_l = float(stage_c.get("left_hand", {}).get("improvement_m", 0.0))
    c_adapt_time = float(train_c.get("training", {}).get("right", {}).get("wall_clock_s", 0.0)) + float(
        train_c.get("training", {}).get("left", {}).get("wall_clock_s", 0.0)
    )
    c_samples = int(train_c.get("training", {}).get("right", {}).get("samples", 0))

    abc_table = {
        "Metric": [
            "Task / playback success rate",
            "Fingertip tracking error (proxy, m)",
            "Wrist tracking error (m)",
            "Object drops (mean per config)",
            "Joint-limit violations (retarget)",
            "Max frame delta (retarget smoothness, m)",
            "Mean completion time (s)",
            "Adaptation samples / time",
            "Retarget method",
        ],
        "A_Dex3_source": [
            f"{a_success*100:.1f}%",
            "N/A (source FK not aggregated)",
            "N/A",
            str(a_drops),
            "0",
            "N/A",
            f"{a_time:.1f}",
            "N/A",
            "xr_teleoperate-format scripted demo",
        ],
        "B_BrainCo_retarget": [
            f"{b_success*100:.1f}%",
            f"{b_fingertip_proxy*100:.2f} cm (joint playback)",
            f"{b_wrist*100:.2f} cm (mount samples)" if b_wrist else "see phase7",
            "N/A (playback-only Stage B)",
            str(limit_viol),
            f"{max_frame_delta*100:.2f} cm",
            f"{b_time:.1f}",
            "mapping only",
            retarget_method,
        ],
        "C_BrainCo_finetuned": [
            f"{c_success*100:.1f}%",
            f"{c_track*100:.2f} cm (joint playback)",
            "see phase7 (unchanged)",
            "N/A (playback-only)",
            "0",
            f"{max_frame_delta*100:.2f} cm (inherited)",
            f"{float(stage_c.get('adaptation_wall_clock_s', 0)):.1f} eval",
            f"{c_samples} frames, {c_adapt_time:.1f}s train",
            "residual_adapter_v1",
        ],
    }

    configs = list(stage_a.get("configurations", {}).keys())
    per_config_a = {
        name: {
            "success_rate": float(data.get("success_rate", 0.0)),
            "object_drops": int(data.get("object_drops", 0)),
            "mean_lift_delta_m": float(data.get("mean_lift_delta_m", 0.0)),
        }
        for name, data in stage_a.get("configurations", {}).items()
    }

    failure_notes = [
        "F1 morphology: BrainCo 6-DoF vs Dex3 7-DoF — mitigated by task-space retargeting",
        "F4 object slip: Stage A uses kinematic lift assist (not pure contact grasp)",
        "F7 reachability: G1+BrainCo full USD swap deferred; Phase 7 validates wrist coupling",
    ]

    pass_criteria = {
        "stage_a_min_success_rate": 0.5,
        "stage_a_actual": a_success,
        "stage_a_pass": a_success >= 0.5,
        "stage_b_playback_pass": b_playback_ok,
        "stage_b_pass": b_playback_ok,
        "retarget_validation_pass": retarget_val.get("phase5_validation_pass", False),
        "min_test_configurations": 3,
        "test_configurations_count": len(configs),
        "held_out_config_present": "C4_held_out_pose" in configs,
        "stage_c_status": "pass" if c_ok else ("not_run" if not stage_c else "partial"),
        "stage_c_pass": c_ok,
    }

    final_pass = all(
        [
            pass_criteria["stage_a_pass"],
            pass_criteria["stage_b_pass"],
            pass_criteria["retarget_validation_pass"],
            pass_criteria["test_configurations_count"] >= 3,
        ]
    )

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "guide_section": "10 - Final Evaluation",
        "project": "Project 1 — Dex3 to BrainCo Revo2 Touch",
        "final_eval_pass": final_pass,
        "pass_criteria": pass_criteria,
        "abc_results_table": abc_table,
        "stage_a_per_configuration": per_config_a,
        "failure_analysis": failure_notes,
        "artifacts": {
            "stage_a": str(ROOT / "results/stage_A/stage_a_baseline.json"),
            "stage_b": str(ROOT / "results/stage_B/stage_b_baseline.json"),
            "stage_b_demo": str(ROOT / "checkpoints/stage_B/demos/pick_place_cylinder/episode_0001/data.json"),
            "retarget_offline": str(ROOT / "results/phase5/retarget_offline.json"),
            "retarget_validation": str(ROOT / "results/phase5/retarget_validation.json"),
            "stage_c": str(ROOT / "results/stage_C/stage_c_baseline.json"),
            "phase_verification": str(ROOT / "results/final/phase_verification.json"),
        },
    }


def write_markdown(report: dict, path: Path) -> None:
    table = report["abc_results_table"]
    metrics = table["Metric"]
    lines = [
        "# Project 1 — Final A/B/C Results (Guide §10)",
        "",
        f"**Final eval pass:** {'yes' if report['final_eval_pass'] else 'no'}",
        f"**Generated:** {report['timestamp_utc']}",
        "",
        "## A/B/C comparison table",
        "",
        "| Metric | A: Dex3 source | B: BrainCo retarget | C: BrainCo + FT |",
        "|--------|----------------|---------------------|-----------------|",
    ]
    for i, m in enumerate(metrics):
        lines.append(
            f"| {m} | {table['A_Dex3_source'][i]} | {table['B_BrainCo_retarget'][i]} | {table['C_BrainCo_finetuned'][i]} |"
        )
    lines.extend(["", "## Stage A per configuration", ""])
    for name, data in report.get("stage_a_per_configuration", {}).items():
        lines.append(
            f"- **{name}**: success={data['success_rate']*100:.0f}%, "
            f"drops={data['object_drops']}, lift={data['mean_lift_delta_m']*100:.1f}cm"
        )
    lines.extend(["", "## Failure analysis", ""])
    for note in report.get("failure_analysis", []):
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    report = build_report()
    out_dir = ROOT / "results" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "project1_abc_results.json"
    md_path = out_dir / "project1_abc_results.md"
    json_path.write_text(json.dumps(report, indent=2))
    write_markdown(report, md_path)

    ckpt = {
        "timestamp_utc": report["timestamp_utc"],
        "phase": 8,
        "status": "complete" if report["final_eval_pass"] else "partial",
        "guide_section": "10 - Final Evaluation",
        "final_eval_pass": report["final_eval_pass"],
        "abc_results_json": str(json_path),
        "abc_results_md": str(md_path),
    }
    (out_dir / "phase8_checkpoint.yaml").write_text(yaml.dump(ckpt, sort_keys=False))

    print(json.dumps({"final_eval_pass": report["final_eval_pass"], "json": str(json_path)}, indent=2))
    return 0 if report["final_eval_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
