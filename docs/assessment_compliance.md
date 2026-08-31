# Assessment Compliance — Project 1 vs Technical Assessment PDF

**Assignment:** Dex-hand → BrainCo Revo2 Touch embodiment transfer (Assignment 1)  
**Deadline:** August 31, 2026

## Can Project 1 be finished?

**Yes for Stages A, B, and C**, with documented deviations on task choice and full USD swap.

| Requirement (PDF) | Status | Evidence |
|-------------------|--------|----------|
| §1.2 Named source policy/demo | ✅ | `configs/source_policy.yaml`, `results/phase2/source_audit.json` |
| §1.3 Retarget without fine-tuning (Stage B) | ✅ | `results/stage_B/stage_b_baseline.json` |
| §1.4 Evaluate across ≥3 configs + held-out | ✅ | C1–C4 in `results/stage_A/stage_a_baseline.json` |
| §1.5 Fine-tune retargeted policy (Stage C) | ✅ | `results/stage_C/stage_c_baseline.json`, `checkpoints/stage_C/residual_adapter_*.pt` |
| §1.1 Full G1 USD hand swap | ⚠️ Partial | Standalone BrainCo + wrist coupling (`results/phase7/`) |
| Example: bimanual handover | ⚠️ Deviation | Pick-place cylinder task used instead (documented) |
| M1–M6 metrics | ✅ A/B/C | `results/final/project1_abc_results.md` |
| Videos per config | ✅ | `results/videos/video_manifest.json` — MP4 (Stage A C1–C4 + B/C BrainCo camera) |
| Architecture diagram | ✅ | `docs/architecture.md` |
| Technical report | ✅ | `docs/technical_report.md`, `docs/technical_report.pdf` |
| Sim-to-real plan | ✅ | `docs/sim_to_real_plan.md` |

## Reproduce core pipeline

```bash
bash scripts/final/run_final_eval.sh --headless --device cuda --enable_cameras
python3 scripts/final/verify_all_phases.py
```

## Known code limitations (explicit)

1. Stage A uses kinematic lift assist — not pure contact grasp.
2. Stage B = BrainCo hand trajectory playback (not full G1+BrainCo pick-place on one articulation).
3. Stage C residual adapter fine-tunes joint trajectories (570-frame teacher from IK v2); not full RL in sim.

## Priority 1 & 2 deliverables

```bash
bash scripts/submission/run_priority1.sh --headless --device cuda --enable_cameras
bash scripts/phase8/run_phase8.sh --headless --device cuda --enable_cameras
```
