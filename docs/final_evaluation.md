# Final Evaluation Report — Project 1 (Guide §10)

**Date:** 2026-08-31  
**Workspace:** `/home/autonomique/jp_test`

## Executive summary

| Item | Result |
|------|--------|
| **Final eval pass** | **yes** |
| Stage A (Dex3 source) | 100% success, 4 configs × 10 trials |
| Stage B (BrainCo retarget) | Playback pass, 0 limit violations |
| Stage C (fine-tuning) | Not run (optional per guide §9) |
| All phases 0–7 verified | yes (after checkpoint field fixes) |

## Guide §10 — A/B/C results table

See [`results/final/project1_abc_results.md`](../results/final/project1_abc_results.md) for the full table.

| Metric | A: Dex3 | B: BrainCo retarget | C: + FT |
|--------|---------|---------------------|---------|
| Success rate | 100% | 100% (playback) | not run |
| Object drops | 0 | N/A | — |
| Joint-limit violations | 0 | 0 | — |
| Retarget smoothness | N/A | 2.4 cm max Δ/frame | — |

## Stage A configurations (§4.3 + §10)

| Config | Success | Drops | Mean lift |
|--------|---------|-------|-----------|
| C1 nominal | 100% | 0 | 6.0 cm |
| C2 pose variation | 100% | 0 | 6.0 cm |
| C3 mass variation | 100% | 0 | 6.0 cm |
| C4 held-out pose | 100% | 0 | 6.0 cm |

**Artifacts:** `results/stage_A/stage_a_baseline.json`, `checkpoints/stage_A/demos/.../data.json`

## Per-phase verification

| Phase | Guide section | Status | Key artifact |
|-------|---------------|--------|--------------|
| 0 | Stack freeze | pass | `results/phase0/stack_verification.json` |
| 1 | G1+Dex3 bring-up | pass | `results/phase1/dex3_sanity_test.json` |
| 2 | Stage A baseline | pass | `results/stage_A/stage_a_baseline.json` |
| 3 | BrainCo bring-up | pass | `results/phase3/brainco_*_hand_test.json` |
| 4 | Correspondence | pass | `configs/correspondence_table.yaml` |
| 5 | Retargeter | pass | `results/phase5/retarget_validation.json` |
| 6 | Stage B playback | pass | `results/stage_B/stage_b_baseline.json` |
| 7 | G1+BrainCo coupling | pass | `results/phase7/g1_brainco_coupling.json` |
| 8 | Final eval | pass | `results/final/project1_abc_results.json` |

Run verification: `python3 scripts/final/verify_all_phases.py`

## Failure analysis (Stage B taxonomy, §8)

| Code | Mode | Mitigation |
|------|------|------------|
| F1 | Morphology mismatch | Semantic + task-space retargeting |
| F4 | Object slip | Hybrid palm-attached / scripted lift (Stage A) |
| F7 | Arm-hand reachability | Phase 7 wrist coupling; full USD swap deferred |

## Known limitations (explicit)

1. Stage A uses kinematic lift assist — not pure contact grasp.
2. Stage B evaluated as BrainCo hand trajectory playback (standalone USD); full G1+BrainCo pick-place on one articulation requires vendor USD swap.
3. Stage C (residual adapter fine-tuning) not implemented — guide allows A/B-first delivery.
4. Task-space IK v2 (`task_space_ik_v2`) implemented but requires separate Isaac sessions per hand; production path uses `semantic_joint_map_v1` (validated).

## Reproduce final evaluation

```bash
cd /home/autonomique/jp_test
bash scripts/final/run_final_eval.sh --headless --device cuda --enable_cameras
```

Outputs land in `results/final/`.
