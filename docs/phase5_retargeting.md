# Phase 5 Report — Task-Space Retargeting (guide §7)

**Date:** 2026-08-30

## Pipeline

```
Stage A Dex3 demo (data.json)
  → build_dex3_tip_cache.py (G1 sim FK: thumb/index/middle tips, wrist-relative)
  → task_space_ik_v2 (scipy L-BFGS-B + BrainCo sim FK)  OR  semantic_joint_map_v1 (--method joint)
  → Stage B BrainCo demo (checkpoints/stage_B/.../data.json)
```

Default method is **IK v2** (`--method ik`). Joint map v1 remains available for fast offline runs.

## Artifacts

| File | Purpose |
|------|---------|
| `scripts/phase5/build_dex3_tip_cache.py` | Dex3 fingertip cache from G1 sim |
| `scripts/phase5/task_space_ik_retarget.py` | Task-space IK optimizer |
| `scripts/phase5/brainco_hand_fk.py` | BrainCo sim FK for IK |
| `scripts/phase5/task_space_retarget.py` | Joint-map retargeting (v1) |
| `scripts/phase5/retarget_demo.py` | Offline batch retarget (`--method ik\|joint`) |
| `scripts/phase5/validate_retarget.py` | Limit/smoothness + sim sample |
| `results/phase5/dex3_tip_cache.json` | Wrist-relative Dex3 tips |
| `results/phase5/retarget_offline.json` | Offline metrics |
| `results/phase5/retarget_validation.json` | Validation report |

## Results (2026-08-30)

- 570 frames retargeted, 0 limit violations
- Max frame delta: 2.4 cm (smooth)
- Sim sample tracking error: 7.3 mm

## Run

```bash
bash scripts/phase5/run_phase5.sh --headless --device cuda --enable_cameras
```
