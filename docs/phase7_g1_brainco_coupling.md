# Phase 7 — G1 + BrainCo wrist coupling

Overcomes the Phase 4 deferral of a full G1+BrainCo USD wrist swap by validating **runtime coupling**:

1. **G1 arm replay** — Stage B demo arm joints played on G1+Dex3 (hands open); wrist poses recorded.
2. **BrainCo mount validation** — BrainCo hand spawned at sampled wrist poses with Stage B hand joint targets; joint tracking error measured.

## Run

```bash
bash scripts/phase7/run_phase7.sh --headless --device cuda --enable_cameras
```

## Outputs

- `results/phase7/g1_brainco_coupling.json`
- `results/phase7/phase7_checkpoint.yaml`

## Pass criteria

- `wrist_frames_captured` > 0
- `brainco_mount_validation.mount_pass`: max joint track error < 8 cm (rad-equivalent joint space)

## Remaining gap

Full pick-place on a single G1 articulation with BrainCo USD at the wrist requires a vendor G1+BrainCo robot asset swap (not in public brainco-description). Phase 7 proves the **data path** (Stage B → wrist pose → BrainCo q) without that asset.
