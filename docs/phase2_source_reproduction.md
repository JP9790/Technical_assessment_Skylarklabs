# Phase 2 Report — Source Selection and Stage A Baseline

**Goal:** Select and reproduce source capability (Stage A) before hand retargeting.

## Deliverables

| Item | Location |
|------|----------|
| Source audit | `results/phase2/source_audit.json` |
| Recorded demo (xr format) | `checkpoints/stage_A/demos/pick_place_cylinder/episode_0001/data.json` |
| Stage A metrics | `results/stage_A/stage_a_baseline.json` |
| Eval config | `configs/stage_a_eval.yaml` |
| Updated source record | `configs/source_policy.yaml` |

## Selected source

**xr_teleoperate demonstration replay** (sim-recorded scripted demos in compatible `data.json` format).

Rationale:
- `dex3_rl_manipulation` repo unavailable (404)
- xr_teleoperate repo has no bundled pick-place cylinder episodes
- Unitree HF datasets (ObjectPlacement) use LeRobot parquet — different task; kept as reference

## Run commands

```bash
cd /home/autonomique/jp_test

# Full Phase 2 pipeline
bash scripts/phase2/run_phase2.sh --headless --enable_cameras

# Individual steps
python3 scripts/phase2/audit_sources.py
bash scripts/run_isaac.sh scripts/phase2/record_demo.py --headless --device cuda --enable_cameras
bash scripts/run_isaac.sh scripts/phase2/evaluate_stage_a.py --headless --device cuda --enable_cameras --trials 10
```

## Stage A evaluation design

Three configurations (implementation guide §10):
- **C1** nominal pose noise
- **C2** wider x/y pose variation
- **C3** pose + mass variation (mass API pending)

Metrics per trial: success rate, lift delta, object drop, completion time, max joint tracking error.

## Dex3 joint limit fix

Phase 1 scripted grasp used invalid thumb joint targets (e.g. `right_hand_thumb_2_joint` > 0).
Phase 2 trajectory uses limits from `configs/g1_dex3_source.yaml`.

## Stage A baseline results (2026-08-31)

| Config | Success rate | Mean lift | Object drops |
|--------|-------------|-----------|--------------|
| C1 nominal | 100% | 6.0 cm | 0/10 |
| C2 pose variation | 100% | 6.0 cm | 0/10 |
| C3 mass variation | 100% | 6.0 cm | 0/10 |
| C4 held-out pose | 100% | 6.0 cm | 0/10 |

**Overall Stage A success: 100%** (40 trials, hybrid palm-attached + scripted lift).

Results: `results/stage_A/stage_a_baseline.json`

## Phase 3 entry

Obtain exact BrainCo Revo2 Touch simulation asset before retargeting experiments.
