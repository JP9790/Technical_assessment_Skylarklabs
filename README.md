# Project 1 — Dex3 to BrainCo Revo2 Touch Embodiment Transfer

Unitree G1 29DoF take-home assessment: transfer dexterous manipulation from **Dex3** hands to **BrainCo Revo2 Touch** hands via retargeting and simulation fine-tuning.

**Current phase:** Phase 8 complete — final A/B evaluation passes (guide §10). Stage C fine-tuning optional.

## Experiment stages

| Stage | Description |
|-------|-------------|
| A | Source baseline — G1 + Dex3 + existing policy / demos |
| B | Retargeting only — G1 + BrainCo, no task fine-tuning |
| C | Retargeting + limited simulation fine-tuning |

## Quick start (Phase 0 verification)

```bash
cd /home/autonomique/jp_test
source scripts/env_legion.sh
python scripts/verify_phase0_stack.py
python scripts/audit_source_policies.py
```

Results are written to `results/phase0/`.

## Legion environment

This project uses the **existing** Isaac Sim / Isaac Lab install on `autonomique-legion`:

| Tool | Path |
|------|------|
| Isaac Sim 6.0.0-rc.59 | `/home/autonomique/AVSR/isaac_sim` |
| Isaac Lab | `/home/autonomique/AVSR/IsaacLab` |
| GPU | RTX 5090 |

No files outside `jp_test` are modified by this project.

## Repository layout

```
jp_test/
├── configs/           # Frozen stack, Dex3 joint map, source policy, retargeting
├── scripts/           # env setup and verification
├── external/          # unitree_sim_isaaclab, GMR (pinned commits)
├── checkpoints/       # Policy checkpoints (Phase 2+)
├── results/           # Metrics, plots, phase checkpoints
└── docs/              # Phase reports
```

## Phase 1 quick start

```bash
cd /home/autonomique/jp_test
bash scripts/phase1/run_phase1.sh --headless --enable_cameras
```

## Phase 2 quick start (Stage A)

```bash
cd /home/autonomique/jp_test
bash scripts/phase2/run_phase2.sh --headless --enable_cameras
```

## Phase 3 quick start (BrainCo Revo2)

```bash
cd /home/autonomique/jp_test
bash scripts/phase3/run_phase3.sh --headless --device cuda
```

## Phase 4 quick start (correspondence)

```bash
cd /home/autonomique/jp_test
bash scripts/phase4/run_phase4.sh --headless --device cuda --enable_cameras
```

## Phase 5 quick start (retargeting)

```bash
cd /home/autonomique/jp_test
bash scripts/phase5/run_phase5.sh --headless --device cuda --enable_cameras
```

## Phase 6 quick start (Stage B playback)

```bash
cd /home/autonomique/jp_test
bash scripts/phase6/run_phase6.sh --headless --device cuda --enable_cameras
```

## Final evaluation (Phase 8 — guide §10)

Run the full pipeline: Stage A (4 configs) → retarget → Stage B → G1 coupling → A/B/C table.

```bash
cd /home/autonomique/jp_test
bash scripts/final/run_final_eval.sh --headless --device cuda --enable_cameras
```

Outputs:
- `results/final/project1_abc_results.md` — guide §10 A/B/C table
- `results/final/project1_abc_results.json` — machine-readable metrics
- `results/final/phase_verification.json` — per-phase pass/fail
- `results/final/phase8_checkpoint.yaml`

## Phase 7 quick start (G1 + BrainCo coupling)

```bash
cd /home/autonomique/jp_test
bash scripts/phase7/run_phase7.sh --headless --device cuda --enable_cameras
```

## Key configs

- `configs/g1_dex3_source.yaml` — Dex3 source interface (Phase 1)
- `configs/g1_brainco_target.yaml` — BrainCo target interface (Phase 3)
- `configs/correspondence_table.yaml` — semantic correspondence table (Phase 4)
- `configs/dex3_to_brainco.yaml` — retargeting config + neutral poses (Phase 4)
- `configs/source_policy.yaml` — Stage A source selection (Phase 2)
- `configs/stage_a_eval.yaml` — Stage A evaluation settings

## Known limitations (and mitigations)

| Limitation | Mitigation | Status |
|------------|------------|--------|
| Stage A used pure Z-script lift (no palm follow) | **Hybrid lift**: palm-attached 3D follow when palm ≤18 cm; scripted Z-lift fallback otherwise | Implemented — Stage A criteria preserved |
| Retargeting was semantic joint-map v1 only | **Task-space IK v2**: Dex3 fingertip cache + scipy L-BFGS-B on BrainCo sim FK; ring/pinky spread from middle | Implemented in `scripts/phase5/task_space_ik_retarget.py` |
| G1+BrainCo full USD wrist swap deferred | **Phase 7 coupling**: replay G1 arms from Stage B demo, validate BrainCo hand at wrist poses | Implemented — mount + joint tracking smoke |
| True physics contact grasp | Not solved — assist remains kinematic attachment / scripted lift | Open |

Set `JP_USE_SCRIPTED_LIFT=1` to force legacy vertical-only lift (debug).

See `docs/final_evaluation.md`, `docs/assessment_compliance.md`.
