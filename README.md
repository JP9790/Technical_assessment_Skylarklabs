# Project 1 — Dex3 to BrainCo Revo2 Touch Embodiment Transfer

Unitree G1 29DoF take-home assessment: transfer dexterous manipulation from **Dex3** hands to **BrainCo Revo2 Touch** hands via retargeting and simulation fine-tuning.

**Current phase:** Phase 8 complete — final A/B evaluation passes (guide §10). Stage C fine-tuning optional.

## Experiment stages

| Stage | Description |
|-------|-------------|
| A | Source baseline — G1 + Dex3 + existing policy / demos |
| B | Retargeting only — G1 + BrainCo, no task fine-tuning |
| C | Retargeting + limited simulation fine-tuning |

## Solution report

### Problem and approach

The assessment asks for **embodiment transfer**: reproduce dexterous manipulation on G1 + Dex3 (Stage A), then carry that behavior to G1 + BrainCo Revo2 Touch **without** retraining a full policy from scratch (Stage B), optionally improving with limited fine-tuning (Stage C).

**Core design choices:**

1. **Demo-first, not RL-first** — Public RL checkpoints for this exact task were unavailable; the reproducible source is an `xr_teleoperate`-compatible demonstration (`data.json`) recorded in the Unitree G1 + Dex3 sim. Stage A measures whether that motion achieves pick-and-lift under variation configs (C1–C4).

2. **Two-layer retargeting** — Dex3 (7 DoF/hand) and BrainCo (6 actuated DoF/hand, 5 fingertips) differ in morphology. A **semantic joint map (v1)** gives a fast deterministic baseline; **task-space IK (v2)** optimizes BrainCo joint angles against Dex3 fingertip targets from a sim FK cache (scipy L-BFGS-B). Production validation uses v1; v2 is available for higher-fidelity offline runs.

3. **Incremental sim bring-up** — Each phase adds one capability (stack → Dex3 env → Stage A → BrainCo hand → correspondence → retarget → Stage B playback → wrist coupling → final A/B table). Isaac Lab 6.x compatibility shims live under `scripts/phase1/` so `unitree_sim_isaaclab` runs without patching upstream installs.

4. **Documented deviations** — Task is **cylinder pick-place** (not bimanual handover). BrainCo assets come from public `brainco-description` (not a vendor bundle). Full **G1+BrainCo single-articulation USD swap** is deferred; Phase 7 validates the wrist coupling data path instead. Stage A uses **kinematic lift assist** (hybrid palm-attached follow + scripted Z-lift), not pure contact grasp.

5. **Stage C (optional)** — A small **residual adapter** (`q* = q_joint + f(q_joint)`) can fine-tune retargeted trajectories using IK v2 as teacher signal (`scripts/phase8/`). Final eval prioritizes Stages A and B per guide §9.

**End-to-end data flow:**

```
Stage A demo (Dex3 q, G1 arms)
  → Dex3 FK tip cache + correspondence table
  → retarget (joint map v1 or IK v2)
  → Stage B demo (BrainCo q)
  → standalone BrainCo playback (Phase 6)
  → G1 arm replay + BrainCo at wrist (Phase 7)
  → optional residual adapter → Stage C demo
  → A/B/C metrics table (final eval)
```

See `docs/architecture.md` for the component diagram and `docs/technical_report.md` for full results.

### Phases — goal and rationale

| Phase | Guide / topic | Goal | Why this phase exists |
|-------|---------------|------|------------------------|
| **0** | Stack freeze | Pin Isaac Sim, Isaac Lab, `unitree_sim_isaaclab`, GMR; audit source-policy candidates; scaffold configs and verification scripts | Retargeting and eval are sensitive to versions and asset paths; freezing the stack first avoids debugging sim issues and policy ambiguity later |
| **1** | G1 + Dex3 bring-up | Launch `Isaac-PickPlace-Cylinder-G129-Dex3-Joint`, extract joint/link interface, run smoke and sanity tests | Confirms the **source embodiment** sim works before investing in demos or target-hand work; produces `g1_dex3_source.yaml` |
| **2** | Stage A baseline | Select source (`xr_teleoperate` demo replay), record demo, evaluate C1–C4 (pose/mass/held-out variations) | Assessment requires a **named, reproducible source capability** and multi-config metrics before any hand swap |
| **3** | BrainCo bring-up | Import BrainCo Revo2 Touch USD/URDF, extract 6-DoF interface, standalone open/close/sweep tests | Validates the **target embodiment** in isolation before mapping Dex3 semantics onto it |
| **4** | Correspondence (§6) | Define wrist, palm, fingertip, and joint regularization weights in `correspondence_table.yaml` | Retargeting needs an explicit **semantic map** between unlike hands (Dex3 has no ring/pinky; BrainCo has mimic joints and touch links) |
| **5** | Retargeting (§7) | Build Dex3 tip cache; run joint-map or IK retarget; validate limits and smoothness; write Stage B demo | Core **embodiment transfer** step: Dex3 motion → BrainCo joint trajectories without task fine-tuning |
| **6** | Stage B playback (§8) | Replay Stage B demo on standalone BrainCo USD; measure tracking error and failure modes | Proves retargeted trajectories are **physically plausible on the target hand** before coupling to the full robot |
| **7** | G1 + BrainCo coupling | Replay G1 arms from Stage B; spawn BrainCo at recorded wrist poses; validate mount + joint tracking | Bridges the gap where a **full G1+BrainCo USD** is not publicly available; validates the runtime coupling path |
| **8** | Final eval (§10) | Re-run Stage A → retarget → Stage B → coupling; verify all phase checkpoints; produce A/B/C results table | Single **reproducible pipeline** and consolidated metrics for submission |

**Stage C fine-tuning** (assessment §1.5, guide §9) is implemented in `scripts/phase8/run_phase8.sh` (residual adapter train/apply/eval). It is optional relative to the A/B-first final eval in `scripts/final/run_final_eval.sh`.

### Results summary

| Stage | Outcome | Key artifact |
|-------|---------|--------------|
| A (Dex3) | 100% success, 4 configs × 10 trials | `results/stage_A/stage_a_baseline.json` |
| B (BrainCo retarget) | Playback pass, 0 limit violations, ~2.4 cm max frame delta | `results/stage_B/stage_b_baseline.json` |
| C (+ fine-tuning) | Residual adapter available; optional in default final run | `results/stage_C/stage_c_baseline.json` |

Full pass criteria, per-phase metrics, and artifact index: [`results/README.md`](results/README.md).

Detailed per-phase reports: `docs/phase0_stack_freeze.md` … `docs/phase7_g1_brainco_coupling.md`, `docs/final_evaluation.md`.

## Prerequisites

- **NVIDIA Isaac Sim** — `unitree_sim_isaaclab` lists Isaac Sim 4.5 / 5.x as tested; this project was also run on 6.0.x (see `configs/stack.yaml`).
- **Isaac Lab** — pinned commit in `external/PINS.md`.
- **CUDA-capable GPU** recommended for headless sim runs.

Pinned simulation and asset dependencies live under `external/` (see `external/PINS.md`).

## Environment setup

Before running verification or phase scripts, export paths to your local Isaac Sim and Isaac Lab installs:

| Variable | Description |
|----------|-------------|
| `JP_TEST_ROOT` | Path to this repository root |
| `ISAAC_SIM_PATH` | Isaac Sim install directory |
| `ISAACLAB_PATH` | Isaac Lab install directory |

```bash
export JP_TEST_ROOT="$(git rev-parse --show-toplevel)"
export ISAAC_SIM_PATH="/path/to/isaac_sim"
export ISAACLAB_PATH="/path/to/isaac_lab"

# Optional: if your Isaac Sim tree provides setup_conda_env.sh
# source "${ISAAC_SIM_PATH}/setup_conda_env.sh"

export PYTHONPATH="${ISAACLAB_PATH}/source/isaaclab:${ISAACLAB_PATH}/source/isaaclab_tasks:${ISAACLAB_PATH}/source/isaaclab_rl:${JP_TEST_ROOT}/external/unitree_sim_isaaclab:${PYTHONPATH:-}"
```

Phase run scripts resolve `JP_TEST_ROOT` automatically; the exports above are required for Phase 0 verification and any direct `python` / `scripts/run_isaac.sh` usage.

## Quick start (Phase 0 verification)

```bash
cd "$(git rev-parse --show-toplevel)"  # repository root
# Set JP_TEST_ROOT, ISAAC_SIM_PATH, ISAACLAB_PATH, and PYTHONPATH (see above)
python scripts/verify_phase0_stack.py
python scripts/audit_source_policies.py
```

Results are written to `results/phase0/`.

## Repository layout

```
├── configs/           # Frozen stack, Dex3 joint map, source policy, retargeting
├── scripts/           # Environment setup and verification
├── external/          # unitree_sim_isaaclab, GMR (pinned commits)
├── checkpoints/       # Policy checkpoints (Phase 2+)
├── results/           # Metrics, plots, phase checkpoints (see results/README.md)
└── docs/              # Phase reports
```

## Phase 1 quick start

```bash
bash scripts/phase1/run_phase1.sh --headless --enable_cameras
```

## Phase 2 quick start (Stage A)

```bash
bash scripts/phase2/run_phase2.sh --headless --enable_cameras
```

## Phase 3 quick start (BrainCo Revo2)

```bash
bash scripts/phase3/run_phase3.sh --headless --device cuda
```

## Phase 4 quick start (correspondence)

```bash
bash scripts/phase4/run_phase4.sh --headless --device cuda --enable_cameras
```

## Phase 5 quick start (retargeting)

```bash
bash scripts/phase5/run_phase5.sh --headless --device cuda --enable_cameras
```

## Phase 6 quick start (Stage B playback)

```bash
bash scripts/phase6/run_phase6.sh --headless --device cuda --enable_cameras
```

## Final evaluation (Phase 8 — guide §10)

Run the full pipeline: Stage A (4 configs) → retarget → Stage B → G1 coupling → A/B/C table.

```bash
bash scripts/final/run_final_eval.sh --headless --device cuda --enable_cameras
```

Outputs:
- `results/final/project1_abc_results.md` — guide §10 A/B/C table
- `results/final/project1_abc_results.json` — machine-readable metrics
- `results/final/phase_verification.json` — per-phase pass/fail
- `results/final/phase8_checkpoint.yaml`

## Phase 7 quick start (G1 + BrainCo coupling)

```bash
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
