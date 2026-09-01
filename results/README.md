# Results — achievements, pass criteria, and artifacts

This document summarizes **what was achieved**, **which phases pass**, and **what each pass gate checks**. Numbers below come from the committed result files in this directory (generated 2026-08-31).

**Quick verification:**

```bash
python scripts/final/verify_all_phases.py          # phases 0–8 milestone gates
python scripts/final/evaluate_final_abc.py         # guide §10 A/B/C table (no GPU)
bash scripts/final/run_final_eval.sh --headless --device cuda --enable_cameras  # full re-run
```

---

## Executive summary

| Gate | Status | Meaning |
|------|--------|---------|
| **All phases 0–8** (`results/final/phase_verification.json`) | **PASS** | Every implementation-phase milestone satisfied |
| **Final eval** (`results/final/project1_abc_results.json`) | **PASS** | Guide §10 minimum: Stage A ≥50% success, Stage B playback, retarget validation, ≥3 test configs |
| **Stage A** (Dex3 source) | **100%** | 4 configs × 10 trials, 0 drops, 6.0 cm mean lift |
| **Stage B** (BrainCo retarget, no FT) | **PASS** | Standalone hand playback, 0 limit violations on retarget |
| **Stage C** (residual adapter) | **PASS** | Adapter trained and playback passes (optional for default final eval) |

Detailed A/B/C table: [`final/project1_abc_results.md`](final/project1_abc_results.md).

---

## Two kinds of “pass”

### 1. Per-phase milestones (Phases 0–8)

Automated by `scripts/final/verify_all_phases.py`. Writes [`final/phase_verification.json`](final/phase_verification.json).

| Phase | Status | Pass indicates | Verification script / evaluator |
|-------|--------|----------------|----------------------------------|
| 0 | PASS | Stack frozen; paths, versions, and `isaaclab` import OK | `scripts/verify_phase0_stack.py` |
| 1 | PASS | G1+Dex3 sim runs; sanity lift **> 3 cm** | `results/phase1/phase1_checkpoint.yaml` + `dex3_sanity_test.json` |
| 2 | PASS | Stage A **success rate ≥ 50%** | `results/phase2/phase2_checkpoint.yaml` + `stage_A/stage_a_baseline.json` |
| 3 | PASS | Both BrainCo hands + mount smoke | `phase3/phase3_checkpoint.yaml` + `brainco_*_hand_test.json` |
| 4 | PASS | Correspondence YAML verified against sim frames | `phase4/phase4_checkpoint.yaml` + `correspondence_verification.json` |
| 5 | PASS | Retarget offline + validation both pass | `phase5/phase5_checkpoint.yaml` + `retarget_offline.json` + `retarget_validation.json` |
| 6 | PASS | Stage B playback on standalone BrainCo USD | `phase6/phase6_checkpoint.yaml` + `stage_B/stage_b_baseline.json` |
| 7 | PASS | Wrist poses captured; BrainCo mount tracking OK | `phase7/phase7_checkpoint.yaml` + `g1_brainco_coupling.json` |
| 8 | PASS | Stage C residual adapter eval passes | `stage_C/stage_c_checkpoint.yaml` + `stage_C/stage_c_baseline.json` |

### 2. Final evaluation (Guide §10)

Automated by `scripts/final/evaluate_final_abc.py`. **Stage C is not required** for `final_eval_pass`.

| Criterion | Threshold | Actual | Pass |
|-----------|-----------|--------|------|
| Stage A success rate | ≥ 50% | **100%** | yes |
| Stage B playback | both hands `playback_pass` | right + left pass | yes |
| Retarget validation | `phase5_validation_pass` | true | yes |
| Test configurations | ≥ 3 | **4** (C1–C4) | yes |
| Held-out config | C4 present | C4_held_out_pose | yes |
| Stage C | optional | pass | yes (bonus) |

---

## Per-phase detail — criteria, results, artifacts

### Phase 0 — Stack freeze

**Goal:** Pin dependencies and verify the workspace before simulation bring-up.

| Pass criterion | How it is checked |
|----------------|-------------------|
| `phase0_status` ≠ `fail` | Required paths exist; no hard errors in verification report |
| `isaaclab` import | `imports.isaaclab == "ok"` when PYTHONPATH is set |
| Unitree assets | `unitree_assets_downloaded` (warning if missing, not always fail) |

**Achieved:** Isaac Sim 6.0.0-rc.59, Isaac Lab `ad5b07e33e`, `unitree_sim_isaaclab` `e30c25b`, GMR `bb1bbe4` recorded in [`phase0/stack_verification.json`](phase0/stack_verification.json).

**Key artifacts:** `phase0/stack_verification.json`, `phase0/stack_checkpoint.yaml`, `phase0/source_policy_audit.json`

---

### Phase 1 — G1 + Dex3 bring-up

**Goal:** Launch cylinder pick-place task; extract Dex3 robot interface; prove lift is possible.

| Pass criterion | Threshold | Achieved |
|----------------|-----------|----------|
| Checkpoint `status` | `complete` | yes |
| `lift_success` in sanity test | lift **> 3 cm** (`lift_delta_m > 0.03`) | **6.0 cm** |
| Sim smoke | 43 joints, 55 bodies, 50+ physics steps | yes |

**Evaluator:** `scripts/phase1/dex3_sanity_test.py` — trial success uses same lift rule as Stage A: `lift_delta > 0.03 m` and `xy_shift < 0.08 m`.

**Key artifacts:** `phase1/g1_dex3_interface.json`, `phase1/dex3_sanity_test.json`, `phase1/phase1_checkpoint.yaml`, `configs/g1_dex3_source.yaml`

**Note:** Per-segment `ok` flags in sanity JSON can be false while overall `phase1_sanity_status` is pass (lift segment succeeds).

---

### Phase 2 — Stage A baseline

**Goal:** Named reproducible Dex3 source capability before any hand swap.

| Pass criterion | Threshold | Achieved |
|----------------|-----------|----------|
| Checkpoint `status` | `complete` | yes |
| `overall_success_rate` | ≥ **50%** (milestone gate) | **100%** |
| Per-trial success | `lift_delta > 0.03 m` and `xy_shift < 0.08 m` | 40/40 trials |
| Object drop | `lift_delta < -0.01 m` | 0 drops |

**Stage A configurations (guide §4.3 + §10):**

| Config | Success | Drops | Mean lift |
|--------|---------|-------|-----------|
| C1_nominal | 100% | 0 | 6.0 cm |
| C2_pose_variation | 100% | 0 | 6.0 cm |
| C3_mass_variation | 100% | 0 | 6.0 cm |
| C4_held_out_pose | 100% | 0 | 6.0 cm |

**Source:** `xr_teleoperate`-format scripted demo — [`checkpoints/stage_A/demos/pick_place_cylinder/episode_0001/data.json`](../../checkpoints/stage_A/demos/pick_place_cylinder/episode_0001/data.json)

**Key artifacts:** `stage_A/stage_a_baseline.json`, `phase2/source_audit.json`, `phase2/phase2_checkpoint.yaml`, `configs/source_policy.yaml`

---

### Phase 3 — BrainCo Revo2 bring-up

**Goal:** Import target hand USD; validate kinematic control on standalone BrainCo.

| Pass criterion | How it is checked |
|----------------|-------------------|
| `phase3_hand_status` | `pass` for **both** right and left |
| Kinematic core | Per-joint sweep, open/close within error budget |
| Mount smoke | G1 wrist capture + BrainCo spawn (`g1_brainco_attachment_smoke.json`) |

**Achieved:** 6 actuated DoF/hand, 11 USD revolute joints (mimics), 23 bodies; kinematic open/close pass; PD dynamic tests may fail (expected — validation uses kinematic writes).

**Key artifacts:** `phase3/brainco_right_hand_test.json`, `phase3/brainco_left_hand_test.json`, `phase3/brainco_interface.json`, `phase3/g1_brainco_attachment_smoke.json`, `configs/g1_brainco_target.yaml`

---

### Phase 4 — Semantic correspondence

**Goal:** Formal Dex3→BrainCo frame and weight table (guide §6).

| Pass criterion | Threshold | Achieved |
|----------------|-----------|----------|
| `yaml_verification_pass` | all required Dex3 + BrainCo frames found in sim | **true** |
| `dex3_missing` / `brainco_missing` | empty lists | both empty |

**Key artifacts:** `configs/correspondence_table.yaml`, `configs/dex3_to_brainco.yaml`, `phase4/correspondence_verification.json`

---

### Phase 5 — Retargeting

**Goal:** Convert Stage A Dex3 demo to Stage B BrainCo joint trajectories (guide §7).

| Pass criterion | Threshold | Achieved |
|----------------|-----------|----------|
| `phase5_retarget_pass` | `limit_violations == 0`, frames > 0 | **0** violations, **570** frames |
| `phase5_validation_pass` | `limit_violations == 0` and `max_frame_delta ≤ 0.35 m` | max Δ **1.83 cm** |
| Method | — | `task_space_ik_v2` |

**Retarget smoothness:** mean frame delta **0.39 cm**; max **1.83 cm** (joint-space proxy).

**Key artifacts:** `phase5/retarget_offline.json`, `phase5/retarget_validation.json`, `phase5/dex3_tip_cache.json`, `checkpoints/stage_B/demos/.../data.json`

---

### Phase 6 — Stage B playback

**Goal:** Replay retargeted demo on standalone BrainCo USD without fine-tuning (guide §8).

| Pass criterion | Threshold | Achieved |
|----------------|-----------|----------|
| Per-hand `playback_pass` | `max_joint_track_error < 0.08 m` and `max_cmd_delta < 0.35 m` | **~1.2 cm** track error |
| `stage_b_status` | `pass` | pass |

| Hand | Frames | Max track error | Playback pass |
|------|--------|-----------------|---------------|
| Right | 285 | 1.19 cm | yes |
| Left | 285 | 1.20 cm | yes |

**Key artifacts:** `stage_B/stage_b_baseline.json`, `stage_B/stage_b_right.json`, `stage_B/stage_b_left.json`

---

### Phase 7 — G1 + BrainCo wrist coupling

**Goal:** Validate runtime coupling when full G1+BrainCo single USD is unavailable.

| Pass criterion | Threshold | Achieved |
|----------------|-----------|----------|
| `wrist_frames_captured` | > 0 | **143** |
| `brainco_mount_validation.mount_pass` | `max_joint_track_error < 0.08 m` | **0.59 cm** (8 samples) |
| `phase7_status` | `pass` | pass |

**Key artifacts:** `phase7/g1_brainco_coupling.json`, `phase7/phase7_checkpoint.yaml`

---

### Phase 8 / Stage C — Residual adapter fine-tuning

**Goal:** Limited fine-tuning on retargeted trajectories (assessment §1.5, guide §9). Implemented in `scripts/phase8/`; counted as Phase 8 in phase verification.

| Pass criterion | Threshold | Achieved |
|----------------|-----------|----------|
| Training | checkpoints written | `residual_adapter_{right,left}.pt` |
| `stage_c_status` | `pass` | pass |
| Per-hand playback | `max_joint_track_error < 0.08 m` | ~1.20 cm (similar to Stage B) |

**Training:** 570 frames/side, 200 epochs, ~141 s total wall clock ([`stage_C/finetune_train.json`](stage_C/finetune_train.json)).

**Note:** Adapter improvement vs Stage B is small in joint playback space; primary value is demonstrating the Stage C pipeline.

**Key artifacts:** `stage_C/stage_c_baseline.json`, `stage_C/finetune_train.json`, `checkpoints/stage_C/residual_adapter_*.pt`, `checkpoints/stage_C/demos/.../data.json`

---

## A/B/C experiment stages (assessment)

| Stage | What was measured | Result | Primary artifact |
|-------|-------------------|--------|------------------|
| **A** — Dex3 source | Pick-place + lift under C1–C4 | 100% success, 0 drops | `stage_A/stage_a_baseline.json` |
| **B** — Retarget only | BrainCo trajectory playback + retarget quality | Playback pass, 0 limit violations | `stage_B/stage_b_baseline.json`, `phase5/retarget_validation.json` |
| **C** — + fine-tuning | Residual adapter on retargeted demo | Playback pass | `stage_C/stage_c_baseline.json` |

See [`final/project1_abc_results.md`](final/project1_abc_results.md) for the full metrics table (tracking error, smoothness, timing).

---

## Directory map

```
results/
├── README.md                 ← this file
├── phase0/                   Stack verification, source policy audit
├── phase1/                   G1+Dex3 interface + sanity
├── phase2/                   Source audit, reach probes (diagnostics)
├── phase3/                   BrainCo hand tests, mount smoke
├── phase4/                   Correspondence verification
├── phase5/                   Tip cache, retarget offline + validation
├── phase6/                   Phase 6 checkpoint (Stage B eval lives in stage_B/)
├── phase7/                   G1 wrist + BrainCo mount coupling
├── stage_A/                  Stage A evaluation metrics
├── stage_B/                  Stage B playback metrics
├── stage_C/                  Stage C adapter train + eval
└── final/                    Phase verification, A/B/C table, final checkpoint
```

---

## Known gaps (not failing gates, but important)

| Item | Status | Where documented |
|------|--------|------------------|
| Pure contact grasp | Open — kinematic / hybrid lift assist | `README.md` Known limitations |
| Full G1+BrainCo single articulation USD | Deferred — Phase 7 coupling only | `docs/phase7_g1_brainco_coupling.md` |
| Stage B/C object manipulation on BrainCo body | Playback-only — no pick-place on coupled robot | `docs/final_evaluation.md` |
| BrainCo PD dynamic tracking | Loose — kinematic validation used | `phase3/brainco_*_hand_test.json` (`pd_tests` may be false) |
| Task choice | Cylinder pick-place (not bimanual handover) | `docs/assessment_compliance.md` |

---

## Failure taxonomy (Stage B, guide §8)

| Code | Mode | Mitigation in this repo |
|------|------|-------------------------|
| F1 | Morphology mismatch (7 vs 6 DoF) | Semantic map + task-space IK v2 |
| F4 | Object slip | Hybrid palm-attached lift in Stage A |
| F7 | Arm-hand reachability | Phase 7 wrist coupling; full USD swap deferred |
