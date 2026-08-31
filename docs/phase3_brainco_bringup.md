# Phase 3 Report — BrainCo Revo2 Touch Bring-Up

**Date:** 2026-08-30  
**Target:** BrainCo Revo2 Touch dexterous hand on Unitree G1 29DoF

## Asset source

| Field | Value |
|-------|-------|
| Repository | [BrainCoTech/brainco-description](https://github.com/BrainCoTech/brainco-description) |
| Commit | `f332a6f0dc944e26b82976b637074b03f7ee8a2c` |
| Format | URDF + MJCF + **USD** (`revo2_system/usd/`) |
| Deviation | Assessment vendor bundle not on legion; using official BrainCo public Revo2 assets |

## Milestones

| Milestone | Status |
|-----------|--------|
| Obtain BrainCo Revo2 model (URDF/USD/meshes) | Done |
| Import into Isaac Sim | Done — loads as articulation |
| Extract joint/link interface | Done — `configs/g1_brainco_target.yaml` |
| Standalone hand test (right) | Pass (kinematic open/close/joint sweep) |
| Standalone hand test (left) | Pass |
| Update Dex3→BrainCo frame map | Done — `configs/dex3_to_brainco.yaml` |
| Attach to G1 full body | Planned Phase 4 prep |

## BrainCo hand interface

- **Actuated DoF per hand:** 6 (thumb metacarpal + proximal, 4 finger proximals)
- **USD revolute joints:** 11 (includes mimic-coupled distal joints)
- **Bodies:** 23 per hand
- **Touch links:** `*_touch` links in URDF (Revo2 Touch)

## Run commands

```bash
cd /home/autonomique/jp_test
bash scripts/phase3/run_phase3.sh --headless --device cuda
```

## Outputs

- `configs/g1_brainco_target.yaml`
- `configs/dex3_to_brainco.yaml` (frames filled)
- `results/phase3/brainco_{right,left}_hand_test.json`
- `results/phase3/phase3_checkpoint.yaml`

## Known limitations

- **PD dynamic tracking** still loose; kinematic joint writes used for Phase 3 validation.
- **Cylinder grasp test** not yet meaningful (cylinder spawn needs scene tuning).
- **G1 wrist attachment** not done — Dex3 USD swap on G1 is Phase 4 prep.

## Phase 4 entry

Implement semantic Dex3→BrainCo retargeting optimizer using verified frame map.
