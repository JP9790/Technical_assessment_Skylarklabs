# Phase 0 Report — Freeze the Software Stack

**Date:** 2026-08-29  
**Host:** autonomique-legion  
**Workspace:** `/home/autonomique/jp_test`

## Objective

Lock down the software stack, document versions, audit source-policy candidates, and prepare the repository structure before bringing up G1+Dex3 simulation (Phase 1).

## Frozen stack

| Layer | Choice | Version / Path |
|-------|--------|----------------|
| Simulator | NVIDIA Isaac Sim | 6.0.0-rc.59 @ `/home/autonomique/AVSR/isaac_sim` |
| RL framework | Isaac Lab | commit `ad5b07e33e` @ `/home/autonomique/AVSR/IsaacLab` |
| Source sim | unitree_sim_isaaclab | commit `e30c25b` @ `external/unitree_sim_isaaclab` |
| Retargeting ref | GMR | commit `bb1bbe4` @ `external/GMR` |
| Target hand | BrainCo Revo2 Touch | **Not located** |
| GPU | RTX 5090 | driver 595.84 |

## Source embodiment (Dex3)

- Robot: Unitree G1 29DoF
- Hands: Dex3 (7 DoF per hand, 14 total)
- Primary sim task: `Isaac-PickPlace-Cylinder-G129-Dex3-Joint`
- Joint names captured in `configs/dex3_joint_map.yaml`

## Source policy audit

Per assessment rules, training code alone is not a checkpoint.

| Candidate | Artifact | Status |
|-----------|----------|--------|
| xr_teleoperate | Demo trajectories | **Provisional selection** — replay-compatible with unitree_sim |
| dex3_rl_manipulation | RL checkpoint | Unverified — investigate in Phase 2 |
| unitree_sim_isaaclab | Sim only | Environment, not a policy |

**Provisional Stage A plan:** replay xr_teleoperate Dex3 demonstrations in `Isaac-PickPlace-Cylinder-G129-Dex3-Joint`.

## Target embodiment (BrainCo)

No BrainCo Revo2 Touch URDF/USD found on legion. Phase 3 is blocked until the exact assessment/vendor model is obtained. `configs/dex3_to_brainco.yaml` contains a skeleton mapping with `TBD_*` target frames.

## Compatibility notes

1. **Isaac Sim 6.0 vs unitree docs (4.5/5.x):** Must verify `sim_main.py` launches without extension errors in Phase 1.
2. **Missing setup_conda_env.sh:** Isaac Sim tree lacks `setup_conda_env.sh`; `scripts/env_legion.sh` sets PYTHONPATH manually.
3. **Assets not downloaded:** Run `fetch_assets.sh` before first sim launch.

## Deliverables created in Phase 0

- [x] Project directory structure
- [x] `configs/stack.yaml` — pinned stack
- [x] `configs/dex3_joint_map.yaml` — Dex3 interface
- [x] `configs/source_policy.yaml` — source audit
- [x] `configs/dex3_to_brainco.yaml` — retargeting skeleton
- [x] `scripts/env_legion.sh` — legion environment
- [x] `scripts/verify_phase0_stack.py` — automated verification
- [x] `scripts/audit_source_policies.py` — policy candidate audit
- [x] `external/PINS.md` — dependency pins
- [x] `LICENSE_NOTES.md`
- [x] `README.md`

## Phase 1 entry criteria

- [ ] Download unitree_sim_isaaclab USD assets
- [ ] Launch G1+Dex3 cylinder pick-place without physics explosions
- [ ] Programmatically dump joint/link names and update dex3_joint_map.yaml
- [ ] Run scripted open/close/grasp/lift sanity test
