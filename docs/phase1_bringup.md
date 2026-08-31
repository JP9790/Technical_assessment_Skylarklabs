# Phase 1 Report — G1 + Dex3 Simulation Bring-Up

**Date:** 2026-08-30  
**Task:** `Isaac-PickPlace-Cylinder-G129-Dex3-Joint`

## Milestones

| Milestone | Status |
|-----------|--------|
| Unitree USD assets downloaded | Done |
| Isaac Sim 6 + unitree_sim launches headless | Done |
| G1 29DoF robot loads (43 joints, 55 bodies) | Done |
| Dex3 hands present | Done |
| Table + cylinder scene | Done |
| Physics runs 50+ steps without crash | Done |
| Robot interface extracted to YAML | Done |
| Scripted open/close/grasp/lift | Partial — lift not achieved yet |

## Verified robot interface

- **Joints:** 43 (29 body + 14 Dex3)
- **Control rate:** 100 Hz (dt=0.005, decimation=2)
- **Config files:** `configs/g1_dex3_source.yaml`, `results/phase1/g1_dex3_interface.json`

## Isaac Lab 6.x compatibility shims (in `jp_test` only)

| Issue | Fix location |
|-------|----------------|
| `configclass` lazy import | `scripts/phase1/isaac_bootstrap.py` |
| `sim.physx` removed | `scripts/phase1/env_cfg_compat.py` |
| Observation `Device` / dtype types | `scripts/phase1/observation_compat.py` |
| Missing `unitree_sdk2py` / cyclonedds | `scripts/phase1/stubs/` (headless tests) |
| Action joint ordering | `scripts/phase1/action_utils.py` |

## Run commands

```bash
cd /home/autonomique/jp_test

# Full Phase 1 pipeline (headless, ~2 min)
bash scripts/phase1/run_phase1.sh --headless --enable_cameras

# GUI sim via unitree_sim_main
bash scripts/phase1/launch_dex3_sim.sh

# Individual steps
bash scripts/run_isaac.sh scripts/phase1/smoke_test_env.py --headless --device cuda --enable_cameras
bash scripts/run_isaac.sh scripts/phase1/extract_robot_interface.py --headless --device cuda --enable_cameras
bash scripts/run_isaac.sh scripts/phase1/dex3_sanity_test.py --headless --device cuda --enable_cameras
```

## Known issues / Phase 2 follow-ups

1. **Scripted grasp/lift:** Headless joint-position commands do not yet lift the cylinder (`fail_lift` in sanity JSON). Use `sim_main.py` with DDS teleop or tune poses using GUI.
2. **Real DDS:** Install cyclonedds + `unitree_sdk2_python` for teleoperation replay (Stage A demos).
3. **Isaac Sim 6 vs unitree docs (4.5/5.x):** Working via jp_test shims; document any remaining task failures.

## Phase 2 entry

- Clone `xr_teleoperate` and record/replay Dex3 pick-place demos
- Verify source policy checkpoint (`dex3_rl_manipulation`)
- Run Stage A baseline evaluation
