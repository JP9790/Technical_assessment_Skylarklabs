# Sim-to-Real Deployment Plan — G1 + BrainCo Revo2 Touch

## 1. Calibration

| Item | Procedure |
|------|-----------|
| BrainCo joint zeros | Match URDF zero to open-hand pose; verify against `configs/g1_brainco_target.yaml` limits |
| G1–hand mount | Use Phase 7 wrist poses (`results/phase7/g1_brainco_coupling.json`) as SE(3) mount transform |
| Dex3→BrainCo scale | Re-run correspondence verification (`scripts/phase4/verify_correspondence.py`) on hardware |

## 2. Control stack

- **Rate:** 100 Hz joint targets (per `configs/source_policy.yaml`)
- **Mode:** Position control on BrainCo actuated joints (6 DoF per hand)
- **Retargeting:** Offline demo replay first; on-robot: `q_cmd = q_retarget + adapter(q_retarget)` at 100 Hz

## 3. Latency budget

| Stage | Budget |
|-------|--------|
| State read | ≤ 5 ms |
| Retarget + adapter | ≤ 3 ms (pre-computed table or small MLP) |
| Command write | ≤ 2 ms |
| **Total** | ≤ 10 ms (100 Hz) |

## 4. Safety limits

- Clamp all commands to `g1_brainco_target.yaml` joint limits
- Velocity cap: 50% of URDF velocity limits during bring-up
- E-stop on: joint limit violation, fall detection (IMU), operator override
- Disable adapter until open/close sanity passes

## 5. Staged hardware validation

1. **Week 1:** BrainCo open/close, per-joint sign check (mirror Phase 3 scripts on real hand)
2. **Week 2:** Fixed-base G1 arm replay; BrainCo mounted; no object
3. **Week 3:** Reach + pre-grasp near object; no lift
4. **Week 4:** Full pick-place with human supervision; log drops and retarget errors

## 6. Known sim gaps before hardware

- Kinematic lift assist in Stage A (not contact-based grasp)
- Full G1+BrainCo single-articulation USD not in public assets
- Stage B/C validated as hand trajectory playback; whole-body contact not re-validated on BrainCo body

## 7. Data collection for adapter refinement

- Collect 5–10 teleop demos on real BrainCo after mount
- Fine-tune residual adapter only (keep retargeter frozen)
- Compare fingertip/proprio error vs sim Stage C baseline
