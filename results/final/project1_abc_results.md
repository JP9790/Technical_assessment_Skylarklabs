# Project 1 — Final A/B/C Results (Guide §10)

**Final eval pass:** yes
**Generated:** 2026-08-31T11:38:10.572442+00:00

## A/B/C comparison table

| Metric | A: Dex3 source | B: BrainCo retarget | C: BrainCo + FT |
|--------|----------------|---------------------|-----------------|
| Task / playback success rate | 100.0% | 100.0% | 100.0% |
| Fingertip tracking error (proxy, m) | N/A (source FK not aggregated) | 1.20 cm (joint playback) | 1.24 cm (joint playback) |
| Wrist tracking error (m) | N/A | 0.59 cm (mount samples) | see phase7 (unchanged) |
| Object drops (mean per config) | 0 | N/A (playback-only Stage B) | N/A (playback-only) |
| Joint-limit violations (retarget) | 0 | 0 | 0 |
| Max frame delta (retarget smoothness, m) | N/A | 1.83 cm | 1.83 cm (inherited) |
| Mean completion time (s) | 30.5 | 4.8 | 43.3 eval |
| Adaptation samples / time | N/A | mapping only | 570 frames, 141.2s train |
| Retarget method | xr_teleoperate-format scripted demo | task_space_ik_v2 | residual_adapter_v1 |

## Stage A per configuration

- **C1_nominal**: success=100%, drops=0, lift=6.0cm
- **C2_pose_variation**: success=100%, drops=0, lift=6.0cm
- **C3_mass_variation**: success=100%, drops=0, lift=6.0cm
- **C4_held_out_pose**: success=100%, drops=0, lift=6.0cm

## Failure analysis

- F1 morphology: BrainCo 6-DoF vs Dex3 7-DoF — mitigated by task-space retargeting
- F4 object slip: Stage A uses kinematic lift assist (not pure contact grasp)
- F7 reachability: G1+BrainCo full USD swap deferred; Phase 7 validates wrist coupling
