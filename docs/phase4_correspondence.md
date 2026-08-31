# Phase 4 Report — Dex3→BrainCo Semantic Correspondence

**Date:** 2026-08-30  
**Guide:** §6 — Define Dex3-to-BrainCo Correspondence

## Deliverables

| Artifact | Purpose |
|----------|---------|
| `configs/correspondence_table.yaml` | Formal semantic correspondence table (guide §6) |
| `configs/dex3_to_brainco.yaml` | Machine-readable retargeting config + neutral poses |
| `scripts/phase4/verify_correspondence.py` | YAML + Isaac Sim frame verification |
| `scripts/phase4/sync_retargeting_config.py` | Regenerate dex3_to_brainco from table |
| `results/phase4/correspondence_verification.json` | Verification report |

## Correspondence summary

| Semantic element | Dex3 (source) | BrainCo (target) | Mapping |
|------------------|---------------|------------------|---------|
| Wrist pose | `*_wrist_yaw_link` | `*_hand_base_link` | SE(3), w=1.0 |
| Palm pose | `*_hand_palm_link` | `*_hand_base_link` | SE(3), w=0.8 |
| Thumb tip | `*_hand_thumb_2_link` | `*_thumb_tip` | Cartesian, w=3.0 |
| Index tip | `*_hand_index_1_link` | `*_index_tip` | Cartesian, w=2.5 |
| Middle tip | `*_hand_middle_1_link` | `*_middle_tip` | Cartesian, w=1.5 |
| Ring / pinky | — (no Dex3 DoF) | `*_ring_tip`, `*_pinky_tip` | weight 0 |
| Joint config | 14-D hand q | 12-D actuated q | regularization w=0.05 |
| Contacts | none in Dex3 USD | touch links in URDF | deferred Phase 5 |
| Action | joint_position 14 | joint_position 12 via IK | retargeting function |

## Morphology notes

- **Dex3:** 7 DoF/hand (thumb×3, index×2, middle×2); no ring/pinky.
- **BrainCo Revo2:** 6 actuated/hand + mimic distal joints; 5 fingertip frames.
- Ring/pinky targets receive **zero weight** until a source signal exists.

## Run

```bash
cd /home/autonomique/jp_test
bash scripts/phase4/run_phase4.sh --headless --device cuda --enable_cameras
```

## Prerequisites

- **Phase 3:** complete (hand bring-up + mount smoke).
- **Phase 2:** Stage A lift baseline may still be partial; correspondence definition does not require a passing lift demo.

## Next — Phase 5

Implement the task-space optimizer:

```
recorded Dex3 q(t) → FK → wrist/palm/fingertip targets → IK → BrainCo q(t) → Isaac playback
```

See guide §7.1–7.3.
