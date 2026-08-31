# Phase 6 Report — Stage B Playback (guide §8)

**Date:** 2026-08-30

## Goal

Replay retargeted BrainCo hand trajectory on standalone Revo2 USD (retargeting-only, no G1 body swap).

## Artifacts

| File | Purpose |
|------|---------|
| `scripts/phase6/evaluate_stage_b.py` | Kinematic playback eval |
| `results/stage_B/stage_b_baseline.json` | Stage B metrics |
| `checkpoints/stage_B/demos/.../data.json` | Retargeted demo |

## Results (2026-08-30)

| Hand | Frames | Max track error | Pass |
|------|--------|-----------------|------|
| Right | 285 | 1.2 cm | yes |
| Left | 285 | 1.2 cm | yes |

**Stage B status: pass**

## Run

```bash
bash scripts/phase6/run_phase6.sh --headless --device cuda --enable_cameras
```
