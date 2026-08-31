# System Architecture — Project 1

## Data flow

```mermaid
flowchart LR
  subgraph StageA["Stage A — Dex3 source"]
    DEMO["Stage A demo\n(data.json)"]
    EVALA["evaluate_stage_a.py\nC1–C4 configs"]
  end

  subgraph Retarget["Retargeting layer"]
    CACHE["Dex3 tip cache"]
    JMAP["semantic_joint_map_v1"]
    IK["task_space_ik_v2"]
    ADAPT["residual_adapter_v1\n(Stage C)"]
  end

  subgraph StageB["Stage B — BrainCo retarget"]
    DEMOB["Stage B demo"]
    PLAY["BrainCo hand playback"]
  end

  subgraph StageC["Stage C — Fine-tuned"]
    DEMOC["Stage C demo"]
  end

  subgraph Sim["Isaac Sim / Isaac Lab"]
    G1["G1 + Dex3 env"]
    BC["BrainCo standalone USD"]
    COUP["G1 wrist coupling\n(Phase 7)"]
  end

  DEMO --> CACHE
  DEMO --> JMAP
  CACHE --> IK
  JMAP --> DEMOB
  IK --> DEMOB
  JMAP --> ADAPT
  ADAPT --> DEMOC
  DEMO --> EVALA
  G1 --> EVALA
  DEMOB --> PLAY
  BC --> PLAY
  DEMOC --> PLAY
  DEMO --> G1
  DEMOB --> COUP
```

## Component responsibilities

| Component | Role | Frozen / trainable |
|-----------|------|-------------------|
| Source demo | Motion prior (xr_teleoperate format) | Frozen |
| Dex3 scripted policy | Stage A trajectory generator | Frozen |
| Joint-map retargeter | Dex3 7-DoF → BrainCo 6-DoF | Frozen (deterministic) |
| Task-space IK | Tip-target optimization | Frozen (deterministic) |
| Residual adapter | `q* = q_joint + f(q_joint)` | **Trainable** (Stage C) |
| BrainCo sim playback | Stage B/C validation | Eval only |

## Evaluation loop

```
configs/stage_a_eval.yaml  →  Stage A trials  →  results/stage_A/
configs/finetune.yaml      →  Stage C train   →  checkpoints/stage_C/
retarget + validate        →  Stage B demo    →  results/stage_B/
scripts/final/evaluate_final_abc.py  →  results/final/project1_abc_results.md
```

## Key paths

- Source: `checkpoints/stage_A/demos/.../data.json`
- Retarget: `checkpoints/stage_B/demos/.../data.json`
- Fine-tuned: `checkpoints/stage_C/demos/.../data.json`
- Checkpoints: `checkpoints/stage_C/residual_adapter_{right,left}.pt`
