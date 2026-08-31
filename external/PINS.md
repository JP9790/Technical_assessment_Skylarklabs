# External dependency pins — Phase 0 freeze

All paths are relative to `jp_test/external/` unless noted.

| Component | Repository | Commit | License | Role |
|-----------|-----------|--------|---------|------|
| unitree_sim_isaaclab | https://github.com/unitreerobotics/unitree_sim_isaaclab | `e30c25b1dffdf92ada1d6c8c1fe9a47bdde0fecc` | Apache-2.0 | G1+Dex3 Isaac Lab simulation |
| GMR | https://github.com/YanjieZe/GMR | `bb1bbe40774794fceb2a7c579a3464a28e68c844` | MIT | Retargeting reference |
| brainco-description | https://github.com/BrainCoTech/brainco-description | `f332a6f0dc944e26b82976b637074b03f7ee8a2c` | Check repo LICENSE | BrainCo Revo2 Touch USD/URDF (Phase 3) |

## System installs (read-only, outside jp_test)

| Component | Path on legion | Version |
|-----------|----------------|---------|
| Isaac Sim | `/home/autonomique/AVSR/isaac_sim` | 6.0.0-rc.59 |
| Isaac Lab | `/home/autonomique/AVSR/IsaacLab` | `ad5b07e33e` |

## Optional DDS deps (Phase 2, under jp_test/external)

| Component | URL | Purpose |
|-----------|-----|---------|
| cyclonedds | https://github.com/eclipse-cyclonedds/cyclonedds | DDS for sim_main teleop replay |
| unitree_sdk2_python | https://github.com/unitreerobotics/unitree_sdk2_python | Unitree DDS Python bindings |

Build: `bash scripts/phase2/setup_dds_deps.sh`

## Unavailable sources (Phase 2 audit)

| Component | URL | Status |
|-----------|-----|--------|
| dex3_rl_manipulation | https://github.com/PabloKevin/dex3_rl_manipulation | Repository not found (404) |

## Reference datasets (not primary Stage A)

| Dataset | URL | Notes |
|---------|-----|-------|
| G1_Dex3_ObjectPlacement_Dataset | https://huggingface.co/datasets/unitreerobotics/G1_Dex3_ObjectPlacement_Dataset | 210 teleop episodes, different task |

## Assets (Phase 1)

| Asset | Source | Fetch command |
|-------|--------|---------------|
| Unitree sim USDs | https://huggingface.co/datasets/unitreerobotics/unitree_sim_isaaclab_usds | `cd external/unitree_sim_isaaclab && bash fetch_assets.sh` |

| xr_teleoperate | https://github.com/unitreerobotics/xr_teleoperate | `845b25a32f7febedf220e830952a7134897adb9d` | Apache-2.0 | Stage A demo format |

## Target asset (Phase 3 — resolved)

BrainCo Revo2 Touch via `external/brainco-description` → symlink `assets/brainco/revo2_system/`.
