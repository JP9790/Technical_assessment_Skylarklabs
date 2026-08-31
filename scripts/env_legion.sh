#!/usr/bin/env bash
# Source this file before running any jp_test scripts on autonomique-legion.
# Uses the existing Isaac Sim / Isaac Lab install — does NOT modify system paths.
#
# Usage:
#   source /home/autonomique/jp_test/scripts/env_legion.sh

set -euo pipefail

export JP_TEST_ROOT="/home/autonomique/jp_test"
export ISAAC_SIM_PATH="/home/autonomique/AVSR/isaac_sim"
export ISAACLAB_PATH="/home/autonomique/AVSR/IsaacLab"
export UNITREE_SIM_PATH="${JP_TEST_ROOT}/external/unitree_sim_isaaclab"

# Isaac Sim environment (if setup script exists)
if [[ -f "${ISAAC_SIM_PATH}/setup_conda_env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${ISAAC_SIM_PATH}/setup_conda_env.sh"
fi

# Isaac Lab PYTHONPATH
export PYTHONPATH="${ISAACLAB_PATH}/source/isaaclab:${ISAACLAB_PATH}/source/isaaclab_tasks:${ISAACLAB_PATH}/source/isaaclab_rl:${UNITREE_SIM_PATH}:${PYTHONPATH:-}"

# Asset path for unitree_sim_isaaclab (set after fetch_assets.sh)
if [[ -d "${UNITREE_SIM_PATH}/unitree_sim_isaaclab_usds/assets" ]]; then
    export UNITREE_SIM_ASSETS="${UNITREE_SIM_PATH}/unitree_sim_isaaclab_usds/assets"
elif [[ -d "${UNITREE_SIM_PATH}/assets" ]]; then
    export UNITREE_SIM_ASSETS="${UNITREE_SIM_PATH}/assets"
fi

export JP_PROJECT_PHASE="0"
echo "[jp_test] env_legion.sh loaded"
echo "  JP_TEST_ROOT=${JP_TEST_ROOT}"
echo "  ISAAC_SIM_PATH=${ISAAC_SIM_PATH} ($(cat "${ISAAC_SIM_PATH}/VERSION" 2>/dev/null || echo 'unknown'))"
echo "  ISAACLAB_PATH=${ISAACLAB_PATH}"
echo "  UNITREE_SIM_PATH=${UNITREE_SIM_PATH}"
