#!/usr/bin/env bash
# Run a Python script inside Isaac Sim's bundled interpreter (legion install).
# All paths stay under jp_test except the read-only Isaac Sim / Isaac Lab trees.
#
# Usage:
#   scripts/run_isaac.sh scripts/phase1/extract_robot_interface.py --headless
#   scripts/run_isaac.sh external/unitree_sim_isaaclab/sim_main.py --headless --task ...

set -euo pipefail

JP_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_SIM="/home/autonomique/AVSR/isaac_sim"
ISAACLAB="/home/autonomique/AVSR/IsaacLab"
UNITREE_SIM="${JP_TEST_ROOT}/external/unitree_sim_isaaclab"

export PROJECT_ROOT="${UNITREE_SIM}"
export JP_TEST_ROOT="${JP_TEST_ROOT}"

# Isaac Lab sources
export PYTHONPATH="${ISAACLAB}/source/isaaclab:${ISAACLAB}/source/isaaclab_tasks:${ISAACLAB}/source/isaaclab_rl:${ISAACLAB}/source/isaaclab_assets:${UNITREE_SIM}:${PYTHONPATH:-}"

# Prefer GPU when available
export OMNI_KIT_ACCEPT_EULA=YES

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <script.py> [args...]"
  exit 1
fi

SCRIPT="$1"
shift

if [[ ! "$SCRIPT" = /* ]]; then
  SCRIPT="${JP_TEST_ROOT}/${SCRIPT}"
fi

if [[ ! -f "$SCRIPT" ]]; then
  echo "Script not found: $SCRIPT"
  exit 1
fi

# Deactivate conda so Isaac Sim python.sh does not warn / pick wrong interpreter
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate 2>/dev/null || true
fi

exec "${ISAAC_SIM}/python.sh" "$SCRIPT" "$@"
