#!/usr/bin/env bash
# Launch full unitree_sim_isaaclab with G1+Dex3 cylinder pick-place (GUI or headless).
#
# GUI:
#   scripts/phase1/launch_dex3_sim.sh
# Headless + no cameras (fastest smoke):
#   scripts/phase1/launch_dex3_sim.sh --headless --no-render

set -euo pipefail
JP_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SIM="${JP_TEST_ROOT}/external/unitree_sim_isaaclab/sim_main.py"

exec "${JP_TEST_ROOT}/scripts/run_isaac.sh" "$SIM" \
  --device cuda \
  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint \
  --enable_dex3_dds \
  --robot_type g129 \
  "$@"
