#!/usr/bin/env bash
# Build cyclonedds + unitree_sdk2_python under jp_test/external (no system changes).
set -euo pipefail

JP_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXT="${JP_TEST_ROOT}/external"
CYCLONE="${EXT}/cyclonedds"
SDK="${EXT}/unitree_sdk2_python"

echo "=== Phase 2 DDS deps (optional, for sim_main.py teleop replay) ==="

if ! command -v cmake >/dev/null; then
  echo "cmake not found; skip DDS build"
  exit 0
fi

cd "${EXT}"
if [[ ! -d "${CYCLONE}" ]]; then
  git clone --depth 1 -b releases/0.10.x https://github.com/eclipse-cyclonedds/cyclonedds.git
fi
if [[ ! -d "${CYCLONE}/install/lib" ]]; then
  echo "Building CycloneDDS..."
  cmake -S "${CYCLONE}" -B "${CYCLONE}/build" -DCMAKE_INSTALL_PREFIX="${CYCLONE}/install"
  cmake --build "${CYCLONE}/build" --target install -j"$(nproc)"
fi

if [[ ! -d "${SDK}" ]]; then
  git clone --depth 1 https://github.com/unitreerobotics/unitree_sdk2_python.git
fi

export CYCLONEDDS_HOME="${CYCLONE}/install"
echo "CYCLONEDDS_HOME=${CYCLONEDDS_HOME}"
echo "To install unitree_sdk2_python into Isaac Sim python:"
echo "  CYCLONEDDS_HOME=${CYCLONEDDS_HOME} /home/autonomique/AVSR/isaac_sim/python.sh -m pip install -e ${SDK}"
echo "DDS deps ready under ${EXT}"
