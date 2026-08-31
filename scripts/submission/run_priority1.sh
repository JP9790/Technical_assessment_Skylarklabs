#!/usr/bin/env bash
# Priority 1 submission deliverables: videos + docs + report build.
set -euo pipefail

JP_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="${HOME}/miniconda3/bin:${PATH}"
RUN="${JP_TEST_ROOT}/scripts/run_isaac.sh"
HEADLESS="${1:---headless}"
EXTRA="${@:2}"

echo "========== Priority 1: Submission deliverables =========="

echo "=== 1. Record labeled videos (Stage A/B/C) ==="
"$RUN" "${JP_TEST_ROOT}/scripts/tools/record_submission_videos.py" \
  --stage all $HEADLESS --device cuda --enable_cameras $EXTRA

echo "=== 2. Build technical report (Markdown) ==="
bash "${JP_TEST_ROOT}/scripts/submission/build_technical_report.sh"

echo "=== Done ==="
echo "Videos: ${JP_TEST_ROOT}/results/videos/video_manifest.json"
echo "Report: ${JP_TEST_ROOT}/docs/technical_report.md"
if [[ -f "${JP_TEST_ROOT}/docs/technical_report.pdf" ]]; then
  echo "PDF:    ${JP_TEST_ROOT}/docs/technical_report.pdf"
fi
