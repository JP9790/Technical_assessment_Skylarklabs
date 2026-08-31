#!/usr/bin/env bash
# Concatenate phase reports into a single technical report; optionally build PDF.
set -euo pipefail

JP_TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PATH="${HOME}/miniconda3/bin:${PATH}"
OUT="${JP_TEST_ROOT}/docs/technical_report.md"

cat > "$OUT" <<'HDR'
# Project 1 — Technical Report
## Dex3 to BrainCo Revo2 Touch Embodiment Transfer

**Assignment:** Unitree G1_29DoF Take-Home Assessment — Assignment 1  
**Repository:** `/home/autonomique/jp_test`

---

HDR

append() {
  local f="$1"
  if [[ -f "$f" ]]; then
    echo "" >> "$OUT"
    cat "$f" >> "$OUT"
    echo "" >> "$OUT"
  fi
}

append "${JP_TEST_ROOT}/docs/final_evaluation.md"
append "${JP_TEST_ROOT}/docs/architecture.md"
append "${JP_TEST_ROOT}/docs/sim_to_real_plan.md"
append "${JP_TEST_ROOT}/docs/assessment_compliance.md"
append "${JP_TEST_ROOT}/results/final/project1_abc_results.md"

echo "Wrote ${OUT}"

if command -v pandoc >/dev/null 2>&1; then
  PDF="${JP_TEST_ROOT}/docs/technical_report.pdf"
  PANDOC_OPTS=(
    -V geometry:margin=1in
    --metadata title="Project 1 Technical Report"
    --metadata author="jp_test"
  )
  if pandoc "$OUT" -o "$PDF" --pdf-engine=xelatex "${PANDOC_OPTS[@]}" 2>/dev/null; then
    echo "Wrote ${PDF} (xelatex)"
  elif pandoc "$OUT" -o "$PDF" --pdf-engine=pdflatex "${PANDOC_OPTS[@]}" 2>/dev/null; then
    echo "Wrote ${PDF} (pdflatex)"
  elif command -v wkhtmltopdf >/dev/null 2>&1 && \
       pandoc "$OUT" -o "$PDF" --pdf-engine=wkhtmltopdf "${PANDOC_OPTS[@]}" 2>/dev/null; then
    echo "Wrote ${PDF} (wkhtmltopdf)"
  elif python3 - <<'PY'
import markdown
from pathlib import Path
from weasyprint import HTML

root = Path("/home/autonomique/jp_test")
md_path = root / "docs/technical_report.md"
pdf_path = root / "docs/technical_report.pdf"
body = markdown.markdown(md_path.read_text(), extensions=["tables", "fenced_code"])
html = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<style>
body {{ font-family: sans-serif; margin: 2cm; line-height: 1.45; font-size: 11pt; }}
h1,h2,h3 {{ page-break-after: avoid; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: left; }}
code {{ background: #f4f4f4; padding: 1px 4px; }}
pre {{ background: #f4f4f4; padding: 8px; overflow-x: auto; }}
</style></head><body>{body}</body></html>"""
HTML(string=html, base_url=str(md_path.parent)).write_pdf(str(pdf_path))
print(pdf_path.stat().st_size)
PY
  then
    echo "Wrote ${PDF} (weasyprint)"
  else
    echo "pandoc PDF build failed — install texlive-xetex, wkhtmltopdf, or weasyprint"
  fi
else
  echo "pandoc not installed — Markdown report only"
fi
