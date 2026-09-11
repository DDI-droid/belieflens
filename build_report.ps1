# Builds the BeliefLens v2.0 report end-to-end. From the repo root:
#   powershell -ExecutionPolicy Bypass -File build_report.ps1
#
# 1. loads .env (OPENAI_API_KEY) if present
# 2. regenerates every figure          -> report/figs/*.pdf
# 3. compiles the report               -> report/main.pdf
# 4. renders every page to PNG         -> report/pages/p01.png ...
# 5. copies the PDF to the repo root   -> belieflens_report_v2.0.pdf

$ErrorActionPreference = "Continue"

if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$') {
      [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
    }
  }
  Write-Host "loaded .env"
}

Write-Host "`n=== 1/3  figures ==="
python report\figures.py
if (-not $?) { Write-Host "figures failed"; exit 1 }

Write-Host "`n=== 2/3  tectonic ==="
tools\tectonic.exe report\main.tex
if (-not $?) { Write-Host "LaTeX failed -- see the error above"; exit 1 }

Write-Host "`n=== 3/3  page images ==="
python -c @"
import pypdfium2 as pdfium, pathlib
out = pathlib.Path('report/pages'); out.mkdir(exist_ok=True)
for f in out.glob('*.png'): f.unlink()
pdf = pdfium.PdfDocument('report/main.pdf')
for i, page in enumerate(pdf):
    page.render(scale=140/72).to_pil().save(out / ('p%02d.png' % (i+1)))
print('rendered pages:', len(pdf))
"@

Copy-Item report\main.pdf belieflens_report_v2.0.pdf -Force
Write-Host "`ndone -> belieflens_report_v2.0.pdf"
