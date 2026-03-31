$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

python -m PyInstaller --noconfirm --clean tensorearch.spec

Write-Host "Built:" (Join-Path $root "dist\\tensorearch.exe")
