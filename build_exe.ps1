$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

# On this machine `python` is a uv shim (C:\Users\asus\.local\bin\python.cmd)
# which silently fails under PowerShell with "系统找不到指定的路径" and exits
# the script without actually running PyInstaller. Prefer `py -3.13` (PEP 397
# launcher) which always points at the real interpreter. If py.exe is missing
# (non-Windows / stripped install), fall back to plain `python`.
$py = Get-Command py -ErrorAction SilentlyContinue
if ($py) {
    & py.exe -3.13 -m PyInstaller --noconfirm --clean tensorearch.spec
} else {
    Write-Warning "py.exe not found — falling back to 'python'. If the build exits silently, check that 'python' resolves to a real interpreter with PyInstaller installed (Get-Command python)."
    & python -m PyInstaller --noconfirm --clean tensorearch.spec
}

Write-Host "Built:" (Join-Path $root "dist\tensorearch.exe")
