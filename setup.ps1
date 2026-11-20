$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$taskCachePath = Join-Path $PSScriptRoot '.cache'
New-Item -ItemType Directory -Path (Join-Path $taskCachePath 'tmp') -Force | Out-Null
$env:PIP_CACHE_DIR = Join-Path $taskCachePath 'pip'
$env:TEMP = Join-Path $taskCachePath 'tmp'
$env:TMP = $env:TEMP
if (-not (Test-Path -LiteralPath $taskPythonPath)) {
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create virtual environment' }
}
& $taskPythonPath -m pip install -e '.[dev]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& $taskPythonPath doctor.py
