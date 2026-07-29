param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    Write-Error "Python was not found. Activate the vc_sws3026 Conda environment first."
    exit 1
}

$pythonExe = $pythonCommand.Source
$requirements = Join-Path $PSScriptRoot "requirements.txt"

if (-not $CheckOnly) {
    & $pythonExe -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # Ultralytics declares opencv-python as a hard dependency even though this
    # project needs the contrib wheel. Install it without dependency resolution
    # after all runtime dependencies are present.
    & $pythonExe -m pip install "ultralytics>=8.3" --no-deps
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $pythonExe (Join-Path $PSScriptRoot "check_runtime.py")
exit $LASTEXITCODE
