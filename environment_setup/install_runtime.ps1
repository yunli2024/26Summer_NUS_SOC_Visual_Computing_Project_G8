param(
    [switch]$CheckOnly,
    [switch]$Locked
)

$ErrorActionPreference = "Stop"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    Write-Error "Python was not found. Activate the vc_sws3026 Conda environment first."
    exit 1
}

$pythonExe = $pythonCommand.Source
$requirementsName = if ($Locked) { "requirements-lock.txt" } else { "requirements.txt" }
$requirements = Join-Path $PSScriptRoot $requirementsName

if (-not $CheckOnly) {
    Write-Host "Installing from $requirementsName"
    & $pythonExe -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    # Ultralytics declares opencv-python as a hard dependency even though this
    # project needs the contrib wheel. Install it without dependency resolution
    # after all runtime dependencies are present.
    if (-not $Locked) {
        & $pythonExe -m pip install "ultralytics>=8.3" --no-deps
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

& $pythonExe (Join-Path $PSScriptRoot "check_runtime.py")
exit $LASTEXITCODE
