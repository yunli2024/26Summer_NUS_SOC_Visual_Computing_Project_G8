$pythonPath = "D:\miniconda3\envs\vc_sws3026\python.exe"
$entryPoint = Join-Path $PSScriptRoot "VisualComputingProject\bonus_level\main.py"

if (-not (Test-Path $pythonPath)) {
    Write-Error "Python environment not found: $pythonPath"
    exit 1
}

& $pythonPath $entryPoint
