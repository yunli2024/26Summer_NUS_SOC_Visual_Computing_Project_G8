$ErrorActionPreference = "Stop"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    Write-Error "Python was not found. Activate the vc_sws3026 environment first."
    exit 1
}

$pythonExe = $pythonCommand.Source
$lbfModel = Join-Path $PSScriptRoot "resources\face_models\lbfmodel.yaml"

Push-Location $PSScriptRoot
try {
    Write-Host "[1/8] Expert feature tests"
    & $pythonExe expert_level\test_expression_features.py
    if ($LASTEXITCODE -ne 0) { throw "Expert feature tests failed." }

    Write-Host "[2/8] Dance-scoring tests"
    & $pythonExe bonus_level\test_dance_scoring.py
    if ($LASTEXITCODE -ne 0) { throw "Dance-scoring tests failed." }

    Write-Host "[3/8] Mario gesture and game-mechanics tests"
    & $pythonExe -m unittest discover -s bonus_level_mario -p "test_*.py" -v
    if ($LASTEXITCODE -ne 0) { throw "Mario tests failed." }

    Write-Host "[4/8] Expression-effects preview"
    & $pythonExe expert_level\task2_realtime.py --preview tmp\expert_effects_preview.png
    if ($LASTEXITCODE -ne 0) { throw "Expression preview failed." }

    Write-Host "[5/8] Pose-scoring deterministic A/B check"
    & $pythonExe bonus_level\scoring_video_tester.py --check
    if ($LASTEXITCODE -ne 0) { throw "Pose-scoring A/B check failed." }

    Write-Host "[6/8] Bonus application preflight"
    & $pythonExe bonus_level\just_dance_app.py --check --no-auto-prepare
    if ($LASTEXITCODE -ne 0) { throw "Bonus application preflight failed." }

    Write-Host "[7/8] Mario application preflight"
    & $pythonExe bonus_level_mario\mario_camera_demo.py --check
    if ($LASTEXITCODE -ne 0) { throw "Mario application preflight failed." }

    Write-Host "[8/8] Beginner setup check"
    if (Test-Path -LiteralPath $lbfModel) {
        & $pythonExe beginner_level\tests\check_part2_setup.py
        if ($LASTEXITCODE -ne 0) { throw "Beginner setup check failed." }
    }
    else {
        Write-Warning "Skipped LBF runtime check; restore resources\face_models\lbfmodel.yaml first."
    }

    Write-Host "Project verification passed: 50 unit tests plus camera-free demo preflights."
}
finally {
    Pop-Location
}
