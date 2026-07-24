$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    Write-Error "Python was not found. Activate the vc_sws3026 environment first."
    exit 1
}
$pythonExe = $pythonCommand.Source

& $pythonExe VisualComputingProject\beginner_level\tests\check_part2_setup.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\expert_level\test_expression_features.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\bonus_level\test_dance_scoring.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\expert_level\task1_pipeline.py --help
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\expert_level\task2_realtime.py --preview `
    tmp\expert_effects_preview.png
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\bonus_level\pose_analyzer.py --help
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\bonus_level\just_dance_app.py --help
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\bonus_level_mario\mario_camera_demo.py --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$reference = "VisualComputingProject\bonus_level\task2_results\dance_example_1\annotated.mp4"
$cache = "VisualComputingProject\bonus_level\task2_results\dance_example_1\pose_cache.npz"
if ((Test-Path -LiteralPath $reference) -and (Test-Path -LiteralPath $cache)) {
    & $pythonExe VisualComputingProject\bonus_level\just_dance_app.py --check
    exit $LASTEXITCODE
}

Write-Warning "Bonus Task 2 runtime cache is not generated; see RUN_GUIDE.md."
exit 0
