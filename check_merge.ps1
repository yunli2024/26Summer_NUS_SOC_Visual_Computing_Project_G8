$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    Write-Error "Python was not found. Activate the vc_sws3026 environment first."
    exit 1
}
$pythonExe = $pythonCommand.Source

& $pythonExe VisualComputingProject\beginner_level\tests\check_part2_setup.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\expert_level\tests\test_keypoint_features.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\expert_level\tests\test_realtime_stability.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe -m unittest discover -s VisualComputingProject\bonus_level\tests -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe -m unittest discover -s VisualComputingProject\bonus_level_mario -p "test_*.py" -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\expert_level\main.py inspect
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\bonus_level\main.py --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pythonExe VisualComputingProject\bonus_level_mario\mario_camera_demo.py --check
exit $LASTEXITCODE
