param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExpertArgs
)

$entryPoint = Join-Path $PSScriptRoot "VisualComputingProject\expert_level\main.py"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    Write-Error "Python was not found. Activate the vc_sws3026 environment first."
    exit 1
}

& $pythonCommand.Source $entryPoint @ExpertArgs
exit $LASTEXITCODE
