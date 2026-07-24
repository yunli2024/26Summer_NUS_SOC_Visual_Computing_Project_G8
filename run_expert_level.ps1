param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExpertArgs
)

$entryPoint = Join-Path $PSScriptRoot "expert_level\task2_realtime.py"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    Write-Error "Python was not found. Activate the vc_sws3026 environment first."
    exit 1
}

& $pythonCommand.Source $entryPoint @ExpertArgs
exit $LASTEXITCODE
