param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BonusArgs
)

$entryPoint = Join-Path $PSScriptRoot "VisualComputingProject\bonus_level\danceapp.py"
$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCommand) {
    Write-Error "Python was not found. Activate the vc_sws3026 environment first."
    exit 1
}

& $pythonCommand.Source $entryPoint @BonusArgs
exit $LASTEXITCODE
