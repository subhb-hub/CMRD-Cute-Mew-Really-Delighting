param(
    [ValidateSet("Both", "FACED", "SEEDIV")]
    [string]$Dataset = "Both",
    [string]$CondaEnv = "cmrd",
    [string]$RunRoot = "runs/slst_direction_v2",
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$MainScript = Join-Path $PSScriptRoot "run_slst_direction_v2.ps1"

function Invoke-Stage {
    param([string]$Stage)
    $Arguments = @(
        "-ExecutionPolicy", "Bypass",
        "-File", $MainScript,
        "-Stage", $Stage,
        "-Dataset", $Dataset,
        "-CondaEnv", $CondaEnv,
        "-RunRoot", $RunRoot
    )
    if ($NoResume) {
        $Arguments += "-NoResume"
    }
    Write-Host ""
    Write-Host ("========== SLST Direction-v2: {0} ==========" -f $Stage) -ForegroundColor Green
    & powershell @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Stage $Stage failed with exit code $LASTEXITCODE"
    }
}

Write-Host "SLST Direction-v2 one-click exploratory run" -ForegroundColor Green
Write-Host ("Dataset={0} RunRoot={1} Resume={2}" -f $Dataset, $RunRoot, (-not $NoResume))
Write-Host "Pipeline: Validate -> Smoke -> CoordinateGate"

Invoke-Stage -Stage "Validate"
Invoke-Stage -Stage "Smoke"
Invoke-Stage -Stage "CoordinateGate"

Write-Host ""
Write-Host "CoordinateGate finished. Review results before running later gates." -ForegroundColor Green
& powershell -ExecutionPolicy Bypass -File $MainScript -Stage Status -Dataset $Dataset -CondaEnv $CondaEnv -RunRoot $RunRoot
