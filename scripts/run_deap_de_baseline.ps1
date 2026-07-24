param(
    [string]$CondaEnv = "cmrd",
    [string]$RunRoot = "runs/deap_de_baseline_v1_seed42",
    [string]$CacheParent = "C:\Users\Lin\Documents\Arbitruam\Dataset\Processed\CMRD\deap\de_rjsd_ica_1s_hop1\b84bab9e4f721dbe",
    [int]$SmokeEpochs = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Arguments = @(
    "scripts/run_deap_de_baseline.py",
    "--config", "configs/native_compact/deap_v1.yaml",
    "--cache-parent", $CacheParent,
    "--run-root", $RunRoot
)
if ($SmokeEpochs -gt 0) {
    $Arguments += @("--smoke-epochs", [string]$SmokeEpochs)
}

& conda run --no-capture-output -n $CondaEnv python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "DEAP DE baseline failed with exit code $LASTEXITCODE"
}
