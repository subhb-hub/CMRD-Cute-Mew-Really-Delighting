param(
    [ValidateSet("Validate", "Smoke", "PrepareBase", "PrepareFeatures", "Pilot", "Core", "Full", "Status", "Summarize")]
    [string]$Stage = "Status",
    [ValidateSet("Both", "FACED", "SEEDIV")]
    [string]$Dataset = "Both",
    [string]$CondaEnv = "cmrd",
    [string]$RunRoot = "runs/spectral_atlas_v1_seed42",
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Runner = "scripts/run_spectral_atlas.py"
$FacedConfig = "configs/spectral_atlas/faced_v1.yaml"
$SeedIvConfig = "configs/spectral_atlas/seediv_v1.yaml"

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & conda run --no-capture-output -n $CondaEnv python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Get-Configs {
    switch ($Dataset) {
        "FACED" { return @($FacedConfig) }
        "SEEDIV" { return @($SeedIvConfig) }
        default { return @($FacedConfig, $SeedIvConfig) }
    }
}

function Get-AllFoldArguments {
    param([string]$Config)
    $Arguments = @()
    $Maximum = if ($Config -eq $FacedConfig) { 10 } else { 15 }
    foreach ($Fold in 1..$Maximum) {
        $Arguments += @("--fold", "$Fold")
    }
    return $Arguments
}

function Invoke-Configured {
    param([string]$Command, [string[]]$Extra = @())
    foreach ($Config in (Get-Configs)) {
        Invoke-Python $Runner $Command --config $Config --run-root $RunRoot @Extra
    }
}

function Invoke-Matrix {
    param([string[]]$Conditions, [string[]]$Models, [switch]$AllFolds)
    foreach ($Config in (Get-Configs)) {
        $Arguments = @($Runner, "matrix", "--config", $Config, "--run-root", $RunRoot)
        if ($AllFolds) {
            $Arguments += Get-AllFoldArguments -Config $Config
        }
        foreach ($Condition in $Conditions) {
            $Arguments += @("--condition", $Condition)
        }
        foreach ($Model in $Models) {
            $Arguments += @("--model", $Model)
        }
        if ($NoResume) {
            $Arguments += "--no-resume"
        }
        Invoke-Python @Arguments
    }
}

$PilotConditions = @(
    "de",
    "scalar_jsd_power",
    "nystrom_landmark_power_cap4",
    "ilr_power_full",
    "log_psd_full",
    "pca_ilr_power_cap4",
    "random_ilr_power_cap4"
)

$CoreConditions = @(
    "de",
    "log_band_power",
    "scalar_jsd_power",
    "ilr_power_full",
    "log_psd_full",
    "raw_landmark_power_cap1",
    "nystrom_landmark_power_cap1",
    "pca_ilr_power_cap1",
    "random_ilr_power_cap1",
    "raw_landmark_power_cap2",
    "nystrom_landmark_power_cap2",
    "pca_ilr_power_cap2",
    "random_ilr_power_cap2",
    "raw_landmark_power_cap4",
    "nystrom_landmark_power_cap4",
    "pca_ilr_power_cap4",
    "random_ilr_power_cap4",
    "raw_landmark_power_cap8",
    "nystrom_landmark_power_cap8",
    "pca_ilr_power_cap8",
    "random_ilr_power_cap8"
)

switch ($Stage) {
    "Validate" { Invoke-Configured -Command "validate" }
    "Smoke" { Invoke-Configured -Command "smoke" }
    "PrepareBase" { Invoke-Configured -Command "prepare-base" }
    "PrepareFeatures" { Invoke-Configured -Command "prepare-features" }
    "Pilot" { Invoke-Matrix -Conditions $PilotConditions -Models @("pooled_mlp") }
    "Core" { Invoke-Matrix -Conditions $CoreConditions -Models @("pooled_mlp") }
    "Full" { Invoke-Matrix -Conditions $CoreConditions -Models @("logistic_regression", "linear_svm", "pooled_mlp") -AllFolds }
    "Status" { Invoke-Python $Runner status --run-root $RunRoot }
    "Summarize" { Invoke-Python $Runner summarize --run-root $RunRoot }
}
