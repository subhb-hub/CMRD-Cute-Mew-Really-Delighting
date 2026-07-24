param(
    [ValidateSet("Validate", "Smoke", "PrepareBase", "PreparePack", "PrepareAtlas", "PilotQuick", "Pilot", "KScreen", "FeatureA", "ArchitectureB", "LandmarkC", "Strict", "Full", "Status", "Summarize")]
    [string]$Stage = "Status",
    [ValidateSet("Both", "FACED", "SEEDIV")]
    [string]$Dataset = "Both",
    [string]$CondaEnv = "cmrd",
    [string]$RunRoot = "runs/slst_v1",
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Runner = "scripts/run_slst.py"
$FacedConfig = "configs/slst/faced_v1.yaml"
$SeedIvConfig = "configs/slst/seediv_v1.yaml"

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
    foreach ($Fold in 1..$Maximum) { $Arguments += @("--fold", "$Fold") }
    return $Arguments
}

function Invoke-Configured {
    param([string]$Command)
    foreach ($Config in (Get-Configs)) {
        Invoke-Python $Runner $Command --config $Config --run-root $RunRoot
    }
}

function Invoke-Matrix {
    param(
        [string[]]$Conditions,
        [string[]]$Architectures = @("B4_slst"),
        [int[]]$Seeds = @(42, 52, 62),
        [switch]$AllFolds,
        [string[]]$Protocols = @(),
        [int]$Landmarks = 8,
        [string]$OnlyConfig = ""
    )
    $Configs = if ($OnlyConfig) { @($OnlyConfig) } else { Get-Configs }
    foreach ($Config in $Configs) {
        $Arguments = @($Runner, "matrix", "--config", $Config, "--run-root", $RunRoot, "--set", "slst.landmarks=$Landmarks")
        if ($AllFolds) { $Arguments += Get-AllFoldArguments -Config $Config }
        foreach ($Protocol in $Protocols) { $Arguments += @("--protocol", $Protocol) }
        foreach ($Condition in $Conditions) { $Arguments += @("--condition", $Condition) }
        foreach ($Architecture in $Architectures) { $Arguments += @("--architecture", $Architecture) }
        foreach ($Seed in $Seeds) { $Arguments += @("--seed", "$Seed") }
        if ($NoResume) { $Arguments += "--no-resume" }
        Invoke-Python @Arguments
    }
}

$PilotConditions = @("A1_de", "A2_full_shape", "A3_scalar_rjsd", "A6_hilbert_landmark", "C2_learnable", "C4_regularized")
$FeatureConditions = @("A0_magnitude", "A1_de", "A2_full_shape", "A3_scalar_rjsd", "A4_raw_landmark", "A5_centered_landmark", "A6_hilbert_landmark")
$LandmarkConditions = @("A6_hilbert_landmark", "C1_random_learnable", "C2_learnable", "C3_anchor", "C4_regularized")
$AllArchitectures = @("B0_flatten_mlp", "B1_flatten_temporal", "B2_band_temporal", "B3_channel_temporal", "B4_slst")

switch ($Stage) {
    "Validate" { Invoke-Configured -Command "validate" }
    "Smoke" { Invoke-Configured -Command "smoke" }
    "PrepareBase" { Invoke-Configured -Command "prepare-base" }
    "PreparePack" { Invoke-Configured -Command "prepare-pack" }
    "PrepareAtlas" { Invoke-Configured -Command "prepare-atlas" }
    "PilotQuick" { Invoke-Matrix -Conditions $PilotConditions -Seeds @(42) }
    "Pilot" { Invoke-Matrix -Conditions $PilotConditions }
    "KScreen" {
        foreach ($K in @(4, 8, 16)) {
            Invoke-Matrix -Conditions @("A6_hilbert_landmark") -Seeds @(42) -Landmarks $K
        }
    }
    "FeatureA" { Invoke-Matrix -Conditions $FeatureConditions -Seeds @(42) }
    "ArchitectureB" { Invoke-Matrix -Conditions @("A6_hilbert_landmark") -Architectures $AllArchitectures -Seeds @(42) }
    "LandmarkC" { Invoke-Matrix -Conditions $LandmarkConditions -Seeds @(42) }
    "Strict" {
        foreach ($Rotation in 0..2) {
            Invoke-Matrix -Conditions @("A1_de", "A6_hilbert_landmark", "C4_regularized") -AllFolds -Protocols @("subject_stimulus_rotation_$Rotation") -OnlyConfig $FacedConfig
        }
    }
    "Full" { Invoke-Matrix -Conditions @("A1_de", "A2_full_shape", "A3_scalar_rjsd", "A6_hilbert_landmark", "C4_regularized") -AllFolds }
    "Status" { Invoke-Python $Runner status --run-root $RunRoot }
    "Summarize" { Invoke-Python $Runner summarize --run-root $RunRoot }
}
