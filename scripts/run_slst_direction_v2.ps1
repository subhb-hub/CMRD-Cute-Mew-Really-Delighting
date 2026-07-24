param(
    [ValidateSet("Validate", "Smoke", "PreparePack", "CoordinateGate", "StabilityGate", "CoverageGate", "LearnabilityGate", "Status", "Summarize")]
    [string]$Stage = "Status",
    [ValidateSet("Both", "FACED", "SEEDIV")]
    [string]$Dataset = "Both",
    [string]$CondaEnv = "cmrd",
    [string]$RunRoot = "runs/slst_direction_v2",
    [string]$BestCondition = "H4_stable_hilbert_lowrank_explicit",
    [switch]$NoResume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Runner = "scripts/run_slst.py"
$FacedConfig = "configs/slst/faced_direction_v2.yaml"
$SeedIvConfig = "configs/slst/seediv_direction_v2.yaml"

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    Write-Host ("[{0}] conda run -n {1} python {2}" -f (Get-Date -Format "HH:mm:ss"), $CondaEnv, ($Arguments -join " ")) -ForegroundColor Cyan
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

function Invoke-Configured {
    param([string]$Command)
    foreach ($Config in (Get-Configs)) {
        Invoke-Python $Runner $Command --config $Config --run-root $RunRoot
    }
}

function Invoke-DirectionMatrix {
    param(
        [string[]]$Conditions,
        [int]$Landmarks = 8,
        [int]$DirectionRank = 4,
        [double]$EigenvalueFloor = 0.001
    )
    if ($DirectionRank -gt $Landmarks) {
        throw "DirectionRank ($DirectionRank) cannot exceed Landmarks ($Landmarks)"
    }
    foreach ($Config in (Get-Configs)) {
        $Arguments = @(
            $Runner, "matrix", "--config", $Config, "--run-root", $RunRoot,
            "--set", "slst.landmarks=$Landmarks",
            "--set", "slst.direction_rank=$DirectionRank",
            "--set", "slst.eigenvalue_floor_ratio=$EigenvalueFloor",
            "--architecture", "B4_slst", "--seed", "42"
        )
        foreach ($Condition in $Conditions) {
            $Arguments += @("--condition", $Condition)
        }
        if ($NoResume) {
            $Arguments += "--no-resume"
        }
        Invoke-Python @Arguments
    }
}

$CoordinateConditions = @(
    "H0_scalar_explicit",
    "H1_raw_inner_explicit",
    "H2_pca_lowrank_explicit",
    "H3_hilbert_lowrank_explicit",
    "H4_stable_hilbert_lowrank_explicit",
    "H5_hilbert_full_explicit",
    "H6_stable_hilbert_lowrank_residual"
)
$LearnabilityConditions = @(
    "L0_fixed",
    "L1_lr3e5_freeze3",
    "L2_lr1e4_freeze3",
    "L3_lr3e4_freeze3",
    "L4_lr1e4_unfrozen",
    "L5_lr1e4_regularized"
)

switch ($Stage) {
    "Validate" { Invoke-Configured -Command "validate" }
    "Smoke" { Invoke-Configured -Command "smoke" }
    "PreparePack" { Invoke-Configured -Command "prepare-pack" }
    "CoordinateGate" {
        # Fixed K=8/r=4: explicit scalar, raw/PCA/Hilbert/stable-Hilbert, full-rank and residual ablations.
        Invoke-DirectionMatrix -Conditions $CoordinateConditions -Landmarks 8 -DirectionRank 4 -EigenvalueFloor 0.001
    }
    "StabilityGate" {
        # Diagnose rank and whitening-floor sensitivity before changing atlas coverage.
        foreach ($Rank in @(2, 4)) {
            foreach ($Floor in @(0.01, 0.001, 0.0001)) {
                Invoke-DirectionMatrix -Conditions @("H4_stable_hilbert_lowrank_explicit") -Landmarks 8 -DirectionRank $Rank -EigenvalueFloor $Floor
            }
        }
    }
    "CoverageGate" {
        # Run only after CoordinateGate identifies the best fixed coordinate family.
        $Pairs = @(
            @(4, 2), @(4, 4),
            @(8, 2), @(8, 4), @(8, 8),
            @(16, 2), @(16, 4), @(16, 8)
        )
        foreach ($Pair in $Pairs) {
            Invoke-DirectionMatrix -Conditions @($BestCondition) -Landmarks $Pair[0] -DirectionRank $Pair[1] -EigenvalueFloor 0.001
        }
    }
    "LearnabilityGate" {
        # L0-L5 share initialization/batch order; per-epoch landmark gradients and drift are written and printed.
        Invoke-DirectionMatrix -Conditions $LearnabilityConditions -Landmarks 8 -DirectionRank 4 -EigenvalueFloor 0.001
    }
    "Status" { Invoke-Python $Runner status --run-root $RunRoot }
    "Summarize" { Invoke-Python $Runner summarize --run-root $RunRoot }
}
