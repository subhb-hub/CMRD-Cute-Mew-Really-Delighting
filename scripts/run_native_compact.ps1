param(
    [ValidateSet("Validate", "Prepare", "Lock", "Smoke", "Fold1", "SqrtJsd", "FisherRao", "Wasserstein", "Summarize", "Status", "All")]
    [string]$Stage = "Status",
    [ValidateSet("Both", "Seed", "SeedIV")]
    [string]$Dataset = "Both",
    [string]$CondaEnv = "cmrd",
    [string]$RunRoot = "runs/native_compact_v1_seed42",
    [switch]$Resume,
    [switch]$RetryFailed,
    [switch]$AllowPartial
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$SeedConfig = "configs/native_compact/seed_v1.yaml"
$SeedIvConfig = "configs/native_compact/seediv_v1.yaml"
$Runner = "scripts/run_native_compact.py"

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & conda run --no-capture-output -n $CondaEnv python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Get-Configs {
    switch ($Dataset) {
        "Seed" { return @($SeedConfig) }
        "SeedIV" { return @($SeedIvConfig) }
        default { return @($SeedConfig, $SeedIvConfig) }
    }
}

function Get-ResumeArguments {
    $Extra = @()
    if ($Resume) { $Extra += "--resume" }
    if ($RetryFailed) { $Extra += "--retry-failed" }
    return $Extra
}

function Invoke-Validate {
    foreach ($Config in (Get-Configs)) {
        Invoke-Python $Runner validate-cache --config $Config
    }
}

function Invoke-Prepare {
    foreach ($Config in (Get-Configs)) {
        Invoke-Python $Runner prepare-features --config $Config --run-root $RunRoot
    }
}

function Invoke-Lock {
    foreach ($Config in (Get-Configs)) {
        Invoke-Python $Runner lock --config $Config --run-root $RunRoot
    }
}

function Invoke-Smoke {
    foreach ($Config in (Get-Configs)) {
        Invoke-Python $Runner smoke --config $Config --run-root "${RunRoot}_smoke" --condition a_native_sqrt_jsd_base_v2 --smoke-epochs 2 --resume --retry-failed
    }
}

function Invoke-Conditions {
    param([Parameter(Mandatory = $true)][string[]]$Conditions)
    $Extra = Get-ResumeArguments
    foreach ($Config in (Get-Configs)) {
        $Arguments = @($Runner, "matrix", "--config", $Config, "--run-root", $RunRoot)
        foreach ($Condition in $Conditions) {
            $Arguments += @("--condition", $Condition)
        }
        $Arguments += $Extra
        Invoke-Python @Arguments
    }
}

function Invoke-Status {
    Invoke-Python $Runner status --run-root $RunRoot
}

function Invoke-Summarize {
    $Extra = @()
    if ($AllowPartial -or $Dataset -ne "Both") { $Extra += "--allow-partial" }
    Invoke-Python $Runner summarize --run-root $RunRoot @Extra
}

switch ($Stage) {
    "Validate" { Invoke-Validate }
    "Prepare" { Invoke-Prepare }
    "Lock" { Invoke-Lock }
    "Smoke" { Invoke-Smoke }
    "Fold1" { Invoke-Conditions -Conditions @("a_native_sqrt_jsd_base_v2", "b_native_fisher_rao_pca_base_v2", "c_native_wasserstein1_base_v2") }
    "SqrtJsd" { Invoke-Conditions -Conditions @("a_native_sqrt_jsd_base_v2") }
    "FisherRao" { Invoke-Conditions -Conditions @("b_native_fisher_rao_pca_base_v2") }
    "Wasserstein" { Invoke-Conditions -Conditions @("c_native_wasserstein1_base_v2") }
    "Summarize" { Invoke-Summarize }
    "Status" { Invoke-Status }
    "All" {
        Invoke-Validate
        Invoke-Lock
        Invoke-Conditions -Conditions @("a_native_sqrt_jsd_base_v2", "b_native_fisher_rao_pca_base_v2", "c_native_wasserstein1_base_v2")
        Invoke-Summarize
    }
}
