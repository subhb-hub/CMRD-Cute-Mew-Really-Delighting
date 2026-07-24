param(
    [ValidateSet("Validate", "Prepare", "Lock", "Smoke", "Fold1", "SqrtJsd", "FisherRao", "Summarize", "Status", "All")]
    [string]$Stage = "Status",
    [string]$CondaEnv = "cmrd",
    [string]$RunRoot = "runs/deap_native_compact_v1_seed42",
    [string]$CacheParent = "C:\Users\Lin\Documents\Arbitruam\Dataset\Processed\CMRD\deap\de_rjsd_ica_1s_hop1\b84bab9e4f721dbe",
    [switch]$Resume,
    [switch]$RetryFailed,
    [switch]$AllowPartial
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Config = "configs/native_compact/deap_v1.yaml"
$Runner = "scripts/run_deap_native_compact.py"

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & conda run --no-capture-output -n $CondaEnv python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Get-ResumeArguments {
    $Extra = @()
    if ($Resume) { $Extra += "--resume" }
    if ($RetryFailed) { $Extra += "--retry-failed" }
    return $Extra
}

function Invoke-Validate {
    Invoke-Python $Runner validate-cache --config $Config --cache-parent $CacheParent
}

function Invoke-Prepare {
    Invoke-Python $Runner prepare-features --config $Config --cache-parent $CacheParent --run-root $RunRoot
}

function Invoke-Lock {
    Invoke-Python $Runner lock --config $Config --cache-parent $CacheParent --run-root $RunRoot
}

function Invoke-Conditions {
    param([Parameter(Mandatory = $true)][string[]]$Conditions)
    $Arguments = @(
        $Runner, "matrix", "--config", $Config,
        "--cache-parent", $CacheParent, "--run-root", $RunRoot
    )
    foreach ($Condition in $Conditions) {
        $Arguments += @("--condition", $Condition)
    }
    $Arguments += Get-ResumeArguments
    Invoke-Python @Arguments
}

function Invoke-Smoke {
    Invoke-Python $Runner smoke --config $Config --cache-parent $CacheParent `
        --run-root "${RunRoot}_smoke" --condition a_native_sqrt_jsd_base_v2 `
        --smoke-epochs 2 --resume --retry-failed
}

function Invoke-Status {
    Invoke-Python $Runner status --run-root $RunRoot
}

function Invoke-Summarize {
    $Extra = @()
    if ($AllowPartial) { $Extra += "--allow-partial" }
    Invoke-Python $Runner summarize --run-root $RunRoot @Extra
}

switch ($Stage) {
    "Validate" { Invoke-Validate }
    "Prepare" { Invoke-Prepare }
    "Lock" { Invoke-Lock }
    "Smoke" { Invoke-Smoke }
    "Fold1" { Invoke-Conditions -Conditions @("a_native_sqrt_jsd_base_v2", "b_native_fisher_rao_pca_base_v2") }
    "SqrtJsd" { Invoke-Conditions -Conditions @("a_native_sqrt_jsd_base_v2") }
    "FisherRao" { Invoke-Conditions -Conditions @("b_native_fisher_rao_pca_base_v2") }
    "Summarize" { Invoke-Summarize }
    "Status" { Invoke-Status }
    "All" {
        Invoke-Validate
        Invoke-Lock
        Invoke-Conditions -Conditions @("a_native_sqrt_jsd_base_v2", "b_native_fisher_rao_pca_base_v2")
        Invoke-Summarize
    }
}
