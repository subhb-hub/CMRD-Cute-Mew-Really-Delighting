param(
    [ValidateSet("Validate", "Lock", "Smoke", "Fold1", "Training", "Representation", "Capacity", "Main", "Control", "Matrix", "Summarize", "Status", "All")]
    [string]$Stage = "Status",
    [string]$CondaEnv = "bilstm",
    [string]$RunRoot = "runs/srjsd_large_v1_seed42",
    [switch]$Resume,
    [switch]$RetryFailed,
    [switch]$AllowPartial
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$SeedConfig = "configs/srjsd_large/seed_v1.yaml"
$SeedIvConfig = "configs/srjsd_large/seediv_v1.yaml"
$Runner = "scripts/run_srjsd_large.py"

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
    Invoke-Python $Runner validate-cache --config $SeedConfig
    Invoke-Python $Runner validate-cache --config $SeedIvConfig
}

function Invoke-Lock {
    Invoke-Python $Runner lock --config $SeedConfig --run-root $RunRoot
    Invoke-Python $Runner lock --config $SeedIvConfig --run-root $RunRoot
}

function Invoke-Smoke {
    Invoke-Python $Runner smoke --config $SeedConfig --run-root "${RunRoot}_smoke" --condition d_srjsd_large_v2 --fold 1 --smoke-epochs 2 --resume --retry-failed
    Invoke-Python $Runner smoke --config $SeedIvConfig --run-root "${RunRoot}_smoke" --condition d_srjsd_large_v2 --fold 1 --smoke-epochs 2 --resume --retry-failed
}

function Invoke-Conditions {
    param(
        [Parameter(Mandatory = $true)][string[]]$Conditions,
        [int[]]$Folds = @()
    )
    $Extra = Get-ResumeArguments
    foreach ($Config in @($SeedConfig, $SeedIvConfig)) {
        $Arguments = @($Runner, "matrix", "--config", $Config, "--run-root", $RunRoot)
        foreach ($Condition in $Conditions) { $Arguments += @("--condition", $Condition) }
        foreach ($Fold in $Folds) { $Arguments += @("--fold", "$Fold") }
        $Arguments += $Extra
        Invoke-Python @Arguments
    }
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
    "Lock" { Invoke-Lock }
    "Smoke" { Invoke-Smoke }
    "Fold1" { Invoke-Conditions -Conditions @("a2_rjsd_base_v2", "b_srjsd_base_v2", "c_rjsd_large_v2", "d_srjsd_large_v2", "e_de_zscore_large_v2") -Folds @(1) }
    "Training" { Invoke-Conditions -Conditions @("a2_rjsd_base_v2") }
    "Representation" { Invoke-Conditions -Conditions @("b_srjsd_base_v2") }
    "Capacity" { Invoke-Conditions -Conditions @("c_rjsd_large_v2") }
    "Main" { Invoke-Conditions -Conditions @("d_srjsd_large_v2") }
    "Control" { Invoke-Conditions -Conditions @("e_de_zscore_large_v2") }
    "Matrix" { Invoke-Conditions -Conditions @("a2_rjsd_base_v2", "b_srjsd_base_v2", "c_rjsd_large_v2", "d_srjsd_large_v2", "e_de_zscore_large_v2") }
    "Summarize" { Invoke-Summarize }
    "Status" { Invoke-Status }
    "All" {
        Invoke-Validate
        Invoke-Lock
        Invoke-Smoke
        Invoke-Conditions -Conditions @("a2_rjsd_base_v2", "b_srjsd_base_v2", "c_rjsd_large_v2", "d_srjsd_large_v2", "e_de_zscore_large_v2")
        Invoke-Summarize
    }
}
