param(
    [ValidateSet("Validate", "DeepValidate", "Lock", "PrepareFeatures", "Smoke", "Matrix", "VectorRJSD", "FisherFull", "Status", "Summarize", "All")]
    [string]$Stage = "Status",
    [string]$CondaEnv = "cmrd",
    [string]$RunRoot = "runs/faced_vector_preserving_monitor_seed42",
    [int[]]$Folds = @(1),
    [switch]$Resume,
    [switch]$RetryFailed,
    [switch]$AllowPartial
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Config = "configs/faced/vector_preserving_monitor.yaml"
$Runner = "scripts/run_faced_vector_preserving.py"

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & conda run --no-capture-output -n $CondaEnv python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Add-Folds {
    param([string[]]$Arguments)
    $Result = @($Arguments)
    foreach ($Fold in $Folds) {
        if ($Fold -lt 1 -or $Fold -gt 10) { throw "FACED fold must be between 1 and 10" }
        $Result += @("--fold", "$Fold")
    }
    return $Result
}

function Invoke-Validate {
    param([switch]$Deep)
    $Arguments = @($Runner, "validate-data", "--config", $Config, "--run-root", $RunRoot)
    if ($Deep) { $Arguments += "--deep" }
    Invoke-Python @Arguments
}

function Invoke-Lock {
    Invoke-Python $Runner lock --config $Config --run-root $RunRoot
}

function Invoke-PrepareFeatures {
    $Arguments = Add-Folds @($Runner, "prepare-features", "--config", $Config, "--run-root", $RunRoot)
    Invoke-Python @Arguments
}

function Invoke-Matrix {
    param([string[]]$Conditions)
    $Arguments = Add-Folds @($Runner, "matrix", "--config", $Config, "--run-root", $RunRoot)
    foreach ($Condition in $Conditions) { $Arguments += @("--condition", $Condition) }
    if ($Resume) { $Arguments += "--resume" }
    if ($RetryFailed) { $Arguments += "--retry-failed" }
    Invoke-Python @Arguments
}

function Invoke-Status {
    Invoke-Python $Runner status --run-root $RunRoot
}

function Invoke-Summarize {
    param([switch]$Partial)
    $Arguments = @($Runner, "summarize", "--run-root", $RunRoot)
    if ($AllowPartial -or $Partial) { $Arguments += "--allow-partial" }
    Invoke-Python @Arguments
}

$BothConditions = @(
    "frequency_vector_rjsd_base",
    "fisher_rao_supervised_full_vector_base"
)

switch ($Stage) {
    "Validate" { Invoke-Validate }
    "DeepValidate" { Invoke-Validate -Deep }
    "Lock" { Invoke-Lock }
    "PrepareFeatures" { Invoke-PrepareFeatures }
    "Smoke" { Invoke-Python $Runner smoke --config $Config --run-root $RunRoot --smoke-epochs 1 }
    "Matrix" { Invoke-Matrix $BothConditions }
    "VectorRJSD" { Invoke-Matrix @("frequency_vector_rjsd_base") }
    "FisherFull" { Invoke-Matrix @("fisher_rao_supervised_full_vector_base") }
    "Status" { Invoke-Status }
    "Summarize" { Invoke-Summarize }
    "All" {
        Invoke-Validate
        Invoke-Lock
        Invoke-Matrix $BothConditions
        Invoke-Summarize -Partial
    }
}
