param(
    [ValidateSet("Validate", "DeepValidate", "Spectra", "Lock", "PrepareFold", "Smoke", "Fold", "Matrix", "DE", "SqrtJsd", "FisherRao", "Status", "Summarize", "All")]
    [string]$Stage = "Status",
    [string]$CondaEnv = "cmrd",
    [string]$RunRoot = "runs/faced_native_compact_base_seed42",
    [int[]]$Folds = @(),
    [switch]$Resume,
    [switch]$RetryFailed,
    [switch]$AllowPartial
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$Config = "configs/faced/native_compact_base.yaml"
$Runner = "scripts/run_faced_native_compact.py"

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

function Invoke-Spectra {
    Invoke-Python $Runner prepare-spectra --config $Config --run-root $RunRoot
}

function Invoke-Lock {
    Invoke-Python $Runner lock --config $Config --run-root $RunRoot
}

function Invoke-PrepareFold {
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
    $Arguments = @($Runner, "summarize", "--run-root", $RunRoot)
    if ($AllowPartial) { $Arguments += "--allow-partial" }
    Invoke-Python @Arguments
}

switch ($Stage) {
    "Validate" { Invoke-Validate }
    "DeepValidate" { Invoke-Validate -Deep }
    "Spectra" { Invoke-Spectra }
    "Lock" { Invoke-Lock }
    "PrepareFold" { Invoke-PrepareFold }
    "Smoke" { Invoke-Python $Runner smoke --config $Config --run-root $RunRoot --smoke-epochs 2 }
    "Fold" { Invoke-Matrix @("de_base", "native_sqrt_jsd_base", "native_fisher_rao_base") }
    "Matrix" { Invoke-Matrix @("de_base", "native_sqrt_jsd_base", "native_fisher_rao_base") }
    "DE" { Invoke-Matrix @("de_base") }
    "SqrtJsd" { Invoke-Matrix @("native_sqrt_jsd_base") }
    "FisherRao" { Invoke-Matrix @("native_fisher_rao_base") }
    "Status" { Invoke-Status }
    "Summarize" { Invoke-Summarize }
    "All" {
        Invoke-Validate
        Invoke-Spectra
        Invoke-Lock
        Invoke-Matrix @("de_base", "native_sqrt_jsd_base", "native_fisher_rao_base")
        Invoke-Summarize
    }
}
