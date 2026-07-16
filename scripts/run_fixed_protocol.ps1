param(
    [ValidateSet("Prepare", "Validate", "LockEpoch", "Smoke", "Classic", "MLP", "Transformer", "Hierarchical", "Matrix", "Mechanism", "Summarize", "Status", "All")]
    [string]$Stage = "Status",
    [string]$CondaEnv = "bilstm",
    [string]$RunRoot = "runs/fixed_protocol_seed42",
    [switch]$Resume,
    [switch]$RetryFailed
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$SeedConfig = "configs/fixed_protocol/seed_rjsd_1s1s.yaml"
$SeedIvConfig = "configs/fixed_protocol/seediv_rjsd_1s1s.yaml"
$Runner = "scripts/run_fixed_protocol.py"

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & conda run --no-capture-output -n $CondaEnv python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

function Invoke-Prepare {
    Invoke-Python scripts/preprocess_seed_de_rjsd_ica.py --config $SeedConfig --stage all --window-seconds 1 --hop-seconds 1 --strict-ica --resume
    Invoke-Python scripts/preprocess_seediv_de_rjsd_ica.py --config $SeedIvConfig --stage all --window-seconds 1 --hop-seconds 1 --strict-ica --resume
}

function Invoke-Validate {
    Invoke-Python $Runner validate-cache --config $SeedConfig
    Invoke-Python $Runner validate-cache --config $SeedIvConfig
}

function Invoke-LockEpoch {
    Invoke-Python $Runner lock-epoch --config $SeedConfig --run-root $RunRoot
    Invoke-Python $Runner lock-epoch --config $SeedIvConfig --run-root $RunRoot
}

function Invoke-Smoke {
    Invoke-Python $Runner smoke --config $SeedConfig --run-root "${RunRoot}_smoke" --resume --retry-failed
    Invoke-Python $Runner smoke --config $SeedIvConfig --run-root "${RunRoot}_smoke" --resume --retry-failed
}

function Invoke-Matrix {
    $Extra = @()
    if ($Resume) { $Extra += "--resume" }
    if ($RetryFailed) { $Extra += "--retry-failed" }
    Invoke-Python $Runner matrix --config $SeedConfig --run-root $RunRoot @Extra
    Invoke-Python $Runner matrix --config $SeedIvConfig --run-root $RunRoot @Extra
}

function Invoke-MLP {
    $Extra = @("--model", "small_mlp")
    if ($Resume) { $Extra += "--resume" }
    if ($RetryFailed) { $Extra += "--retry-failed" }
    Invoke-Python $Runner matrix --config $SeedConfig --run-root $RunRoot @Extra
    Invoke-Python $Runner matrix --config $SeedIvConfig --run-root $RunRoot @Extra
}

function Invoke-ModelStage {
    param([Parameter(Mandatory = $true)][string[]]$Models)
    $Extra = @()
    foreach ($Model in $Models) { $Extra += @("--model", $Model) }
    if ($Resume) { $Extra += "--resume" }
    if ($RetryFailed) { $Extra += "--retry-failed" }
    Invoke-Python $Runner matrix --config $SeedConfig --run-root $RunRoot @Extra
    Invoke-Python $Runner matrix --config $SeedIvConfig --run-root $RunRoot @Extra
}

function Invoke-Mechanism {
    Invoke-Python $Runner mechanism --config $SeedConfig --run-root $RunRoot
    Invoke-Python $Runner mechanism --config $SeedIvConfig --run-root $RunRoot
}

function Invoke-Summarize {
    Invoke-Python $Runner summarize --run-root $RunRoot
}

function Invoke-Status {
    Invoke-Python $Runner status --run-root $RunRoot
}

switch ($Stage) {
    "Prepare" { Invoke-Prepare }
    "Validate" { Invoke-Validate }
    "LockEpoch" { Invoke-LockEpoch }
    "Smoke" { Invoke-Smoke }
    "Classic" { Invoke-ModelStage -Models @("logistic_regression", "linear_svm") }
    "MLP" { Invoke-MLP }
    "Transformer" { Invoke-ModelStage -Models @("plain_transformer") }
    "Hierarchical" { Invoke-ModelStage -Models @("hierarchical_attention") }
    "Matrix" { Invoke-Matrix }
    "Mechanism" { Invoke-Mechanism }
    "Summarize" { Invoke-Summarize }
    "Status" { Invoke-Status }
    "All" {
        Invoke-Prepare
        Invoke-Validate
        Invoke-LockEpoch
        Invoke-Smoke
        Invoke-Matrix
        Invoke-Mechanism
        Invoke-Summarize
    }
}
