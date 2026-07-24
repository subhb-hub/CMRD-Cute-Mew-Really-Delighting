param(
    [ValidateSet("Run", "Smoke", "Status")]
    [string]$Stage = "Run",
    [string]$Config = "configs/faced/subject_adversarial_fold1_light.yaml",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$Command = $Stage.ToLowerInvariant()
$Arguments = @(
    "run", "--no-capture-output", "-n", "bilstm",
    "python", "scripts/run_faced_subject_adversarial.py",
    $Command, "--config", $Config
)
if ($Force) {
    $Arguments += "--force"
}
& conda @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "FACED subject-adversarial stage failed with exit code $LASTEXITCODE"
}
