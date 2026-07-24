param(
    [ValidateSet("Run", "Status")]
    [string]$Stage = "Run",
    [string]$Config = "configs/faced/normative_probe_fold1.yaml",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$Command = $Stage.ToLowerInvariant()
$Arguments = @("run", "--no-capture-output", "-n", "bilstm", "python", "scripts/run_faced_normative_probe.py", $Command, "--config", $Config)
if ($Force) {
    $Arguments += "--force"
}
& conda @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "FACED normative probe failed with exit code $LASTEXITCODE"
}

