param(
    [ValidateSet("Smoke", "InnerCV", "All", "Status")]
    [string]$Stage = "Status",
    [string]$Config = "configs/faced/psd_jsd_recommended.yaml"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = "C:\Users\Lin\miniconda3\envs\cmrd\python.exe"
$stageMap = @{
    Smoke = "smoke"
    InnerCV = "inner-cv"
    All = "all"
    Status = "status"
}

Push-Location $repoRoot
try {
    & $python "scripts/run_faced_psd_jsd_recommended.py" --config $Config --stage $stageMap[$Stage]
    if ($LASTEXITCODE -ne 0) {
        throw "FACED PSD-JSD recommended experiment failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

