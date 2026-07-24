param(
    [ValidateSet('run','smoke','status')]
    [string]$Command = 'run',
    [switch]$Force
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = 'C:\Users\Lin\miniconda3\envs\CMRD\python.exe'
$config = Join-Path $repoRoot 'configs\faced\str_jsd_fold1_light.yaml'
$entrypoint = Join-Path $repoRoot 'scripts\run_faced_str_jsd.py'
$arguments = @($entrypoint, $Command, '--config', $config)
if ($Force) { $arguments += '--force' }
& $python @arguments
exit $LASTEXITCODE
