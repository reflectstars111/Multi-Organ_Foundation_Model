$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$env:FLOWREPO_PARALLEL_WORKERS = '2'
Set-Location $Project
conda run --no-capture-output -p (Join-Path $Project '.r_flowrepo') Rscript download.r
exit $LASTEXITCODE
