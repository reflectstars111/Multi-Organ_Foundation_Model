param([switch]$DryRun)

$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Python = (Get-Command python -ErrorAction Stop).Source
$DatasetFolder = Get-ChildItem -LiteralPath $Project -Directory |
    Where-Object { $_.Name.EndsWith('(Datasets)') } |
    Select-Object -First 1
if ($null -eq $DatasetFolder) { throw 'Dataset root not found' }
$Data = (Resolve-Path (Join-Path $DatasetFolder.FullName 'pointed_data')).Path
$Runs = Join-Path $Project 'runs'
$Moe = Join-Path $Runs 'region_paper5_ct7_2_v1'
$Baselines = Join-Path $Runs 'unet_paper5_ct7_2_v1'
$Logs = Join-Path $Runs 'paper5_ct7_2_local_v1-logs'
if (-not $DryRun) {
    New-Item -ItemType Directory -Path $Logs -Force | Out-Null
}
Set-Location $Project

function Invoke-Training {
    param([string]$Name, [string]$Module, [string[]]$Arguments)
    Write-Output "$(Get-Date -Format o) START $Name"
    if ($DryRun) {
        Write-Output "$Python -m $Module $($Arguments -join ' ')"
        return
    }
    $Stdout = Join-Path $Logs "$Name.stdout.log"
    $Stderr = Join-Path $Logs "$Name.stderr.log"
    & $Python -m $Module @Arguments 1> $Stdout 2> $Stderr
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE; inspect $Stderr"
    }
    Write-Output "$(Get-Date -Format o) COMPLETE $Name"
}

$Common = @('--data-root', $Data, '--epochs', '50', '--size', '256',
    '--base', '16', '--batch-size', '2', '--workers', '2', '--seed', '42',
    '--lr', '0.0003', '--abdomen-split', 'ct7_2', '--abdomen-labels', 'paper5',
    '--abdomen-positive-sampling', '0.5', '--device', 'cuda:0')

$MoeArgs = $Common + @('--output', $Moe, '--top-k', '1', '--quota-per-domain', '500')
if (Test-Path (Join-Path $Moe 'config.json')) { $MoeArgs += '--resume' }
Invoke-Training 'moe' 'unet_moe.region_train' $MoeArgs

$SharedArgs = $Common + @('--output', $Baselines, '--mode', 'shared', '--quota', '500')
if (Test-Path (Join-Path $Baselines 'records.json')) { $SharedArgs += '--resume' }
Invoke-Training 'shared' 'unet_moe.baseline_suite' $SharedArgs

foreach ($Domain in @('BUSI', 'MMOTU', 'DDTI', 'AUL', 'FALLMUD',
                      'Fetal_HC', 'CCA', 'CAMUS', 'AbdomenUS')) {
    $IndependentArgs = $Common + @('--output', $Baselines, '--mode', 'independent',
                                   '--domains', $Domain, '--quota', '500', '--resume')
    Invoke-Training "independent_$Domain" 'unet_moe.baseline_suite' $IndependentArgs
}

$SummaryArgs = $Common + @('--output', $Baselines, '--mode', 'summarize',
                            '--quota', '500', '--resume')
Invoke-Training 'baseline_summary' 'unet_moe.baseline_suite' $SummaryArgs
Invoke-Training 'three_way_summary' 'scripts.summarize_three_way_paper5' @('--runs-root', $Runs)
Write-Output "$(Get-Date -Format o) ALL COMPLETE"
