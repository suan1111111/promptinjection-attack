param(
    [int]$Steps = 1000,
    [double]$Epsilon = 255,
    [switch]$ValidateOnly
)
$ErrorActionPreference = 'Stop'
$project = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$jobRoot = Join-Path $project ('outputs\radio_body_region_' + (Get-Date -Format 'yyyyMMdd_HHmmss_fff'))
$arguments = @('-u', 'pixel_attack_trials/optimize_qwen_pixels.py',
    '--image', 'data/datasets/meta_instruction/Soft-Prompts-Go-Hard-main/clean_images/0.png',
    '--csv', 'data/datasets/meta_instruction/Soft-Prompts-Go-Hard-main/instruction_data/0/Attack/dataset.csv',
    '--constraint', 'flower', '--mask-image', (Join-Path $PSScriptRoot 'radio_body_mask.png'),
    '--patch-x', '0', '--patch-y', '0', '--patch-width', '364', '--patch-height', '252',
    '--epsilon', [string]$Epsilon, '--alpha', '1', '--steps', [string]$Steps,
    '--questions-per-step', '4', '--max-train-rows', '40', '--max-visual-tokens', '128',
    '--target-mode', 'full', '--update-rule', 'revised', '--seed', '7', '--save-every', '50',
    '--eval-rows', '10', '--eval-max-new-tokens', '512', '--run-dir', (Join-Path $jobRoot 'run'))
if ($ValidateOnly) { $arguments += '--validate-only' }
$env:PYTHONUTF8 = '1'
Push-Location $project
try {
    & (Join-Path $project '.venv\Scripts\python.exe') @arguments
    if ($LASTEXITCODE -ne 0) { throw "Training/validation exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
