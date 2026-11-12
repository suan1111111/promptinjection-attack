$ErrorActionPreference = 'Stop'
$project = 'D:\CodexProjects\PrivaSee'
$jobRoot = Join-Path $project ('outputs\radio_reference_patch_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $jobRoot | Out-Null
$arguments = @('-u', 'pixel_attack_trials/optimize_qwen_pixels.py',
 '--image','data/datasets/meta_instruction/Soft-Prompts-Go-Hard-main/clean_images/0.png',
 '--csv','data/datasets/meta_instruction/Soft-Prompts-Go-Hard-main/instruction_data/0/Attack/dataset.csv',
 '--constraint','flower','--flower-image','output/scene_patch_references_v2/assets/label.png',
 '--patch-x','18','--patch-y','121','--patch-width','72','--patch-height','56',
 '--epsilon','8','--alpha','1','--steps','50','--questions-per-step','4',
 '--max-train-rows','40','--max-visual-tokens','128','--target-mode','full',
 '--update-rule','revised','--seed','7','--save-every','10',
 '--eval-rows','5','--eval-max-new-tokens','256','--run-dir',($jobRoot+'\run'))
$env:PYTHONUTF8='1'
$process = Start-Process -FilePath ($project+'\.venv\Scripts\python.exe') -ArgumentList $arguments -WorkingDirectory $project -WindowStyle Hidden -RedirectStandardOutput ($jobRoot+'\stdout.log') -RedirectStandardError ($jobRoot+'\stderr.log') -PassThru
@{pid=$process.Id;directory=$jobRoot;arguments=$arguments;started=(Get-Date -Format o)} | ConvertTo-Json -Depth 4 | Set-Content -Encoding UTF8 ($project+'\output\scene_patch_references_v2\training_launch.json')
Write-Output "PID=$($process.Id) OUTPUT=$jobRoot"
