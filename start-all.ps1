$ErrorActionPreference = 'Stop'
$env:PYTHONUTF8 = '1'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$backendPython = Join-Path $root '.venv-ocr\Scripts\python.exe'
$node = Join-Path $root 'tools\node\node.exe'
$vite = Join-Path $root 'frontend\node_modules\vite\bin\vite.js'
$runDir = Join-Path $root '.run'

New-Item -ItemType Directory -Force -Path $runDir | Out-Null

if (!(Test-Path $python)) { throw "Python environment missing: $python" }
if (!(Test-Path $backendPython)) { throw 'Run ./setup-ocr-runtime.ps1 first to prepare the DeepSeek-OCR runtime.' }
if (!(Test-Path $node)) { throw "Node runtime missing: $node" }
if (!(Test-Path $vite)) { throw "Frontend dependencies missing: $vite" }

$llm = Start-Process -FilePath $python -ArgumentList 'backend\local_llm_server.py' `
  -WorkingDirectory $root -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $runDir 'llm.out.log') `
  -RedirectStandardError (Join-Path $runDir 'llm.err.log')

$backend = Start-Process -FilePath $backendPython -ArgumentList 'app.py' `
  -WorkingDirectory (Join-Path $root 'backend') -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $runDir 'backend.out.log') `
  -RedirectStandardError (Join-Path $runDir 'backend.err.log')

$frontend = Start-Process -FilePath $node -ArgumentList $vite, '--host', '127.0.0.1', '--port', '5173' `
  -WorkingDirectory (Join-Path $root 'frontend') -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput (Join-Path $runDir 'frontend.out.log') `
  -RedirectStandardError (Join-Path $runDir 'frontend.err.log')

@{ llm = $llm.Id; backend = $backend.Id; frontend = $frontend.Id } |
  ConvertTo-Json | Set-Content -Encoding UTF8 (Join-Path $runDir 'pids.json')

Start-Sleep -Seconds 4
Write-Host "PrivaSee started: http://127.0.0.1:5173/"
Write-Host "Backend: http://127.0.0.1:5000/api/health"
Write-Host "Local LLM: http://127.0.0.1:11434/health"

