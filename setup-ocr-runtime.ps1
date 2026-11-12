$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root '.venv\Scripts\python.exe'
$ocrRoot = Join-Path $root '.venv-ocr'
& $python -m venv --without-pip $ocrRoot
if ($LASTEXITCODE -ne 0) { throw 'OCR environment creation failed' }
$sharedPackages = Join-Path $root '.venv\Lib\site-packages'
$sharedPythonLiteral = ConvertTo-Json -InputObject $sharedPackages -Compress
# Process shared package .pth files as well (pywin32 adds its DLL/module paths there).
$sharedBootstrap = "import site; site.addsitedir($sharedPythonLiteral)" + [Environment]::NewLine
[System.IO.File]::WriteAllText((Join-Path $ocrRoot 'Lib\site-packages\privasee-shared.pth'), $sharedBootstrap, (New-Object System.Text.UTF8Encoding($false)))
$env:PIP_CACHE_DIR = Join-Path $root '.pip-cache'
& $python -m pip --python (Join-Path $ocrRoot 'Scripts\python.exe') install --no-deps -r (Join-Path $root 'backend\requirements-ocr-runtime.txt')
if ($LASTEXITCODE -ne 0) { throw 'OCR dependency installation failed' }
Write-Host 'OCR runtime ready; model weights and PyTorch are shared with the main environment.'
