$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $root '.run\pids.json'
if (!(Test-Path $pidFile)) {
  Write-Host 'No PrivaSee PID file found.'
  exit 0
}

$saved = Get-Content $pidFile -Raw | ConvertFrom-Json
foreach ($processId in @($saved.llm, $saved.backend, $saved.frontend)) {
  if ($processId) { Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue }
}
Remove-Item -LiteralPath $pidFile -Force
Write-Host 'PrivaSee stopped.'

