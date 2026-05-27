@echo off
setlocal

echo [1/3] Converting migration files to CRLF...
powershell -NoProfile -Command ^
  "Get-ChildItem -Recurse 'state\migrations\*.sql','state\logs_migrations\*.sql','state\goals_migrations\*.sql' | ForEach-Object { " ^
  "  $c = [IO.File]::ReadAllText($_.FullName); " ^
  "  if ($c -match \"`n\" -and $c -notmatch \"`r`n\") { " ^
  "    [IO.File]::WriteAllText($_.FullName, $c.Replace(\"`n\", \"`r`n\")); " ^
  "    Write-Host ('  converted: ' + $_.Name) " ^
  "  } else { " ^
  "    Write-Host ('  already CRLF: ' + $_.Name) " ^
  "  } " ^
  "}"

echo [2/3] Building release...
cargo build --release -p codex-cli
if errorlevel 1 (
  echo BUILD FAILED
  exit /b 1
)

echo [3/3] Done.
echo Binary: target\release\codex.exe
