@echo off
setlocal

set IMAGE_NAME=fiap-challenge-fase1-9iadt-rm370509
set IMAGE_VERSION=1.0

for /f "tokens=*" %%i in ('powershell -NoProfile -Command "Get-Date -Format \"yyyy-MM-ddTHH:mm:ssZ\" -AsUTC"') do set CREATED_DATETIME=%%i

echo Building Docker image: %IMAGE_NAME%:%IMAGE_VERSION%

docker build ^
  --build-arg CREATED_DATETIME="%CREATED_DATETIME%" ^
  -t "%IMAGE_NAME%:%IMAGE_VERSION%" ^
  -t "%IMAGE_NAME%:latest" ^
  .

if %ERRORLEVEL% neq 0 (
    echo Build failed.
    exit /b %ERRORLEVEL%
)

echo Build completed successfully: %IMAGE_NAME%:%IMAGE_VERSION%
endlocal

