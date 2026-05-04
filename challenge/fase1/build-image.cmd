@echo off
setlocal

set IMAGE_NAME=fiap-challenge-fase1-9iadt-rm370509
set IMAGE_VERSION=1.0

for /f "tokens=*" %%i in ('powershell -NoProfile -Command "Get-Date -Format \"yyyy-MM-ddTHH:mm:ssZ\" -AsUTC"') do set CREATED_DATETIME=%%i

where docker >nul 2>&1
if %ERRORLEVEL% equ 0 (
    set CONTAINER_CLI=docker
) else (
    where podman >nul 2>&1
    if %ERRORLEVEL% equ 0 (
        set CONTAINER_CLI=podman
    ) else (
        echo Error: neither docker nor podman was found. Please install one of them.
        exit /b 1
    )
)

echo Using: %CONTAINER_CLI%
echo Building Docker image: %IMAGE_NAME%:%IMAGE_VERSION%

%CONTAINER_CLI% build ^
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

