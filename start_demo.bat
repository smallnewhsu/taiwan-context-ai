@echo off
setlocal
cd /d "%~dp0"

set "OLLAMA_CONTEXT_LENGTH=8192"
set "TAIWAN_CONTEXT_LLM_MODEL=qwen2.5:1.5b"
set "SPEECH_CONTEXT_MODEL=qwen2.5:1.5b"
set "TAIGI_MODEL=qwen2.5:1.5b"
set "MULTIMODAL_QA_MODEL=qwen2.5:1.5b"
set "VIETNAMESE_ASR_MODEL=turbo"
set "PROJECT_PYTHON=%CD%\services\asr\asr_api\Scripts\python.exe"

if not exist "%PROJECT_PYTHON%" (
    echo [ERROR] Project Python not found:
    echo %PROJECT_PYTHON%
    pause
    exit /b 1
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo [ERROR] ffmpeg is not available in PATH.
    pause
    exit /b 1
)

where ollama >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama is not available in PATH.
    pause
    exit /b 1
)

curl -s --max-time 2 http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [INFO] Starting Ollama...
    start "Ollama Server" /min cmd /c "set OLLAMA_CONTEXT_LENGTH=8192 && ollama serve"
    timeout /t 5 /nobreak >nul
) else (
    echo [OK] Ollama is running.
)

curl -s --max-time 2 http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Ollama did not become ready.
    echo Run "ollama serve" in another terminal and try again.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Taiwan Context AI Demo
echo Home:    http://127.0.0.1:8000/app/
echo Rewrite: http://127.0.0.1:8000/app/rewrite/
echo Vision:  http://127.0.0.1:8000/app/vision/
echo API:     http://127.0.0.1:8000/docs
echo ========================================
echo.

start "Taiwan Context API" "%PROJECT_PYTHON%" -m uvicorn backend.api:app --host 127.0.0.1 --port 8000

echo [INFO] Waiting for API...
for /l %%I in (1,1,30) do (
    curl -s --max-time 1 http://127.0.0.1:8000/health >nul 2>&1 && goto api_ready
    timeout /t 1 /nobreak >nul
)

echo [ERROR] API did not become ready within 30 seconds.
echo Check the Taiwan Context API window for details.
pause
exit /b 1

:api_ready
echo [OK] API is ready.
start "" http://127.0.0.1:8000/app/

echo [OK] Demo services started. Close the API window to stop the backend.
timeout /t 3 /nobreak >nul
endlocal
