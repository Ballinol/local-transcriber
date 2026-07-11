@echo off
setlocal
cd /d "%~dp0"

echo === Local Transcriber: setup (core) ===
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python not found on PATH.
  echo Install Python 3.10+ from https://python.org and tick "Add Python to PATH".
  pause
  exit /b 1
)

python -c "import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)"
if errorlevel 1 ( echo [ERROR] Need Python 3.10+ & python --version & pause & exit /b 1 )

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 ( echo [ERROR] venv failed & pause & exit /b 1 )
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip --quiet
echo Installing core dependencies (faster-whisper, PyAV)...
python -m pip install -r requirements.txt
if errorlevel 1 ( echo [ERROR] install failed & pause & exit /b 1 )

echo.
echo === Done ===
echo Now drag a video/audio file onto transcribe.bat, or run:
echo   .venv\Scripts\python.exe transcribe_file.py "video.mp4"
echo.
echo For speaker diarization (who spoke), run setup_diarize.bat too.
echo First transcription downloads the Whisper model (~1.6 GB), one time.
echo.
pause
