@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo Drag video/audio files onto this file to transcribe them to text.
  echo Formats: mp4, mp3, mkv, wav, m4a, webm ...
  echo Output: a .txt file appears next to each source file.
  pause
  exit /b
)

if not exist ".venv\Scripts\python.exe" (
  echo [!] .venv not found. Run setup.bat first.
  pause
  exit /b 1
)

chcp 65001 >nul
".venv\Scripts\python.exe" transcribe_file.py %*

echo.
echo Done. The .txt files are next to the source files.
pause
