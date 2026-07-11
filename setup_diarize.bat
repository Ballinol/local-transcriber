@echo off
setlocal
cd /d "%~dp0"

echo === Local Transcriber: setup (speaker diarization extras) ===
echo.

if not exist ".venv\Scripts\python.exe" ( echo [ERROR] Run setup.bat first. & pause & exit /b 1 )

call ".venv\Scripts\activate.bat"
echo Installing diarization deps (speechbrain, scikit-learn, torch CPU)...
python -m pip install -r requirements-diarize.txt
if errorlevel 1 ( echo [ERROR] install failed & pause & exit /b 1 )

echo.
echo === Done ===
echo Run: .venv\Scripts\python.exe diarize_transcribe.py "video.mp4" --speakers 3
echo.
echo NOTE: this installed CPU torch. For NVIDIA GPU (much faster) run:
echo   .venv\Scripts\python.exe -m pip uninstall torch torchaudio -y
echo   .venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
echo.
pause
