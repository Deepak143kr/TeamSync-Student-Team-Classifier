@echo off
echo.
echo  ============================================
echo   TeamSync — Student Compatibility Classifier
echo  ============================================
echo.

cd /d "%~dp0"

echo [1/3] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install from https://python.org
    pause & exit /b 1
)

echo [2/3] Installing dependencies...
cd backend
pip install -r requirements.txt --quiet

echo [3/3] Starting Flask server...
echo.
echo  Open your browser at: http://localhost:5000
echo  Press Ctrl+C to stop.
echo.
python app.py
pause
