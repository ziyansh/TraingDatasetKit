@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Starting TraingDatasetKit...
echo.
call venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo Failed to activate virtual environment.
    pause
    exit /b 1
)
python app.py
pause