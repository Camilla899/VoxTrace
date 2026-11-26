@echo off
setlocal
cd /d "%~dp0"
echo ============================================
echo  VoxTrace Studio - environment setup
echo  Python 3.11-3.13 (64-bit) is recommended.
echo ============================================
set "MADE="
where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import sys" >nul 2>nul
  if not errorlevel 1 (
    py -3.12 -m venv .venv
    set "MADE=1"
  )
)
if not defined MADE (
  python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo Could not create .venv. Please install Python 3.12 first.
  pause
  exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Dependency installation failed.
  pause
  exit /b 1
)
echo.
echo Done. Double-click run.bat to launch VoxTrace.
pause
