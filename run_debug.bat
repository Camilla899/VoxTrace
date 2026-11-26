@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" "%~dp0app.py" %*
  goto :end
)
py -3.12 -c "import PySide6" >nul 2>nul
if not errorlevel 1 (
  py -3.12 "%~dp0app.py" %*
  goto :end
)
"C:\Users\Administrator\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe" "%~dp0app.py" %*
:end
pause
endlocal
