@echo off
setlocal
cd /d "%~dp0"
set "PYTHONPATH=%~dp0"
set "DSH_PY=C:\Users\Administrator\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python"

REM Launch with pythonw so the desktop app does not keep a console window open.
REM Use run_debug.bat if you need console diagnostics.
if exist "%~dp0.venv\Scripts\pythonw.exe" (
  start "" "%~dp0.venv\Scripts\pythonw.exe" "%~dp0app.py" %*
  goto :eof
)

where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "import PySide6, onnxruntime, librosa" >nul 2>nul
  if not errorlevel 1 (
    start "" pyw -3.12 "%~dp0app.py" %*
    goto :eof
  )
)

if exist "%DSH_PY%\pythonw.exe" (
  start "" "%DSH_PY%\pythonw.exe" "%~dp0app.py" %*
  goto :eof
)

echo VoxTrace could not find a Python environment with its dependencies.
echo Run setup_env.bat, then try again.
pause
endlocal
