@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=%WAN_PYTHON_EXE%"
if not defined PYTHON_EXE (
  if defined WAN_COMFY_ROOT (
    set "PYTHON_EXE=%WAN_COMFY_ROOT%\.venv\Scripts\python.exe"
  ) else (
    set "PYTHON_EXE=D:\AI\ComfyUI\.venv\Scripts\python.exe"
  )
)
if not exist "%PYTHON_EXE%" (
  echo Python was not found at "%PYTHON_EXE%".
  echo Set WAN_PYTHON_EXE or WAN_COMFY_ROOT before launching.
  pause
  exit /b 1
)
"%PYTHON_EXE%" app.py
pause
