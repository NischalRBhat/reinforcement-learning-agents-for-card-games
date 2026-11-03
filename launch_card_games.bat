@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PYTHON=.venv\Scripts\python.exe"
) else (
  set "PYTHON=python"
)
echo Launching Card Games RL with %PYTHON%...
"%PYTHON%" launch_card_games.py
if errorlevel 1 pause
