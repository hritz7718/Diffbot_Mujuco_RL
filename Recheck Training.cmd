@echo off
"%~dp0.venv\Scripts\python.exe" "%~dp0train_diffbot.py" --check-only > "%~dp0baseline_diagnostics.txt" 2>&1
type "%~dp0baseline_diagnostics.txt"
pause
