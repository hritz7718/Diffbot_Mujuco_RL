@echo off
"%~dp0.venv\Scripts\python.exe" -m pip install mujoco gymnasium stable-baselines3
if errorlevel 1 goto end
"%~dp0.venv\Scripts\python.exe" "%~dp0train_diffbot.py" --check-only
:end
pause
