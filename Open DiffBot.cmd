@echo off
"%~dp0.venv\Scripts\python.exe" -m mujoco.viewer --mjcf="%~dp0diffbot.xml"
if errorlevel 1 pause
