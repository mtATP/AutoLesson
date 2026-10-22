@echo off
cd /d "%~dp0"
set PYTHONUTF8=1
if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m autolesson gui
) else (
    start "" pythonw.exe -m autolesson gui
)
