@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ============================================
echo   AutoLesson 一键部署
echo   将检查 Python、创建 .venv、安装依赖
echo ============================================
echo.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_windows.ps1"
set "CODE=%ERRORLEVEL%"
echo.
if not "%CODE%"=="0" (
    echo [失败] 部署未完成，错误码 %CODE%
    echo 请把上面的红色错误信息截图发给维护者。
) else (
    echo [完成] 环境已准备好，可以双击 启动GUI.bat。
)
echo.
pause
exit /b %CODE%
