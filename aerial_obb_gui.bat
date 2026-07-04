@echo off
chcp 65001 >nul 2>&1
setlocal
set "ROOT=%~dp0"
set "AERIAL_OBB_ROOT=%ROOT%"
cd /d "%ROOT%"

if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" "%ROOT%ui\aerial_gui.py" %*
  goto :done
)
if exist "%ROOT%..\..\.venv\Scripts\python.exe" (
  set "AERIAL_OBB_ROOT=%ROOT%..\.."
  "%ROOT%..\..\.venv\Scripts\python.exe" "%ROOT%ui\aerial_gui.py" %*
  goto :done
)
where py >nul 2>&1 && (
  py -3 "%ROOT%ui\aerial_gui.py" %*
  goto :done
)
where python >nul 2>&1 && (
  python "%ROOT%ui\aerial_gui.py" %*
  goto :done
)

echo.
echo [错误] 未找到 Python 3，无法启动图形界面。
echo 请安装 Python 3.10+ 或在仓库根目录创建 .venv 后重试。
echo 将打开命令行 CLI 启动器…
echo.
if exist "%ROOT%aerial_obb_launcher.bat" (
  start "" "%ROOT%aerial_obb_launcher.bat"
) else (
  pause
)
:done
