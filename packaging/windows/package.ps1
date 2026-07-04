# Package Windows portable distribution + zip archive.
param(
    [string]$Config = "Release",
    [string]$Version = "1.0.0"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$DistName = "zhuangji-aerial-win64"
$DistRoot = Join-Path $RepoRoot "dist/$DistName"
$BuildDir = Join-Path $RepoRoot "cpp/build"
$ExeSrc = Join-Path $BuildDir "$Config/aerial_obb.exe"
if (-not (Test-Path $ExeSrc)) { $ExeSrc = Join-Path $BuildDir "aerial_obb.exe" }

if (-not (Test-Path $ExeSrc)) {
    Write-Host "Binary missing — running build first ..."
    $ExeSrc = & (Join-Path $RepoRoot "packaging/windows/build.ps1") -Config $Config
}

$Onnx = Join-Path $RepoRoot "runs/zhuangji_obb-2/weights/best.onnx"
if (-not (Test-Path $Onnx)) {
    Write-Host "Exporting ONNX model ..."
    & (Join-Path $RepoRoot ".venv/Scripts/python.exe") (Join-Path $RepoRoot "scripts/export_onnx.py")
}

if (Test-Path $DistRoot) { Remove-Item -Recurse -Force $DistRoot }
New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DistRoot "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $DistRoot "bin") | Out-Null

# Main executable + runtime DLLs (already copied beside exe by CMake post-build).
$BinSrc = Split-Path -Parent $ExeSrc
Copy-Item "$BinSrc/aerial_obb.exe" (Join-Path $DistRoot "bin/aerial_obb.exe")
Get-ChildItem $BinSrc -Filter "*.dll" | Copy-Item -Destination (Join-Path $DistRoot "bin")

Copy-Item $Onnx (Join-Path $DistRoot "models/best.onnx")

# Python aerial pipeline (requires user venv or system Python with deps).
Copy-Item (Join-Path $RepoRoot "predict_aerial.py") $DistRoot
Copy-Item (Join-Path $RepoRoot "zhuangji.yaml") $DistRoot -ErrorAction SilentlyContinue

# GUI (tkinter + subprocess; needs Python 3 on the host).
New-Item -ItemType Directory -Force -Path (Join-Path $DistRoot "ui") | Out-Null
Copy-Item (Join-Path $RepoRoot "ui\*.py") (Join-Path $DistRoot "ui")
Copy-Item (Join-Path $RepoRoot "aerial_obb_gui.bat") $DistRoot
New-Item -ItemType Directory -Force -Path (Join-Path $DistRoot "out") | Out-Null

@'
@echo off
setlocal
set ROOT=%~dp0
set BIN=%ROOT%bin
set PATH=%BIN%;%PATH%
"%BIN%\aerial_obb.exe" --model "%ROOT%models\best.onnx" %*
'@ | Set-Content -Encoding ASCII (Join-Path $DistRoot "aerial_obb.bat")

@'
@echo off
chcp 65001 >nul 2>&1
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"
title 装机检测航拍推理 CLI

echo.
echo ============================================================
echo   装机检测航拍推理 (zhuangji-aerial C++ CLI)
echo   安装目录: %ROOT%
echo ============================================================
echo.
echo 【图形界面】双击 aerial_obb_gui.bat 或桌面快捷方式
echo.
echo 【快速示例】
echo   aerial_obb.bat -i 你的影像.tif --json out\result.json --profile
echo.

call "%ROOT%aerial_obb.bat" --help

set "DEMO="
if exist "%ROOT%big_test.tif" set "DEMO=%ROOT%big_test.tif"
if not defined DEMO if exist "%ROOT%..\big_test.tif" set "DEMO=%ROOT%..\big_test.tif"
if not defined DEMO if exist "%ROOT%..\..\big_test.tif" set "DEMO=%ROOT%..\..\big_test.tif"

echo.
if defined DEMO (
  echo 【演示影像】检测到: %DEMO%
  echo   aerial_obb.bat -i "%DEMO%" --json out\demo.json --profile
) else (
  echo 【提示】将 .tif 航拍影像路径替换到上方 -i 参数即可开始推理。
)

echo.
echo ============================================================
echo 窗口保持打开 — 可直接输入命令；输入 exit 关闭。
echo ============================================================
echo.
cmd /k
'@ | Set-Content -Encoding ASCII (Join-Path $DistRoot "aerial_obb_launcher.bat")

@'
@echo off
setlocal
set ROOT=%~dp0
if exist "%ROOT%..\.venv\Scripts\python.exe" (
  "%ROOT%..\.venv\Scripts\python.exe" "%ROOT%predict_aerial.py" -m "%ROOT%models\..\runs\zhuangji_obb-2\weights\best.pt" %*
) else if exist "%ROOT%..\..\runs\zhuangji_obb-2\weights\best.pt" (
  python "%ROOT%predict_aerial.py" -m "%ROOT%..\..\runs\zhuangji_obb-2\weights\best.pt" %*
) else (
  echo Install Python deps: pip install -r requirements.txt
  python "%ROOT%predict_aerial.py" -m "%ROOT%models\best.onnx" %*
)
'@ | Set-Content -Encoding ASCII (Join-Path $DistRoot "predict_aerial.bat")

@'
# 装机检测 (zhuangji OBB) — Windows 便携包

## 内容
- bin/aerial_obb.exe — C++ ONNX Runtime 航拍分块推理（CPU）
- models/best.onnx — 推荐 ONNX 模型
- ui/aerial_gui.py + aerial_obb_gui.bat — 图形界面（需本机 Python 3.10+）
- predict_aerial.py — Python 航拍推理（需单独安装 Python 依赖）
- aerial_obb.bat — C++ CLI 启动器
- aerial_obb_launcher.bat — CLI 命令行窗口（显示帮助）
- predict_aerial.bat — Python CLI 启动器

## 快速开始（图形界面，推荐）

双击桌面/开始菜单快捷方式，或：

```powershell
cd dist\zhuangji-aerial-win64
.\aerial_obb_gui.bat
```

界面内选择影像、输出 JSON，点击「开始推理」。C++ ONNX 后端无需 pip 依赖。

## 命令行（C++）

在命令行：

```powershell
cd dist\zhuangji-aerial-win64
.\aerial_obb.bat -i C:\path\to\large.tif --json out.json --profile
```

## Python 推理（需本机 Python 环境）

```powershell
pip install -r requirements.txt   # 在仓库根目录
.\predict_aerial.bat -i C:\path\to\large.tif --json out.json --backend openvino
```

## 重新构建

```powershell
.\packaging\windows\build.ps1
.\packaging\windows\package.ps1
```

版本: {0}
'@ -f $Version | Set-Content -Encoding UTF8 (Join-Path $DistRoot "README.txt")

$Zip = Join-Path $RepoRoot "dist/$DistName-v$Version.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
Compress-Archive -Path $DistRoot -DestinationPath $Zip
Write-Host "Portable package: $DistRoot"
Write-Host "Zip archive: $Zip"
