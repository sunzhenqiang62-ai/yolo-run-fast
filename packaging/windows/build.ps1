# Build the C++ aerial OBB inference binary on Windows.
param(
    [string]$Config = "Release",
    [switch]$SkipDeps
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path))
$CppDir = Join-Path $RepoRoot "cpp"
$BuildDir = Join-Path $CppDir "build"

$env:Path = "C:\Program Files\CMake\bin;" + $env:Path

if (-not $SkipDeps) {
    & (Join-Path $CppDir "setup_deps.ps1")
}

$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$VsInstall = & $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
$VcVars = Join-Path $VsInstall "VC/Auxiliary/Build/vcvars64.bat"

Write-Host "Using VS: $VsInstall"
& cmake -S $CppDir -B $BuildDir -DCMAKE_BUILD_TYPE=$Config
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }

$buildCmd = "call `"$VcVars`" >nul && cmake --build `"$BuildDir`" --config $Config --parallel"
cmd /c $buildCmd
if ($LASTEXITCODE -ne 0) { throw "cmake build failed" }

$Exe = Join-Path $BuildDir "$Config/aerial_obb.exe"
if (-not (Test-Path $Exe)) { $Exe = Join-Path $BuildDir "aerial_obb.exe" }
if (-not (Test-Path $Exe)) { throw "built executable not found" }
Write-Host "Built: $Exe"
Write-Output $Exe
