# Download native deps for Windows C++ inference (ORT + prebuilt OpenCV).
param(
    [string]$OrtVersion = "1.27.0",
    [string]$OpenCvVersion = "4.10.0"
)

$ErrorActionPreference = "Stop"
$CppDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ThirdParty = Join-Path $CppDir "third_party"

New-Item -ItemType Directory -Force -Path $ThirdParty | Out-Null

# --- ONNX Runtime Windows prebuilt ---
$OrtDir = Join-Path $ThirdParty "onnxruntime-win-x64"
$OrtZip = Join-Path $ThirdParty "onnxruntime-win-x64-$OrtVersion.zip"
$OrtUrl = "https://github.com/microsoft/onnxruntime/releases/download/v$OrtVersion/onnxruntime-win-x64-$OrtVersion.zip"

if (-not (Test-Path (Join-Path $OrtDir "lib/onnxruntime.lib"))) {
    Write-Host "Downloading ONNX Runtime $OrtVersion ..."
    if (-not (Test-Path $OrtZip)) {
        Invoke-WebRequest -Uri $OrtUrl -OutFile $OrtZip -UseBasicParsing
    }
    $ExtractRoot = Join-Path $ThirdParty "ort_extract"
    if (Test-Path $ExtractRoot) { Remove-Item -Recurse -Force $ExtractRoot }
    Expand-Archive -Path $OrtZip -DestinationPath $ExtractRoot
    $Inner = Get-ChildItem $ExtractRoot -Directory | Select-Object -First 1
    if (Test-Path $OrtDir) { Remove-Item -Recurse -Force $OrtDir }
    Move-Item $Inner.FullName $OrtDir
    Remove-Item -Recurse -Force $ExtractRoot
}
Write-Host "ORT ready: $OrtDir"

& (Join-Path $CppDir "setup_onnxruntime_headers.ps1") -Tag "v$OrtVersion"

# --- Prebuilt OpenCV (official Windows pack, vc16/x64) ---
$OpenCvRoot = Join-Path $ThirdParty "opencv"
$OpenCvMarker = Join-Path $OpenCvRoot "build/x64/vc16/lib/opencv_world4100.lib"
if (-not (Test-Path $OpenCvMarker)) {
    $OpenCvExe = Join-Path $ThirdParty "opencv-$OpenCvVersion-windows.exe"
    $OpenCvUrl = "https://github.com/opencv/opencv/releases/download/$OpenCvVersion/opencv-$OpenCvVersion-windows.exe"
    Write-Host "Downloading OpenCV $OpenCvVersion Windows pack ..."
    if (-not (Test-Path $OpenCvExe)) {
        Invoke-WebRequest -Uri $OpenCvUrl -OutFile $OpenCvExe -UseBasicParsing
    }
    if (Test-Path $OpenCvRoot) { Remove-Item -Recurse -Force $OpenCvRoot }
    Write-Host "Extracting OpenCV (silent 7z self-extractor) ..."
    Start-Process -FilePath $OpenCvExe -ArgumentList "-o$ThirdParty", "-y" -Wait -NoNewWindow
    if (-not (Test-Path $OpenCvMarker)) {
        throw "OpenCV extract failed — expected $OpenCvMarker"
    }
}
Write-Host "OpenCV ready: $OpenCvRoot"
