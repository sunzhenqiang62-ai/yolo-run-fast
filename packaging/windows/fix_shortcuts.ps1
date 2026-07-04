# Update desktop/start-menu shortcuts to use aerial_obb_gui.bat (GUI default).
param(
    [Parameter(Mandatory = $true)]
    [string]$InstallDir
)

$ErrorActionPreference = "Stop"
$InstallDir = (Resolve-Path $InstallDir).Path
$GuiLauncher = Join-Path $InstallDir "aerial_obb_gui.bat"
$CliLauncher = Join-Path $InstallDir "aerial_obb_launcher.bat"
if (-not (Test-Path $GuiLauncher)) {
    throw "GUI launcher not found: $GuiLauncher (run package.ps1 first)"
}

function Set-AerialShortcut(
    [string]$LnkPath,
    [string]$Target,
    [string]$Description
) {
    $dir = Split-Path $LnkPath -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $sh = New-Object -ComObject WScript.Shell
    $lnk = $sh.CreateShortcut($LnkPath)
    $lnk.TargetPath = $Target
    $lnk.WorkingDirectory = $InstallDir
    $lnk.Description = $Description
    $lnk.WindowStyle = 1
    $lnk.Save()
    Write-Host "Updated: $LnkPath"
}

function Set-GuiShortcut([string]$LnkPath) {
    Set-AerialShortcut $LnkPath $GuiLauncher "zhuangji-aerial GUI"
}

function Set-CliShortcut([string]$LnkPath) {
    if (-not (Test-Path $CliLauncher)) { return }
    Set-AerialShortcut $LnkPath $CliLauncher "zhuangji-aerial CLI"
}

Get-ChildItem "$env:USERPROFILE\Desktop\*.lnk" -ErrorAction SilentlyContinue | ForEach-Object {
    $sh = New-Object -ComObject WScript.Shell
    $lnk = $sh.CreateShortcut($_.FullName)
    if ($lnk.TargetPath -match "aerial_obb(_launcher|_gui)?\.bat$" -and $lnk.WorkingDirectory -eq $InstallDir) {
        Set-GuiShortcut $_.FullName
    }
}

$names = @("zhuangji-aerial.lnk")
foreach ($n in $names) {
    $p = Join-Path $env:USERPROFILE "Desktop\$n"
    if (Test-Path $p) { Set-GuiShortcut $p }
}

$startDir = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\zhuangji-aerial"
Set-GuiShortcut (Join-Path $startDir "zhuangji-aerial GUI.lnk")
Set-CliShortcut (Join-Path $startDir "zhuangji-aerial CLI.lnk")

Get-ChildItem "$env:APPDATA\Microsoft\Windows\Start Menu" -Recurse -Filter "*.lnk" -ErrorAction SilentlyContinue | ForEach-Object {
    try {
        $sh = New-Object -ComObject WScript.Shell
        $lnk = $sh.CreateShortcut($_.FullName)
        if ($lnk.TargetPath -match "aerial_obb\.bat$" -and $lnk.WorkingDirectory -eq $InstallDir) {
            Set-GuiShortcut $_.FullName
        }
    } catch {}
}

Write-Host "Done. GUI launcher: $GuiLauncher"
