; Inno Setup script — compile with ISCC.exe if installed:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\windows\installer.iss
#define MyAppName "装机检测航拍推理"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "yolo26n"
#define RepoRoot "..\\.."

#if DirExists(RepoRoot + "\dist\zhuangji-aerial-win64")
#else
  #error "Run packaging\windows\package.ps1 to create dist\zhuangji-aerial-win64 before compiling the installer."
#endif

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
DefaultDirName={autopf}\zhuangji-aerial
DefaultGroupName={#MyAppName}
OutputDir={#RepoRoot}\dist
OutputBaseFilename=zhuangji-aerial-setup-v{#MyAppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "{#RepoRoot}\dist\zhuangji-aerial-win64\*"; DestDir: "{app}"; Flags: recursesubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\aerial_obb_gui.bat"; WorkingDir: "{app}"
Name: "{group}\C++ 航拍推理 CLI"; Filename: "{app}\aerial_obb_launcher.bat"; WorkingDir: "{app}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\aerial_obb_gui.bat"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{group}\README"; Filename: "{app}\README.txt"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加图标:"

[Run]
Filename: "{app}\README.txt"; Description: "查看使用说明"; Flags: postinstall shellexec skipifsilent
