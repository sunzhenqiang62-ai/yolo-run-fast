# Fetch ONNX Runtime C++ API headers into third_party/.
# The pip wheel ships DLLs only; headers come from the matching GitHub tag.
param(
    [string]$Tag = "v1.27.0"
)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dst = Join-Path $Here "third_party/onnxruntime/include"
$Base = "https://raw.githubusercontent.com/microsoft/onnxruntime/$Tag"

New-Item -ItemType Directory -Force -Path $Dst | Out-Null

$Files = @{
    "onnxruntime_c_api.h" = "$Base/include/onnxruntime/core/session/onnxruntime_c_api.h"
    "onnxruntime_cxx_api.h" = "$Base/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
    "onnxruntime_cxx_inline.h" = "$Base/include/onnxruntime/core/session/onnxruntime_cxx_inline.h"
    "onnxruntime_float16.h" = "$Base/include/onnxruntime/core/session/onnxruntime_float16.h"
    "onnxruntime_run_options_config_keys.h" = "$Base/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h"
    "onnxruntime_session_options_config_keys.h" = "$Base/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h"
    "cuda_provider_factory.h" = "$Base/onnxruntime/core/providers/cuda/cuda_provider_factory.h"
    "cpu_provider_factory.h" = "$Base/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h"
}

foreach ($name in $Files.Keys) {
    $url = $Files[$name]
    $out = Join-Path $Dst $name
    Write-Host "fetching $name"
    Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing
}

Write-Host "ONNX Runtime headers installed into $Dst"
Get-ChildItem $Dst | Select-Object -ExpandProperty Name
