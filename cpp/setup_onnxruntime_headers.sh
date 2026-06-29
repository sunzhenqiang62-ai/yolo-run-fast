#!/usr/bin/env bash
# Fetch ONNX Runtime C++ API headers (v1.19.2) into third_party/.
# The venv ships only the .so files, not the headers, so we pull the matching
# headers from the microsoft/onnxruntime tag via jsdelivr.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DST="$HERE/third_party/onnxruntime/include"
TAG="v1.19.2"
BASE="https://fastly.jsdelivr.net/gh/microsoft/onnxruntime@${TAG}"

mkdir -p "$DST"

# core/session: the public C and C++ API
declare -A FILES=(
  ["$DST/onnxruntime_c_api.h"]="$BASE/include/onnxruntime/core/session/onnxruntime_c_api.h"
  ["$DST/onnxruntime_cxx_api.h"]="$BASE/include/onnxruntime/core/session/onnxruntime_cxx_api.h"
  ["$DST/onnxruntime_cxx_inline.h"]="$BASE/include/onnxruntime/core/session/onnxruntime_cxx_inline.h"
  ["$DST/onnxruntime_float16.h"]="$BASE/include/onnxruntime/core/session/onnxruntime_float16.h"
  ["$DST/onnxruntime_run_options_config_keys.h"]="$BASE/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h"
  ["$DST/onnxruntime_session_options_config_keys.h"]="$BASE/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h"
  # provider factory headers (note: cpu lives under include/, cuda under onnxruntime/ in the repo tree)
  ["$DST/cuda_provider_factory.h"]="$BASE/onnxruntime/core/providers/cuda/cuda_provider_factory.h"
  ["$DST/cpu_provider_factory.h"]="$BASE/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h"
)

for dst in "${!FILES[@]}"; do
  url="${FILES[$dst]}"
  echo "fetching $(basename "$dst")"
  curl -fsSL "$url" -o "$dst"
done

echo "ONNX Runtime headers installed into $DST"
ls -1 "$DST"
