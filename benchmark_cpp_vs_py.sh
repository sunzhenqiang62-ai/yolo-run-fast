#!/usr/bin/env bash
# Fair C++ (ONNX Runtime) vs Python (ultralytics) speed comparison on one big
# aerial image. Same image / model / conf / strategy for every run.
#
#   GPU:  Python --turbo (pt)        vs  C++ --device 0   (ORT-CUDA)
#   CPU:  Python --backend openvino  vs  C++ --device cpu (ORT-CPU)
#
# Prints detection counts + wall time per run.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="${PY:-.venv/bin/python}"
CPP="${CPP:-cpp/build/aerial_obb}"
ONNX="${ONNX:-runs/zhuangji_obb-2/weights/best.onnx}"
PT="${PT:-runs/zhuangji_obb-2/weights/best.pt}"
IMG="${IMG:-big_test.tif}"
CONF="${CONF:-0.25}"
COARSE_CONF="${COARSE_CONF:-0.35}"
STRATEGY="${STRATEGY:-two-stage}"
OUTDIR="${OUTDIR:-runs/bench}"
mkdir -p "$OUTDIR"

if [[ ! -f "$IMG" ]]; then
  echo "Big test image $IMG missing; generating..."
  "$PY" make_big_test_image.py --out "$IMG"
fi

# Returns wall seconds of the command (stdout of the cmd is teed to a log).
timeit() {
  local log="$1"; shift
  local t0 t1
  t0=$(date +%s.%N)
  "$@" >"$log" 2>&1
  local rc=$?
  t1=$(date +%s.%N)
  echo "$(awk "BEGIN{printf \"%.2f\", $t1-$t0}") $rc"
}

count_json() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['count'])" "$1" 2>/dev/null || echo "?"; }

echo "=================================================================="
echo " Benchmark image: $IMG"
echo " strategy=$STRATEGY conf=$CONF coarse_conf=$COARSE_CONF"
echo "=================================================================="

declare -A WALL DET

run_case() {
  local name="$1"; shift
  local json="$1"; shift
  echo; echo ">>> $name"
  read -r secs rc <<<"$(timeit "$OUTDIR/$name.log" "$@")"
  WALL[$name]=$secs
  if [[ "$rc" != "0" ]]; then
    echo "    FAILED (rc=$rc) — see $OUTDIR/$name.log"; tail -5 "$OUTDIR/$name.log"
    DET[$name]="ERR"
  else
    DET[$name]=$(count_json "$json")
    echo "    wall=${secs}s  detections=${DET[$name]}"
  fi
}

# ---- GPU ----
run_case "py_gpu_turbo" "$OUTDIR/py_gpu.json" \
  "$PY" predict_aerial.py -i "$IMG" -m "$PT" --json "$OUTDIR/py_gpu.json" \
  --turbo --strategy "$STRATEGY" --conf "$CONF" --coarse-conf "$COARSE_CONF" --skip-preview --profile

run_case "cpp_gpu" "$OUTDIR/cpp_gpu.json" \
  "$CPP" -i "$IMG" -m "$ONNX" --json "$OUTDIR/cpp_gpu.json" \
  --device 0 --strategy "$STRATEGY" --conf "$CONF" --coarse-conf "$COARSE_CONF" --skip-preview --profile

# ---- CPU ----
run_case "py_cpu_openvino" "$OUTDIR/py_cpu.json" \
  "$PY" predict_aerial.py -i "$IMG" -m "$PT" --json "$OUTDIR/py_cpu.json" \
  --backend openvino --device cpu --strategy "$STRATEGY" --conf "$CONF" \
  --coarse-conf "$COARSE_CONF" --skip-preview --profile

run_case "cpp_cpu" "$OUTDIR/cpp_cpu.json" \
  "$CPP" -i "$IMG" -m "$ONNX" --json "$OUTDIR/cpp_cpu.json" \
  --device cpu --strategy "$STRATEGY" --conf "$CONF" --coarse-conf "$COARSE_CONF" --skip-preview --profile

echo
echo "=================================================================="
printf "%-18s %10s %12s\n" "case" "wall(s)" "detections"
echo "------------------------------------------------------------------"
for k in py_gpu_turbo cpp_gpu py_cpu_openvino cpp_cpu; do
  printf "%-18s %10s %12s\n" "$k" "${WALL[$k]:-NA}" "${DET[$k]:-NA}"
done
echo "------------------------------------------------------------------"
spd() { awk "BEGIN{if($2>0)printf \"%.2fx\", $1/$2; else print \"NA\"}"; }
[[ -n "${WALL[py_gpu_turbo]:-}" && -n "${WALL[cpp_gpu]:-}" ]] && \
  echo " GPU speedup (py/cpp): $(spd ${WALL[py_gpu_turbo]} ${WALL[cpp_gpu]})"
[[ -n "${WALL[py_cpu_openvino]:-}" && -n "${WALL[cpp_cpu]:-}" ]] && \
  echo " CPU speedup (py/cpp): $(spd ${WALL[py_cpu_openvino]} ${WALL[cpp_cpu]})"
echo "=================================================================="
