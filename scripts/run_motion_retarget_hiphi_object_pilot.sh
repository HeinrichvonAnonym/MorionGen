#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MOTION_RETARGET_PYTHON:-/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/envs/motiogen-smplsim/bin/python}"
CONFIG="${MOTION_RETARGET_OBJECT_CONFIG:-$ROOT/config/motion_retarget_hiphi_object.yaml}"
SELECTION="${MOTION_RETARGET_OBJECT_SELECTION:-$ROOT/config/hiphi_object_pilot_selection.json}"
OUTPUT_ROOT="${MOTION_RETARGET_OUTPUT_ROOT:-/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/motion_retarget_hiphi_object_pilot}"
LOG_ROOT="${MOTION_RETARGET_OBJECT_LOG_ROOT:-$ROOT/develop_logs/test_runs/motion_retarget_hiphi_object_pilot}"
CLI=("$PYTHON" "$ROOT/python/scripts/motion_retarget.py" --config "$CONFIG")
COMMON=(--dataset hiphi --profile source_native --with-objects --selection-manifest "$SELECTION")

export MOTION_RETARGET_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

case "${1:-status}" in
  inventory) "${CLI[@]}" inventory "${COMMON[@]}" | tee "$LOG_ROOT/inventory.log" ;;
  extract) "${CLI[@]}" extract "${COMMON[@]}" --workers 1 --gpus 0 --resume --retries 1 | tee "$LOG_ROOT/extract.log" ;;
  validate) "${CLI[@]}" validate "${COMMON[@]}" --workers 1 --mujoco | tee "$LOG_ROOT/validate.log" ;;
  render) "${CLI[@]}" render "${COMMON[@]}" --workers 1 --mujoco | tee "$LOG_ROOT/render.log" ;;
  summarize) "${CLI[@]}" summarize --with-objects --selection-manifest "$SELECTION" | tee "$LOG_ROOT/summary.json" ;;
  run)
    "$0" inventory
    "$0" extract
    "$0" validate || true
    "$0" render || true
    "$0" summarize
    ;;
  status)
    systemctl --user show motion-retarget-andy-hiphi.service -p ActiveState -p SubState -p MainPID 2>/dev/null || true
    if [[ -f "$LOG_ROOT/extract.log" ]]; then tail -n 20 "$LOG_ROOT/extract.log"; fi
    ;;
  *) echo "usage: $0 {inventory|extract|validate|render|summarize|run|status}" >&2; exit 2 ;;
esac
