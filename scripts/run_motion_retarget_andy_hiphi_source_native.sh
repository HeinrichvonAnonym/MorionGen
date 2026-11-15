#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MOTION_RETARGET_PYTHON:-/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/envs/motiogen-smplsim/bin/python}"
CONFIG="${MOTION_RETARGET_CONFIG:-$ROOT/config/motion_retarget.yaml}"
OUTPUT_ROOT="/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/motion_retarget_andy_hiphi_source_native"
RUN_LOG_ROOT="$ROOT/develop_logs/test_runs/motion_retarget_andy_hiphi_source_native"
CLI=("$PYTHON" "$ROOT/python/scripts/motion_retarget.py" --config "$CONFIG")
COMMON=(--dataset andy --dataset hiphi --profile source_native)

export MOTION_RETARGET_OUTPUT_ROOT="$OUTPUT_ROOT"
export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
mkdir -p "$OUTPUT_ROOT" "$RUN_LOG_ROOT"

if [[ ! -x "$PYTHON" ]]; then
  echo "motion-retarget environment is missing: $PYTHON" >&2
  exit 2
fi

preflight() {
  "${CLI[@]}" preflight "${COMMON[@]}" | tee "$RUN_LOG_ROOT/preflight.log"
}

pilot() {
  "${CLI[@]}" prewarm --dataset andy --profile source_native --workers 1 --gpus 0 \
    --sequence Participant_2193_Setup_B_Seq_6_Trial_4 \
    --sequence Participant_5319_Setup_B_Seq_6_Trial_2 | tee "$RUN_LOG_ROOT/pilot_prewarm.log"
  "${CLI[@]}" extract "${COMMON[@]}" --workers 2 --gpus 0,1 --resume --retries 2 \
    --sequence Participant_2193_Setup_B_Seq_6_Trial_4 \
    --sequence Participant_5319_Setup_B_Seq_6_Trial_2 \
    --sequence Adjusting-adjust_0001 \
    --sequence Adjusting-adjust_0002 | tee "$RUN_LOG_ROOT/pilot_extract.log"
  "${CLI[@]}" validate "${COMMON[@]}" --workers 2 --mujoco \
    --sequence Participant_2193_Setup_B_Seq_6_Trial_4 \
    --sequence Participant_5319_Setup_B_Seq_6_Trial_2 \
    --sequence Adjusting-adjust_0001 \
    --sequence Adjusting-adjust_0002 | tee "$RUN_LOG_ROOT/pilot_validate.log"
}

extract_hiphi() {
  "${CLI[@]}" extract --dataset hiphi --profile source_native --workers 2 --gpus 0,1 \
    --resume --retries 2 | tee "$RUN_LOG_ROOT/extract_hiphi.log" || true
}

extract_andy() {
  "${CLI[@]}" prewarm --dataset andy --profile source_native --workers 1 --gpus 0 \
    | tee "$RUN_LOG_ROOT/prewarm_andy.log"
  "${CLI[@]}" extract --dataset andy --profile source_native --workers 2 --gpus 0,1 \
    --resume --retries 2 | tee "$RUN_LOG_ROOT/extract_andy.log" || true
}

extract_all() {
  extract_hiphi
  extract_andy
}

validate_hiphi() {
  "${CLI[@]}" validate --dataset hiphi --profile source_native --workers 2 --mujoco \
    | tee "$RUN_LOG_ROOT/validate_hiphi.log" || true
}

validate_andy() {
  "${CLI[@]}" validate --dataset andy --profile source_native --workers 2 --mujoco \
    | tee "$RUN_LOG_ROOT/validate_andy.log" || true
}

validate_all() {
  validate_hiphi
  validate_andy
}

retry_dataset() {
  local dataset="$1"
  "${CLI[@]}" extract --dataset "$dataset" --profile source_native --workers 2 --gpus 0,1 \
    --resume --retries 2 --only-status failed \
    | tee "$RUN_LOG_ROOT/retry_extract_${dataset}.log" || true
  "${CLI[@]}" validate --dataset "$dataset" --profile source_native --workers 2 --mujoco \
    --only-status extracted | tee "$RUN_LOG_ROOT/retry_validate_extracted_${dataset}.log" || true
  "${CLI[@]}" validate --dataset "$dataset" --profile source_native --workers 2 --mujoco \
    --only-status validation_failed | tee "$RUN_LOG_ROOT/retry_validate_failed_${dataset}.log" || true
}

retry_failed() {
  retry_dataset hiphi
  retry_dataset andy
}

render_previews() {
  "${CLI[@]}" render-previews "${COMMON[@]}" --workers 2 --mujoco \
    | tee "$RUN_LOG_ROOT/render_previews.log" || true
}

summarize() {
  "${CLI[@]}" summarize | tee "$RUN_LOG_ROOT/summary.json"
}

run_all() {
  echo "$$" > "$OUTPUT_ROOT/batch.pid"
  trap 'rm -f "$OUTPUT_ROOT/batch.pid"' EXIT
  preflight
  pilot
  extract_hiphi
  validate_hiphi
  retry_dataset hiphi
  extract_andy
  validate_andy
  retry_dataset andy
  render_previews
  summarize
}

case "${1:-status}" in
  preflight) preflight ;;
  pilot) pilot ;;
  extract) extract_all ;;
  validate) validate_all ;;
  retry) retry_failed ;;
  render-previews) render_previews ;;
  summarize) summarize ;;
  run) run_all ;;
  start)
    if [[ -f "$OUTPUT_ROOT/batch.pid" ]] && kill -0 "$(cat "$OUTPUT_ROOT/batch.pid")" 2>/dev/null; then
      echo "batch already running with PID $(cat "$OUTPUT_ROOT/batch.pid")"
      exit 0
    fi
    nohup "$0" run >> "$RUN_LOG_ROOT/batch.log" 2>&1 &
    echo "$!" > "$OUTPUT_ROOT/batch.pid"
    echo "started PID $!; log: $RUN_LOG_ROOT/batch.log"
    ;;
  status)
    if [[ -f "$OUTPUT_ROOT/batch.pid" ]] && kill -0 "$(cat "$OUTPUT_ROOT/batch.pid")" 2>/dev/null; then
      echo "running PID $(cat "$OUTPUT_ROOT/batch.pid")"
    else
      echo "not running"
    fi
    stage_log="$(find "$RUN_LOG_ROOT" -maxdepth 1 -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
    if [[ -n "$stage_log" ]]; then
      echo "latest stage log: $stage_log"
      tail -n 30 "$stage_log"
    fi
    ;;
  *)
    echo "usage: $0 {preflight|pilot|extract|validate|retry|render-previews|summarize|run|start|status}" >&2
    exit 2
    ;;
esac
