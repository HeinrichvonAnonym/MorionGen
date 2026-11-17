#!/usr/bin/env bash
# Three bounded CPU workers; never launches a K fit or a holdout evaluation.
set -euo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
task_python="${DEGRADATION_PYTHON:-/home/heinrich/.local/share/opensim-api/bin/python}"
if [[ $# -ne 2 ]]; then
  echo 'Usage: evalkit_degradation_audit.sh OUTPUT_DIR BASELINE_AUDIT_DIR' >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$task_root"
task_log="develop_logs/test_runs/20261117_degradation/$(basename "$1")"
mkdir -p "$task_log"
task_pids=()
trap 'for task_pid in "${task_pids[@]}"; do kill "$task_pid" 2>/dev/null || true; done' INT TERM
for task_span in '0 28' '28 55' '55 82'; do
  read -r task_start task_stop <<< "$task_span"
  "$task_python" python/scripts/evalkit_degradation_audit.py \
    --source-geometry --output "$1" --baseline-run "$2" --resume \
    --start "$task_start" --stop "$task_stop" > "$task_log/$task_start.log" 2>&1 &
  task_pids+=("$!")
done
task_failed=0
for task_pid in "${task_pids[@]}"; do
  wait "$task_pid" || task_failed=1
done
exit "$task_failed"
