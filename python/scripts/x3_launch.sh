#!/usr/bin/env bash
# X3 orchestration (config/x3_cross_method_layers_20261104.json).
# Usage: x3_launch.sh gen|dart|omnicontrol|kimodo|e STAGE   (STAGE = pilot | confirm | top_up)
#   dart:        DART generation + W1 equivariance check on X3_DART_GPU (0), then its output-layer arms (CPU)
#   omnicontrol: OmniControl generation on X3_OMNI_GPU (1), then its output-layer arms (CPU)
#   gen:         dart and omnicontrol in parallel
#   kimodo: Kimodo generation on X3_KIMODO_GPU (0), retried every 600 s while its reviewed RAM gate refuses
#   e:      E plan for the finished generations, then runner (4 workers) and guard; refuses while a runner is alive
set -u
PART=$1; STAGE=$2
cd "$(dirname "$0")/../.."
ENVS=/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/envs
OPENSIM=$HOME/.local/share/opensim-api/bin/python
LOG=develop_logs/test_runs/20261104_x3/logs; mkdir -p "$LOG"
export PYTHONNOUSERSITE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
stamp() { echo "$(date -Iseconds) $*" >> "$LOG/launch.log"; }
x3() { # env gpu log args...
  local env=$1 gpu=$2 log=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpu "$ENVS/$env/bin/python" python/scripts/run_x3.py "$@" >> "$LOG/$log" 2>&1
}

dart_chain() {
  local gpu=${X3_DART_GPU:-0}
  stamp "$STAGE dart start (gpu $gpu)"; x3 motiogen-dart "$gpu" "${STAGE}_dart.log" --method dart --stage "$STAGE"; stamp "$STAGE dart exit=$?"
  x3 motiogen-dart "$gpu" "${STAGE}_dart.log" --method dart --stage "$STAGE" --part equivariance; stamp "$STAGE dart equivariance exit=$?"
  x3 motiogen-kimodo "" "${STAGE}_dart_post.log" --method dart --stage "$STAGE" --part post; stamp "$STAGE dart post exit=$?"
  touch "$LOG/${STAGE}_dart_done"
}
omni_chain() {
  local gpu=${X3_OMNI_GPU:-1}
  stamp "$STAGE omnicontrol start (gpu $gpu)"; x3 motiogen-omnicontrol "$gpu" "${STAGE}_omnicontrol.log" --method omnicontrol --stage "$STAGE"
  stamp "$STAGE omnicontrol exit=$?"
  x3 motiogen-kimodo "" "${STAGE}_omnicontrol_post.log" --method omnicontrol --stage "$STAGE" --part post; stamp "$STAGE omnicontrol post exit=$?"
  touch "$LOG/${STAGE}_omnicontrol_done"
}
case $PART in
dart) dart_chain ;;
omnicontrol) omni_chain ;;
gen)
  dart_chain & omni_chain &
  wait; stamp "$STAGE gen done"; touch "$LOG/${STAGE}_gen_done" ;;
kimodo)
  for attempt in $(seq 1 144); do
    x3 motiogen-kimodo "${X3_KIMODO_GPU:-0}" "${STAGE}_kimodo.log" --method kimodo --stage "$STAGE"; code=$?
    if tail -3 "$LOG/${STAGE}_kimodo.log" | grep -q ResourceGate; then
      stamp "$STAGE kimodo gate refused (attempt $attempt)"; sleep 600
    else
      stamp "$STAGE kimodo exit=$code"; break
    fi
  done
  touch "$LOG/${STAGE}_kimodo_done" ;;
e)
  OUT=develop_logs/test_runs/20261104_x3/e
  if pgrep -f '[p]ython/scripts/run_x3_e.py --workers' > /dev/null; then stamp "$STAGE e refused: runner alive"; exit 1; fi
  "$ENVS/motiogen-ardy/bin/python" python/scripts/run_x3_e.py --plan "$STAGE" >> "$LOG/${STAGE}_e_plan.log" 2>&1
  stamp "$STAGE e plan exit=$?"
  rm -f "$OUT/guard_done" "$OUT/progress.json"
  nohup setsid "$OPENSIM" python/scripts/run_x3_e.py --workers 4 >> "$OUT/worker.log" 2>&1 < /dev/null &
  sleep 5
  PROTO=$([ "$STAGE" = pilot ] && echo x3_pilot || echo x3)
  nohup setsid "$ENVS/motiogen-ardy/bin/python" python/scripts/e1_guard.py --protocol "$PROTO" >> "$OUT/guard_$STAGE.out" 2>&1 < /dev/null &
  stamp "$STAGE e runner and guard ($PROTO) launched" ;;
esac
