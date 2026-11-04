#!/usr/bin/env bash
# X3 unattended chain: once OmniControl confirm, Kimodo confirm and DART top_up generation are done and the
# pilot E runner has exited, run the E criterion over all cells (seeds 40-42) through the guard, then analyse.
set -u
cd "$(dirname "$0")/../.."
X3=develop_logs/test_runs/20261104_x3; LOG=$X3/logs
ENVS=/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/envs
stamp() { echo "$(date -Iseconds) chain: $*" >> "$LOG/launch.log"; }
stamp "waiting for confirm generation"
until [ -e "$LOG/confirm_omnicontrol_done" ] && [ -e "$LOG/confirm_kimodo_done" ] && [ -e "$LOG/top_up_dart_done" ]; do sleep 300; done
while pgrep -f '[p]ython/scripts/run_x3_e.py --workers' > /dev/null; do sleep 300; done
python/scripts/x3_launch.sh e top_up
sleep 60
until [ -e "$X3/e/guard_done" ]; do sleep 300; done
for stage in confirm top_up; do
  PYTHONNOUSERSITE=1 "$ENVS/motiogen-ardy/bin/python" python/scripts/analyze_x3.py --stage $stage > "$LOG/analysis_$stage.txt" 2>&1
  stamp "analysis $stage exit=$?"
done
touch "$LOG/chain_done"
