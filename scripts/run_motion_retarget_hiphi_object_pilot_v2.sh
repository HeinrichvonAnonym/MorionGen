#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MOTION_RETARGET_OBJECT_CONFIG="$ROOT/config/motion_retarget_hiphi_object_v2.yaml"
export MOTION_RETARGET_OUTPUT_ROOT="/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/outputs/motion_retarget_hiphi_object_pilot_v2"
export MOTION_RETARGET_OBJECT_LOG_ROOT="$ROOT/develop_logs/test_runs/motion_retarget_hiphi_object_pilot_v2"
exec "$ROOT/scripts/run_motion_retarget_hiphi_object_pilot.sh" "$@"
