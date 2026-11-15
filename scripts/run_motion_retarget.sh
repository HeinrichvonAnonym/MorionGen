#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${MOTION_RETARGET_CONFIG:-$ROOT/config/motion_retarget.yaml}"
PYTHON="${MOTION_RETARGET_PYTHON:-/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen/envs/motiogen-smplsim/bin/python}"
export PYTHONNOUSERSITE=1

if [[ ! -x "$PYTHON" ]]; then
  echo "motion-retarget environment is missing: $PYTHON" >&2
  echo "Run scripts/setup_motion_retarget.sh first." >&2
  exit 2
fi

exec "$PYTHON" "$ROOT/python/scripts/motion_retarget.py" --config "$CONFIG" "$@"
