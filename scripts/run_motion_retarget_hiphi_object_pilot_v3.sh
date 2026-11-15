#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${MORIONGEN_DATA_ROOT:-/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c95839/MorionGen}"
PYTHON="$DATA_ROOT/envs/motiogen-smplsim/bin/python"
CONFIG="$ROOT/config/motion_retarget_hiphi_object_v3.yaml"
SELECTION="$ROOT/config/hiphi_object_pilot_selection_v3.json"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing motion-retarget Python: $PYTHON" >&2
  exit 1
fi

command_name="${1:-}"
shift || true
case "$command_name" in
  inventory|extract|validate|render|render-previews)
    exec "$PYTHON" "$ROOT/python/scripts/motion_retarget.py" \
      --config "$CONFIG" "$command_name" --dataset hiphi --profile source_native \
      --with-objects --selection-manifest "$SELECTION" "$@"
    ;;
  summarize)
    exec "$PYTHON" "$ROOT/python/scripts/motion_retarget.py" \
      --config "$CONFIG" summarize --with-objects --selection-manifest "$SELECTION" "$@"
    ;;
  *)
    echo "Usage: $0 {inventory|extract|validate|render|render-previews|summarize} [args...]" >&2
    exit 2
    ;;
esac
