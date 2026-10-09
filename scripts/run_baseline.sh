#!/usr/bin/env bash
set -euo pipefail
MOTIOGEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MOTIOGEN_DRIVER_PYTHON="${MOTIOGEN_DRIVER_PYTHON:-/home/heinrich/anaconda3/bin/python}"
MOTIOGEN_DATA_ROOT="${MORIONGEN_DATA_ROOT:-$($MOTIOGEN_DRIVER_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_root"])' "$MOTIOGEN_ROOT/config/baselines.json")}"
export TMPDIR="$MOTIOGEN_DATA_ROOT/package-cache/tmp"
mkdir -p "$TMPDIR"
export PYTHONPATH="$MOTIOGEN_ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$MOTIOGEN_ROOT/develop_logs/test_runs/.cache/matplotlib"
exec "$MOTIOGEN_DRIVER_PYTHON" "$MOTIOGEN_ROOT/python/scripts/baseline.py" "$@"
