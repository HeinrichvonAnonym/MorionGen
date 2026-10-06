#!/usr/bin/env bash
set -euo pipefail
MOTIOGEN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MOTIOGEN_CONDA="${MOTIOGEN_CONDA:-/home/heinrich/anaconda3/condabin/conda}"
export PYTHONPATH="$MOTIOGEN_ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$MOTIOGEN_ROOT/develop_logs/test_runs/.cache/matplotlib"
exec "$MOTIOGEN_CONDA" run --no-capture-output -n motiogen-tools python "$MOTIOGEN_ROOT/python/scripts/baseline.py" "$@"
