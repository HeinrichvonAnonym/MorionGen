#!/usr/bin/env bash
set -euo pipefail
MOTIOGEN_DOWNLOAD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MOTIOGEN_DATA_ROOT="${MORIONGEN_DATA_ROOT:-$(/home/heinrich/anaconda3/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_root"])' "$MOTIOGEN_DOWNLOAD_ROOT/config/baselines.json")}"
export TMPDIR="$MOTIOGEN_DATA_ROOT/package-cache/tmp"
mkdir -p "$TMPDIR"
export PYTHONPATH="$MOTIOGEN_DOWNLOAD_ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec flock -n "$MOTIOGEN_DOWNLOAD_ROOT/develop_logs/test_runs/.asset-download.lock" /home/heinrich/anaconda3/bin/python "$MOTIOGEN_DOWNLOAD_ROOT/python/scripts/download_baseline_assets.py" "$@"
