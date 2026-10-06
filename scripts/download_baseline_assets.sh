#!/usr/bin/env bash
set -euo pipefail
MOTIOGEN_DOWNLOAD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$MOTIOGEN_DOWNLOAD_ROOT/utils${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
exec flock -n "$MOTIOGEN_DOWNLOAD_ROOT/develop_logs/test_runs/.asset-download.lock" /home/heinrich/anaconda3/envs/motiogen-tools/bin/python "$MOTIOGEN_DOWNLOAD_ROOT/python/scripts/download_baseline_assets.py" "$@"
