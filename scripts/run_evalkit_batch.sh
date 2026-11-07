#!/usr/bin/env bash
# Full evalkit line over a manifest: residual / RULA-REBA / muscle / kinematic per row (OpenSim python), per-method sanity
# check, then the distribution smoke of every registered evaluator that has rows (torch python; tmr_soma re-executes
# itself in the Kimodo environment).
#   scripts/run_evalkit_batch.sh --manifest M.json --out DIR [--workers 6] [--dims residual,ergo,muscle,kinematic]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSIM_PY="${OSIM_PY:-$HOME/.local/share/opensim-api/bin/python}"
TORCH_PY="${TORCH_PY:-$("$OSIM_PY" -c "import sys; sys.path.insert(0, '$ROOT/utils'); from motiogen.config import data_root, load_config; print(data_root(load_config())/'envs/motiogen-motionlcm/bin/python')")}"
MANIFEST="" OUT="" WORKERS=6 DIMS=residual,ergo,muscle,kinematic
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --dims) DIMS="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MANIFEST" && -n "$OUT" ]] || { echo "usage: $0 --manifest M.json --out DIR [--workers N] [--dims D]" >&2; exit 2; }
mkdir -p "$OUT/distribution"
cd "$ROOT"
nice -n "${NICE:-19}" "$OSIM_PY" python/scripts/evalkit_batch.py run --manifest "$MANIFEST" --out "$OUT/rows" --workers "$WORKERS" --dims "$DIMS"
nice -n "${NICE:-19}" "$OSIM_PY" python/scripts/evalkit_batch.py check --manifest "$MANIFEST" --rows "$OUT/rows" --out "$OUT/check.json"
for E in t2m omomo flowmdm_babel tmr_soma; do
  nice -n "${NICE:-19}" "$TORCH_PY" python/scripts/evalkit_distribution.py smoke --evaluator "$E" --manifest "$MANIFEST" --out "$OUT/distribution"
done
