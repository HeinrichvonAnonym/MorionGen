#!/usr/bin/env bash
# Full evalkit v1.2 line over a manifest (config/evalkit_v12_20261108.json: repair the floor first, then the loads):
#   pass 1  every dimension on the method floor (residual primary: does the method keep its own floor)
#   pass 2  the self floor of every row (still-landmark level of its pass-1 frames), refitted, every dimension (loads primary)
#   merged per-method check (residual from pass 1, loads from pass 2, method-floor loads alongside), then the distribution
#   smoke of every registered evaluator that has rows (torch python; tmr_soma re-executes itself in the Kimodo environment).
#   scripts/run_evalkit_batch.sh --manifest M.json --out DIR [--workers 6] [--dims residual,ergo,muscle,kinematic[,upper]]
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
B=(nice -n "${NICE:-19}" "$OSIM_PY" python/scripts/evalkit_batch.py)
"${B[@]}" run --manifest "$MANIFEST" --out "$OUT/rows" --workers "$WORKERS" --dims "$DIMS"
"${B[@]}" manifest-self-floor --manifest "$MANIFEST" --rows "$OUT/rows" --out "$OUT/manifest_selffloor.json"
"${B[@]}" run --manifest "$OUT/manifest_selffloor.json" --out "$OUT/rows_selffloor" --workers "$WORKERS" --dims "$DIMS"
"${B[@]}" check --manifest "$MANIFEST" --rows "$OUT/rows" --self-manifest "$OUT/manifest_selffloor.json" --self-rows "$OUT/rows_selffloor" \
  --out "$OUT/check.json"
for E in t2m omomo flowmdm_babel tmr_soma; do
  nice -n "${NICE:-19}" "$TORCH_PY" python/scripts/evalkit_distribution.py smoke --evaluator "$E" --manifest "$MANIFEST" --out "$OUT/distribution"
done
