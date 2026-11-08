#!/usr/bin/env bash
# Full evalkit v1.2 line over a manifest (config/evalkit_v12_20261108.json: repair the floor first, then the loads):
#   pass 1  every dimension on the method floor (residual primary: does the method keep its own floor)
#   pass 2  the self floor of every row (still-landmark level of its pass-1 frames), refitted, every dimension (loads primary)
#   merged per-method check (residual from pass 1, loads from pass 2, method-floor loads alongside), then the distribution
#   smoke of every registered evaluator that has rows (torch python; tmr_soma re-executes itself in the Kimodo environment).
#   scripts/run_evalkit_batch.sh --manifest M.json --out DIR [--workers 6] [--dims residual,ergo,muscle,kinematic[,upper]]
#     [--arms both|self|method] [--lumbar-per-cell N] [--no-distribution] [--offset-reference FROZEN.json] [--no-neck-rotation]
# Arms (config/evalkit_batch_protocol_20261108.json):
#   both    pass 1 every dimension, pass 2 every dimension (the full line above)
#   self    pass 1 only residual and kinematic (the self floor needs its frames), pass 2 every dimension: the primary
#           loads first; rerunning later with both adds the method-floor loads to the pass-1 rows without refitting
#   method  pass 1 every dimension, no pass 2 (method-floor loads only, descriptive)
# --lumbar-per-cell N: upper (lumbar) only on the first N seeds of every method x condition x task x text condition x
#   split; rerunning with a larger N (0: every row) adds it to the newly sampled rows only.
# --lumbar-defer-split S: sampled rows whose split is in S (comma list; default deck_recheck, the reused Kimodo pairs)
#   get their lumbar last: both passes run without it, then a second round adds only their upper (no refit).
# --offset-reference: the frozen per-method offset reference (evalkit_batch.py offset-reference); without it the
#   reference is the manifest median, which moves when rows are streamed in and re-evaluates the method's rows.
# --no-distribution: skip the distribution smoke (fill-in reruns; per-task batches, see the protocol)
# --no-neck-rotation: skip the rotation-reading RULA (sensitivity next to the joint-line RULA on rows with ergo; default on)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSIM_PY="${OSIM_PY:-$HOME/.local/share/opensim-api/bin/python}"
TORCH_PY="${TORCH_PY:-$("$OSIM_PY" -c "import sys; sys.path.insert(0, '$ROOT/utils'); from motiogen.config import data_root, load_config; print(data_root(load_config())/'envs/motiogen-motionlcm/bin/python')")}"
MANIFEST="" OUT="" WORKERS=6 DIMS=residual,ergo,muscle,kinematic ARMS=both LUMBAR="" LUMBAR_DEFER=deck_recheck DISTRIBUTION=1 OFFREF=() NECK=1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest) MANIFEST="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --dims) DIMS="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --lumbar-per-cell) LUMBAR="$2"; shift 2 ;;
    --lumbar-defer-split) LUMBAR_DEFER="$2"; shift 2 ;;
    --no-distribution) DISTRIBUTION=0; shift ;;
    --offset-reference) OFFREF=(--offset-reference "$2"); shift 2 ;;
    --no-neck-rotation) NECK=0; shift ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
[[ -n "$MANIFEST" && -n "$OUT" ]] || { echo "usage: $0 --manifest M.json --out DIR [--workers N] [--dims D] [--arms both|self|method] [--lumbar-per-cell N]" >&2; exit 2; }
[[ "$ARMS" =~ ^(both|self|method)$ ]] || { echo "--arms: both, self or method" >&2; exit 2; }
PASS1="$DIMS"
if [[ "$ARMS" == self ]]; then
  PASS1="$(tr ',' '\n' <<< "$DIMS" | grep -E '^(residual|kinematic)$' | paste -sd, -)"
  [[ ",$PASS1," == *,residual,* ]] || { echo "--arms self needs residual in --dims" >&2; exit 2; }
fi
mkdir -p "$OUT/distribution"
cd "$ROOT"
B=(nice -n "${NICE:-19}" "$OSIM_PY" python/scripts/evalkit_batch.py)
passes() {
  "${B[@]}" run --manifest "$1" --out "$OUT/rows" --workers "$WORKERS" --dims "$PASS1" "${OFFREF[@]}"
  [[ "$NECK" == 1 && ",$PASS1," == *,ergo,* ]] && "${B[@]}" neck-rotation --manifest "$1" --rows "$OUT/rows" --workers "$WORKERS"
  [[ "$ARMS" == method ]] && return 0
  "${B[@]}" manifest-self-floor --manifest "$1" --rows "$OUT/rows" --out "$OUT/manifest_selffloor.json"
  "${B[@]}" run --manifest "$OUT/manifest_selffloor.json" --out "$OUT/rows_selffloor" --workers "$WORKERS" --dims "$DIMS"
  [[ "$NECK" == 1 && ",$DIMS," == *,ergo,* ]] && "${B[@]}" neck-rotation --manifest "$OUT/manifest_selffloor.json" --rows "$OUT/rows_selffloor" --workers "$WORKERS"
  return 0
}
if [[ -n "$LUMBAR" ]]; then
  "${B[@]}" manifest-lumbar-sample --manifest "$MANIFEST" --per-cell "$LUMBAR" --defer-split "$LUMBAR_DEFER" --out "$OUT/manifest_lumbar_first.json"
  passes "$OUT/manifest_lumbar_first.json"
  "${B[@]}" manifest-lumbar-sample --manifest "$MANIFEST" --per-cell "$LUMBAR" --out "$OUT/manifest_lumbar.json"
  MANIFEST="$OUT/manifest_lumbar.json"
  DEFERRED="$("$OSIM_PY" -c "import json, sys; print(json.load(open(sys.argv[1]))['lumbar_subsample']['deferred'])" "$OUT/manifest_lumbar_first.json")"
  [[ "$DEFERRED" == 0 ]] || passes "$MANIFEST"
else
  passes "$MANIFEST"
fi
if [[ "$ARMS" == method ]]; then
  "${B[@]}" check --manifest "$MANIFEST" --rows "$OUT/rows" --out "$OUT/check.json"
else
  "${B[@]}" check --manifest "$MANIFEST" --rows "$OUT/rows" --self-manifest "$OUT/manifest_selffloor.json" --self-rows "$OUT/rows_selffloor" \
    --out "$OUT/check.json"
fi
[[ "$DISTRIBUTION" == 1 ]] || exit 0
for E in t2m omomo flowmdm_babel tmr_soma; do
  nice -n "${NICE:-19}" "$TORCH_PY" python/scripts/evalkit_distribution.py smoke --evaluator "$E" --manifest "$MANIFEST" --out "$OUT/distribution"
done
