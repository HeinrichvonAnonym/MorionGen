#!/usr/bin/env bash
# One evalkit evaluation (residual force, RULA/REBA classes, muscle load) of a run directory with motion.npz.
#   scripts/run_evalkit.sh --run DIR --out DIR --floor M --reference VARIANT [--mass KG] [--method KEY] [--prompt TEXT] [--dims residual,ergo,muscle]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OSIM_PY="${OSIM_PY:-$HOME/.local/share/opensim-api/bin/python}"
export PYTHONPATH="$ROOT/utils/motiogen${PYTHONPATH:+:$PYTHONPATH}"
exec nice -n "${NICE:-19}" "$OSIM_PY" -m evalkit "$@"
