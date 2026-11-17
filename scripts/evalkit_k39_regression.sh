#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
K39_PYTHON=/home/heinrich/.local/share/opensim-api/bin/python
K39_OUTPUT=/media/heinrich/9bbdd73e-4410-4279-9639-6161f0c958391/MorionGen/outputs
"$K39_PYTHON" python/scripts/evalkit_k39_regression.py --out "$K39_OUTPUT/evalkit_k39_full_v2_20261117" --workers 3
"$K39_PYTHON" python/scripts/evalkit_k39_real_samples.py --out "$K39_OUTPUT/evalkit_k39_real_v2_20261117" --workers 3
"$K39_PYTHON" python/scripts/evalkit_k39_summarize.py "$K39_OUTPUT/evalkit_k39_full_v2_20261117"
"$K39_PYTHON" python/scripts/evalkit_k39_visualize.py "$K39_OUTPUT/evalkit_k39_full_v2_20261117"
