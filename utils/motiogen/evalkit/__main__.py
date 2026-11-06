"""python -m evalkit --run DIR --out DIR --floor M --reference VARIANT [--mass KG] [--method KEY] [--prompt TEXT]
[--dims residual,ergo,muscle]. OpenSim python, utils/motiogen on PYTHONPATH (scripts/run_evalkit.sh)."""
import argparse
import json

from .run import DIMS, evaluate

a = argparse.ArgumentParser(prog='evalkit')
a.add_argument('--run', required=True, help='directory with motion.npz (canonical Z-up joints)')
a.add_argument('--out', required=True)
a.add_argument('--floor', type=float, required=True, help='floor height of this motion (m, canonical Z)')
a.add_argument('--reference', required=True, help='floor-reference variant (config/e_v2_floor_reference_20261106.json or the evalkit calibrated entries)')
a.add_argument('--mass', type=float, default=75.)
a.add_argument('--method', help='ergo neutral-offset key (default: the reference variant)')
a.add_argument('--prompt', help='task text; none means object-free')
a.add_argument('--dims', default=','.join(DIMS))
args = a.parse_args()
print(json.dumps(evaluate(args.run, args.out, args.floor, args.reference, args.mass, args.method, args.prompt,
                          tuple(args.dims.split(','))), indent=1))
