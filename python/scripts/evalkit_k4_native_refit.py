"""D-only full-window solver control on an immutable existing model and IK seed."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'utils'), str(ROOT / 'utils/motiogen')]
import opensim as osim
import biomech
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.refinement import refine
from motiogen.evalkit.k.k4_fitting import write_coordinates
from motiogen.evalkit.k.timebase import native_clock


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'input', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--method', required=True)
    args = parser.parse_args()
    osim.Logger.setLevelString('error')
    config = json.loads(args.config.read_text())
    motion = load(args.input, args.method, config=ROOT/'config/evalkit_k4_20261117/input.json')
    source_receipt = json.loads((args.source/'receipt.json').read_text())
    if source_receipt['key']['inputs'] != motion.provenance['inputs']:
        raise ValueError('Frozen D identity changed')
    source_model = args.source/'model.osim'
    source_seed = args.source/'ik/ik.mot'
    if sha(source_model) != config['source_model_sha256'] or sha(source_seed) != config['source_seed_sha256']:
        raise ValueError('Frozen model or seed changed')
    if shutil.disk_usage(args.output.parent).free < 80*1024**3:
        raise RuntimeError('80 GiB reserve required')
    args.output.mkdir(exist_ok=False)
    shutil.copyfile(source_model, args.output/'model.osim')
    time, labels, q = biomech.read_storage(source_seed)
    time, clock = native_clock(time, motion.times)
    points, names, _ = motion.legacy()
    refined, report = refine(args.output/'model.osim', labels, q, points, names,
                             config['refinement'], times=motion.times)
    write_coordinates(args.output/'coordinates_refined_native.mot', time, labels, refined)
    (args.output/'refinement.json').write_text(json.dumps(report, indent=2)+'\n')
    receipt = {'key': source_receipt['key'], 'source_fit': str(args.source),
               'source_model_sha256': sha(source_model), 'source_seed_sha256': sha(source_seed),
               'native_clock': clock, 'config': config, 'config_sha256': sha(args.config),
               'commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']).decode().strip(),
               'implementation_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                   [Path(__file__), *sorted((ROOT/'utils/motiogen/evalkit/k').rglob('*.py'))]},
               'scope': 'Solver-only full-window control: unchanged model, ranges, registration and native IK seed; D only. No default promotion.',
               'kinematic_only': True, 'force_eligible': False}
    (args.output/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({'status': 'completed', 'frames': len(time), 'model_unchanged': sha(source_model)==sha(args.output/'model.osim')}))


if __name__ == '__main__':
    main()
