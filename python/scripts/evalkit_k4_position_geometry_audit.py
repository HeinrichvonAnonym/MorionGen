"""Audit fixed geometry policies on frozen position inputs without fitting or R."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--code-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path[:0]=[str(args.code_root/'utils'),str(args.code_root/'utils/motiogen')]
    from motiogen.evalkit.k.motion import load
    from motiogen.evalkit.k.position_geometry import trunk_chain_proxy
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    config=args.code_root/'config/evalkit_k4_20261117/input.json'
    policies=['observed_chain_proxy_v1','observed_chain_total_proxy_v1'];rows=[];sources={str(config):sha(config)}
    for method in ('motionlcm','omnicontrol'):
        jobs=sorted((args.run/'jobs').glob('*_'+method+'.json'))
        if len(jobs)!=82:raise ValueError('Expected fixed82 windows per position format')
        for path in jobs:
            job=json.loads(path.read_text());sources[str(path)]=sha(path)
            source=Path(job['input']);files=sorted(source.rglob('*')) if source.is_dir() else [source]
            hashes={str(p):sha(p) for p in files if p.is_file()}
            motion=load(source,method,config=config)
            values={policy:trunk_chain_proxy(motion,policy) for policy in policies}
            if hashes!={str(p):sha(p) for p in files if p.is_file()}:raise ValueError('Input changed during audit')
            rows.append({'window':path.stem.split('_')[0],'method':method,'input_hashes':hashes,
                         'frames':len(motion.times),'policies':{p:{'estimate_m':v,'receipt':r} for p,(v,r) in values.items()},
                         'geometry_target_changed':values[policies[0]][0]!=values[policies[1]][0]})
    result={'expected':164,'completed':len(rows),'changed':sum(r['geometry_target_changed'] for r in rows),'records':rows,
            'source_sha256':sources,'implementation_sha256':sha(args.code_root/'utils/motiogen/evalkit/k/position_geometry.py'),
            'scope':'Input-only proxy eligibility and fixed length targets; no fitting, no R, no accuracy or migration qualification. Unchanged targets alone do not replace final same-version656 regression.',
            'new_holdout_uses':0}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError('Keep previous audit; choose a new output')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({k:result[k] for k in ('expected','completed','changed')})


if __name__=='__main__':main()
