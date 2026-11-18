"""Audit point/support aliases on the frozen656 D inputs without fitting or R."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
from motiogen.evalkit.k.motion import load
from motiogen.degradation.metrics import points


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Audit output already exists')
    jobs=sorted((a.run/'jobs').glob('[0-9][0-9][0-9]_*.json'))
    if len(jobs)!=656:raise ValueError('Expected fixed656 job denominator')
    rows=[]
    for job in jobs:
        raw=json.loads(job.read_text());w=job.stem.split('_',1)[0];method=raw['method']
        motion=load(raw['input'],method,config=ROOT/'config/evalkit_k4_20261117/input.json')
        expected=json.loads((a.run/'common_v1'/w/method/'receipt.json').read_text())['key']['inputs']
        if motion.provenance['inputs']!=expected:raise ValueError('Frozen input identity changed: '+job.stem)
        raw_prior=set(motion.provenance.get('degradation',{}).get('prior_joints',[]))
        old={n.removesuffix('_body') for n in raw_prior};new=motion.legacy_prior_names()
        canonical=points(motion);changes=[]
        _,names,_=motion.legacy()
        if len(names)!=len(set(names)):raise ValueError('Canonical node collision')
        for name,position in canonical.items():
            count=int(np.isfinite(position).all(-1).sum())
            before=0 if name in old else count;after=0 if name in new else count
            if before!=after:changes.append(dict(metric_node=name,old_supported=before,new_supported=after,total=len(motion.times)))
        rows.append(dict(window=w,method=method,prior_raw=sorted(raw_prior),prior_canonical=sorted(new),changes=changes,
                         inputs_sha256=motion.provenance['inputs']))
    methods=sorted({r['method'] for r in rows});summary={m:dict(records=sum(r['method']==m for r in rows),changed_records=sum(r['method']==m and bool(r['changes']) for r in rows)) for m in methods}
    if len(methods)!=8 or any(v['records']!=82 for v in summary.values()):raise ValueError('82x8 denominator mismatch')
    result=dict(expected=656,completed=len(rows),summary=summary,records=rows,new_holdout_uses=0,
                scope='Frozen D-only node/support alias audit; changes are K runtime support semantics, not altered input or scoring masks. Observed body wrist remains distinct from prior hand tip.',
                implementation_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),ROOT/'utils/motiogen/evalkit/k/motion.py']})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
