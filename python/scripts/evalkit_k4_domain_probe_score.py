"""Independent original-R scoring of a D-only diagnostic; three frames, no acceptance."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
from motiogen.evalkit.k.config import ROOT
from motiogen.evalkit.k.motion import load,Motion
from motiogen.evalkit.k.scoring import score
from motiogen.degradation.reference import sample
from evalkit_k4_ablation_score import add_pelvis_diagnostic
from evalkit_k4_score import serializable


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','probe'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('window','method'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=a.probe/'independent_scores.json'
    if out.exists():raise ValueError('Immutable score snapshot exists')
    frozen=Path(json.loads((ROOT/'config/k_degradation_20261117/freeze.json').read_text())['run'])
    job=json.loads((a.run/'jobs'/f'{a.window}_{a.method}.json').read_text())
    d=load(job['input'],a.method,config=ROOT/'config/evalkit_k4_20261117/input.json')
    with np.load(frozen/'references'/f'{a.window}.npz') as z:r=Motion(z['positions'],list(z['names']),z['times'],'reference')
    identity=json.loads((a.probe/'identity.json').read_text());indices=identity['indices'];clock=d.times[indices]
    rd=sample(r,clock);dd=sample(d,clock);rows=[]
    for condition in identity['protocol']['conditions']:
        for seed in identity['protocol']['seeds']:
            arrays=[]
            for frame in indices:
                with np.load(a.probe/condition/f'{frame}_{seed}'/'probe.npz') as z:
                    arrays.append(z['positions']);names=list(z['names'])
            positions=np.concatenate(arrays)
            # Same declared root rule as full-window runtime: observed hip midpoint.
            for j,frame in enumerate(indices):
                target=d.positions[frame,[d.names.index(n) for n in ('left_hip','right_hip')]].mean(0)
                current=positions[j,[names.index(n) for n in ('left_hip','right_hip')]].mean(0)
                positions[j]+=target-current
            f=Motion(positions,names,clock,'model_fk')
            result=score(rd,dd,f,normalization_reference=r,fixed_observation_support=True)
            add_pelvis_diagnostic(result,rd,dd,f,np.zeros(len(clock),bool))
            rows.append({'condition':condition,'seed':seed,'score':result})
            x=result['comparisons']['F-R']['full']
            print(condition,seed,'max observed direction',max(v['max'] for v in x['directions_deg'].values() if v['n']),
                  'knees',{n:x['joint_angles_deg'][n]['max'] for n in ('left_knee','right_knee')})
    out.write_text(json.dumps(serializable({'scope':'Prespecified isolated frames only; no full-window or temporal acceptance',
        'reference':str(frozen/'references'/f'{a.window}.npz'),'rows':rows}),indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
