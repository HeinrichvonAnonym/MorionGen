"""Read Han raw markers in the existing Nimble environment; no modelling or scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import nimblephysics as nimble

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
ev=json.loads((ROOT/'config/gold_k_20261115/evidence.json').read_text())
rows=[json.loads(l) for l in (ROOT/'config/gold_k_20261115/trials.jsonl').read_text().splitlines()]
rows=[r for r in rows if r['dataset']=='han' and r['status']=='included']
args.out.mkdir(parents=True,exist_ok=True);current=None;s=None
for i,row in enumerate(rows):
    src=row['source'];target=args.out/(hashlib.sha256(row['id'].encode()).hexdigest()[:20]+'.npz')
    if target.exists():continue
    if current!=src['file']:
        current=src['file'];s=nimble.biomechanics.SubjectOnDisk(str(Path(ev['local_roots']['han'])/current))
    ti=next(j for j in range(s.getNumTrials()) if s.getTrialName(j)==src['trial'])
    fs=s.readFrames(ti,0,s.getTrialLength(ti),includeSensorData=True,includeProcessingPasses=False)
    names=sorted({k for f in fs for k,v in f.markerObservations});idx={n:j for j,n in enumerate(names)}
    p=np.full((len(fs),len(names),3),np.nan)
    for j,f in enumerate(fs):
        for k,v in f.markerObservations:p[j,idx[k]]=v
    np.savez_compressed(target,points=p,names=names,times=np.arange(len(fs))*s.getTrialTimestep(ti),record_id=row['id'])
    print(f'{i+1}/{len(rows)} {row["subject"]}',flush=True)
