"""Read-only audit of the eight previously registered actual baseline outputs."""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
import numpy as np
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.native_frames import read as native_frames
from motiogen.evalkit.k.candidate_quality import DIRECTIONS
from motiogen.degradation.metrics import points,EDGES
from evalkit_k_regression import sample_path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('Keep audit versions immutable')
    source=ROOT/'config/gold_k_20261115/evidence.json';samples=json.loads(source.read_text())['baseline_samples'];rows=[]
    for method,entries in samples.items():
        path=sample_path(entries[0]['path']);row=dict(method=method,input=str(path));rows.append(row)
        try:
            m=load(path,method,config=ROOT/'config/evalkit_k4_20261117/input.json');p=points(m)
            for n in m.legacy_prior_names():
                if n in p:p[n]=np.full_like(p[n],np.nan)
            p['pelvis_mid']=(p['left_hip']+p['right_hip'])/2;p['shoulder_mid']=(p['left_shoulder']+p['right_shoulder'])/2
            segments={}
            for start,end in EDGES:
                length=np.linalg.norm(p[end]-p[start],axis=-1);valid=length[np.isfinite(length)&(length>1e-9)]
                segments[start+'→'+end]=dict(observed=len(valid),denominator=len(m.times),length_quantiles_m=np.percentile(valid,[5,50,95]).tolist() if len(valid) else None)
            support={}
            for name,(start,end) in DIRECTIONS.items():
                v=p[end]-p[start];support[name]=int((np.isfinite(v).all(-1)&(np.linalg.norm(v,axis=-1)>1e-9)).sum())
            native=native_frames(m)
            row.update(status='read',frames=len(m.times),time_range_s=m.times[[0,-1]].tolist(),fps=m.fps,
                       uniform_clock=bool(np.allclose(np.diff(m.times),1/m.fps,rtol=1e-3,atol=1e-6)),input_hashes=m.provenance['inputs'],
                       declared_rig=bool(m.provenance.get('input_rig')),raw_segment_rotations=m.rotations is not None,
                       native_rotation_status=native['status'],native_rotation_reason=native.get('reason'),
                       prior_nodes=sorted(m.legacy_prior_names()),position_support=support,segments=segments)
        except Exception as error:row.update(status='failed',reason=f'{type(error).__name__}: {error}')
    result=dict(expected=8,rows=rows,evidence_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),new_holdout_uses=0,
                scope='Existing actual generated outputs only. No paired original R, no fitting or accuracy claim. Future results must be labelled F-D input fidelity; undeclared frame transport stays unknown.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:r.get(k) for k in ['method','status','frames','declared_rig','raw_segment_rotations','native_rotation_status','reason']} for r in rows],indent=2))


if __name__=='__main__':main()
