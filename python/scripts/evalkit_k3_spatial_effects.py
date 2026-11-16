#!/usr/bin/env python3
"""Paired changes in endpoints, velocities and common-plane contact diagnostics."""
import argparse,json,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen'),str(ROOT/'python/scripts')]
from motiogen.evalkit.k.bones import NAMES
from motiogen.evalkit.k.scoring import stats
from evalkit_k3_report import participant_mean,VARIANTS
JOINTS=['left_wrist','right_wrist','left_foot','right_foot']


def changes(a,b,times,floor):
    result={}
    velocity_a=np.gradient(a,times,axis=0);velocity_b=np.gradient(b,times,axis=0)
    for name in JOINTS:
        j=NAMES.index(name);valid=np.isfinite(a[:,j]).all(1)&np.isfinite(b[:,j]).all(1)
        value={'displacement_m':stats(np.where(valid,np.linalg.norm(a[:,j]-b[:,j],axis=-1),np.nan),False),
            'velocity_difference_mps':stats(np.where(valid,np.linalg.norm(velocity_a[:,j]-velocity_b[:,j],axis=-1),np.nan),False)}
        if name.endswith('foot'):
            near_a=a[:,j,1]-floor<=.03;near_b=b[:,j,1]-floor<=.03
            below_a=a[:,j,1]-floor<-.01;below_b=b[:,j,1]-floor<-.01
            value['near_plane_switch_share']=float(np.mean((near_a^near_b)[valid])) if valid.any() else None
            value['below_plane_switch_share']=float(np.mean((below_a^below_b)[valid])) if valid.any() else None
        result[name]=value
    return result


def one(row):
    with np.load(row['arrays']) as z:a={key:z[key] for key in z.files if key in ['D','G','times'] or key.startswith(('P:','F:'))}
    original=a['G'] if 'G' in a else a['D'];floor=float(np.nanpercentile(original[:,[5,9],1],5))
    variants={v:{'projection':changes(a['P:'+v],original,a['times'],floor),
        'optimization':changes(a['F:'+v],a['F:legacy_raw'],a['times'],floor) if 'F:legacy_raw' in a else None} for v in VARIANTS if 'F:'+v in a}
    rec={'id':row.get('id',row['source']),'method':row['method'],'group_id':row.get('group_id',row['method']),
        'variants':variants,'scope':'common original input-derived plane and common physical clock; velocity-vector difference, not difference of speed norms'}
    path=Path(row['arrays']).with_name('spatial_effects.json');path.write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
    rec['path']=str(path);return rec


def aggregate(rows):
    result={}
    for method in sorted({r['method'] for r in rows}):
        pool=[r for r in rows if r['method']==method];vout={}
        for variant in VARIANTS:
            available=[r for r in pool if variant in r['variants']];terms={}
            for kind in ['projection','optimization']:
                measured=[r for r in available if r['variants'][variant][kind] is not None];joints={}
                for joint in JOINTS:
                    joints[joint]={key:participant_mean(measured,lambda r:r['variants'][variant][kind][joint][key]['p95_abs']) for key in ['displacement_m','velocity_difference_mps']}
                    if joint.endswith('foot'):
                        joints[joint].update({key:participant_mean(measured,lambda r:r['variants'][variant][kind][joint][key]) for key in ['near_plane_switch_share','below_plane_switch_share']})
                terms[kind]=joints
            vout[variant]={'records':len(available),**terms}
        result[method]={'records':len(pool),'participants':len({r['group_id'] for r in pool}),'variants':vout}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--workers',type=int,default=3);a=ap.parse_args()
    controlled=[json.loads(p.read_text()) for p in (a.out/'fits').glob('*/*/result.json')]
    controlled=[r for r in controlled if r['status']=='computed']
    actual=[r for r in json.loads((a.out/'actual_summary.json').read_text()) if r['status']=='computed']
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        c=list(pool.map(one,controlled));g=list(pool.map(one,actual))
    result={'scope':'changes caused by projection or selected-fit geometry; contact is a landmark proxy, without measured ground/support forces',
        'controlled':aggregate(c),'actual_output_fidelity':aggregate(g),'record_paths':[r['path'] for r in c+g],
        'floor':'5th percentile of original input foot-joint heights, shared by all views',
        'velocity':'numpy gradient on the common fitted clock; descriptive kinematic change, not an independent velocity observation'}
    (ROOT/'config/evalkit_k3_20261116/spatial_effects_summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('spatial effects',len(c),'controlled;',len(g),'actual')


if __name__=='__main__':main()
