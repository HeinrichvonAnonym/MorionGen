#!/usr/bin/env python3
"""Record model constraints at actual-output knee-error peaks; no policy changes."""
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.scoring import angles


def read_fit(folder,variant):
    path=folder/variant
    with np.load(path/'kinematics.npz') as z:data={k:z[k] for k in z.files}
    model=ET.parse(path/'model.osim')
    definitions={e.attrib['name']:{'range_deg':(np.fromstring(e.findtext('range',''),sep=' ')*180/np.pi).tolist(),
                  'locked':e.findtext('locked','false')=='true'} for e in model.findall('.//Coordinate')}
    return data,definitions


def inspect(folder,variant):
    data,definitions=read_fit(folder,variant);labels=list(data['labels']);coords=data['coordinates']
    source=angles(common(data['source'],list(data['names'])));fit=angles(common(data['fitted'],list(data['names'])))
    result={}
    for side,s in [('left','l'),('right','r')]:
        metric=side+'_knee_flexion';error=abs(fit[metric]-source[metric]);index=int(np.nanargmax(error));q={}
        for prefix in ['hip_flexion','hip_adduction','hip_rotation','knee_angle','ankle_angle','subtalar_angle']:
            name=prefix+'_'+s;v=coords[:,labels.index(name)];d=definitions[name];lo,hi=d['range_deg']
            q[name]={**d,'at_error_peak_deg':float(v[index]),'near_bound_share':float(np.mean((v<=lo+.5)|(v>=hi-.5))),
                     'trajectory_min_max_deg':[float(v.min()),float(v.max())]}
        result[side]={'error_peak_time_s':float(data['times'][index]),'source_geometric_knee_deg':float(source[metric][index]),
                      'fitted_geometric_knee_deg':float(fit[metric][index]),'coordinates':q}
    return result


def inspect_tail(row,variant,body_region):
    from motiogen.evalkit.k.scoring import region
    path=Path(row['arrays']);folder=path.parent
    with np.load(path) as z:
        clock=z['times'];values={key:angles(z[key]) for key in ['R','D','F:'+variant,'F:legacy_raw']}
    metrics=[k for k in values['R'] if region(k)==body_region]
    error=np.mean([abs(values['F:'+variant][m]-values['R'][m]) for m in metrics],axis=0)
    index=int(np.nanargmax(error));time=float(clock[index]);side='l' if body_region.startswith('left') else 'r'
    if body_region.endswith('arm'):prefixes=['arm_flex','arm_add','arm_rot','elbow_flex']
    elif body_region.endswith('leg'):prefixes=['hip_flexion','hip_adduction','hip_rotation','knee_angle','ankle_angle','subtalar_angle']
    else:raise ValueError('Tail inspector expects an arm or leg')
    result={'region':body_region,'peak_time_s':time,
        'geometry_deg':{m:{key:float(v[m][index]) for key,v in values.items()} for m in metrics},'coordinates':{}}
    for setting in ['legacy_raw',variant]:
        data,definitions=read_fit(folder,setting);labels=list(data['labels']);at=int(np.argmin(abs(data['times']-time)));coords={}
        for prefix in prefixes:
            name=prefix+'_'+side;v=float(data['coordinates'][at,labels.index(name)]);d=definitions[name];lo,hi=d['range_deg']
            coords[name]={**d,'at_error_peak_deg':v,'near_bound':bool(v<=lo+.5 or v>=hi-.5)}
        result['coordinates'][setting]=coords
    with np.load(folder/variant/'region_quality_alias_fixed.npz') as z:
        valid=z[body_region+'_angle_valid'];at=int(np.argmin(abs(z['times']-time)))
        result['input_direction_support']={'share':float(np.mean(valid)),'at_error_peak':bool(valid[at])}
    result['interpretation']='coordinate bounds and support are diagnostic facts; individual causal contributions require separate solver/model ablations'
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    methods=[p.name for p in (a.out/'actual').iterdir() if (p/'result.json').exists()]
    result={'scope':'actual source-fidelity peak diagnostics; model bounds and locks are observed facts, their error contribution needs a separate ablation',
            'interpretation':'the frozen optimizer retains source-model coordinate locks; subtalar therefore does not add an independent fitted degree of freedom',
            'methods':{m:{v:inspect(a.out/'actual'/m,v) for v in ['legacy_raw','optimized_raw']} for m in methods}}
    (ROOT/'config/evalkit_k3_20261116/constraint_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print('methods',len(methods))

if __name__=='__main__':main()
