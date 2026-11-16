#!/usr/bin/env python3
"""Record model constraints at actual-output knee-error peaks; no policy changes."""
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k.bones import common
from motiogen.evalkit.k.scoring import angles


def inspect(folder,variant):
    path=folder/variant
    with np.load(path/'kinematics.npz') as z:data={k:z[k] for k in z.files}
    labels=list(data['labels']);coords=data['coordinates'];model=ET.parse(path/'model.osim')
    definitions={e.attrib['name']:{'range_deg':(np.fromstring(e.findtext('range',''),sep=' ')*180/np.pi).tolist(),
                  'locked':e.findtext('locked','false')=='true'} for e in model.findall('.//Coordinate')}
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


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    methods=[p.name for p in (a.out/'actual').iterdir() if (p/'result.json').exists()]
    result={'scope':'actual source-fidelity peak diagnostics; model bounds and locks are observed facts, their error contribution needs a separate ablation',
            'interpretation':'the frozen optimizer retains source-model coordinate locks; subtalar therefore does not add an independent fitted degree of freedom',
            'methods':{m:{v:inspect(a.out/'actual'/m,v) for v in ['legacy_raw','optimized_raw']} for m in methods}}
    (ROOT/'config/evalkit_k3_20261116/constraint_diagnostics.json').write_text(json.dumps(result,indent=2)+'\n');print('methods',len(methods))

if __name__=='__main__':main()
