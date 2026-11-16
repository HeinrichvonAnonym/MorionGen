#!/usr/bin/env python3
"""Inspect K output or fit full-body motion using the existing OpenSim environment."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'utils'),str(ROOT/'utils/motiogen')]
from motiogen.evalkit.k import load,evaluate,CONFIG
from motiogen.evalkit.k.geometry import recover_short_gaps,valid_rotation
from motiogen.evalkit.k.posture import summary
from motiogen.evalkit.k.config import implementation_key, resolve_config


def process(source,method,out,calibration=None,fit=False,mass=75.,floor=0.,predict_short_gaps=False,config=None):
    settings=resolve_config(config)
    motion=load(source,method,config=settings)
    bases=json.loads(Path(calibration).read_text())['anatomical_bases'] if calibration else None
    result=evaluate(motion,bases,config=settings);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    arrays={'times':motion.times,'positions':motion.positions,'joint_names':motion.names}
    arrays['position_valid'] = np.isfinite(motion.positions).all(-1)
    prior = set(motion.provenance.get('degradation',{}).get('prior_joints',[]))
    arrays['position_valid'][:,[i for i,n in enumerate(motion.names) if n in prior]] = False
    if motion.rotations is not None:
        arrays.update(segment_rotations=motion.rotations,rotation_names=motion.rotation_names)
        rotation_valid=valid_rotation(motion.rotations)
        aliases={}
        if motion.method=='chois':aliases={'left_wrist':'left_wrist_body','right_wrist':'right_wrist_body'}
        if motion.method=='closd':
            from motiogen.evalkit.k.calibration import CLOSD
            aliases=dict(zip(CLOSD,json.loads((ROOT/'config/evalkit_k_calibration/closd.json').read_text())['names']))
        rotation_valid[:,[i for i,n in enumerate(motion.rotation_names) if aliases.get(n,n) in prior]]=False
        arrays['segment_valid']=rotation_valid
    for name,item in result['body'].items():
        arrays[name+'_degrees']=item['values_deg'];arrays[name+'_valid']=item['valid']
    for region,r in result['metrics'].items():
        arrays[region+'_rotations']=r['rotations'];arrays[region+'_valid']=r['valid']
        if 'anatomical_valid' in r:arrays[region+'_anatomical_valid']=r['anatomical_valid']
    rec={**summary(result),'provenance':motion.provenance,'implementation':implementation_key(),
         'calibration_file':str(calibration) if calibration else None}
    if predict_short_gaps:
        rec['recovery']={'mode':'offline_interpolation','max_gap_seconds':settings['recovery']['max_gap_seconds'],
                         'scope':'prediction_only; observed validity and anatomical reliability unchanged','regions':{}}
        for name,item in result['metrics'].items():
            recovered,predicted=recover_short_gaps(motion.times,item['rotations'],item['valid'],settings['recovery']['max_gap_seconds'])
            arrays[name+'_with_predictions']=recovered;arrays[name+'_predicted']=predicted
            rec['recovery']['regions'][name]={'predicted_frames':int(predicted.sum())}
    if fit:
        from motiogen.evalkit.k.fitting import fit as fit_body
        try:
            fitted=fit_body(source,out/'fit',mass,floor,config=settings)
        except (ValueError,RuntimeError) as error:
            rec['fit']={'status':'unavailable','reason':str(error)}
        else:
            rec['fit']={'status':'computed','receipt':fitted['receipt'],'valid_fraction':float(np.mean(~fitted['unmeasurable'])),
                        'position_fit_valid_fraction':float(np.mean(fitted['position_fit_valid'])),
                        'calibration_valid':fitted['calibration_valid'],
                        'time_mapping':{'source_time_origin_s':float(motion.times[0]),'fit_time_origin_s':float(fitted['times'][0]),'fit_clock':'relative_seconds'}}
            np.savez_compressed(out/'fit_coordinates.npz',times=fitted['times'],labels=fitted['labels'],coordinates=fitted['coords'],valid=~fitted['unmeasurable'],
                                **{'arm_'+s+'_valid':v for s,v in fitted['arm_valid'].items()},
                                **{k+'_angle_valid':v for k,v in fitted.get('region_angle_valid',{}).items()},
                                **{k+'_registered_valid':v for k,v in fitted.get('region_registered_valid',{}).items()},
                                **{k+'_position_valid':v for k,v in fitted.get('region_position_valid',{}).items()})
    np.savez_compressed(out/'k.npz',**arrays)
    (out/'k.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return rec


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,required=True);ap.add_argument('--method');ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--calibration',type=Path);ap.add_argument('--fit',action='store_true');ap.add_argument('--mass',type=float,default=75.);ap.add_argument('--floor',type=float,default=0.)
    ap.add_argument('--predict-short-gaps',action='store_true');ap.add_argument('--config',type=Path)
    a=ap.parse_args();r=process(a.source,a.method,a.out,a.calibration,a.fit,a.mass,a.floor,a.predict_short_gaps,a.config)
    print(json.dumps({'method':r['method'],'metrics':r['metrics'],'fit':r.get('fit',{}).get('valid_fraction')},ensure_ascii=False))
