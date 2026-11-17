"""Isolated fitting worker: its job contains input/model policy, never R or scores."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--job',type=Path,required=True)
ap.add_argument('--code-root',type=Path,required=True);args=ap.parse_args()
sys.path[:0]=[str(args.code_root/'utils'),str(args.code_root/'utils/motiogen')]
import opensim as osim
osim.Logger.setLevelString('error')
import biomech
from motiogen.evalkit.k.motion import load
from motiogen.evalkit.k.fitting import fit

job=json.loads(args.job.read_text())
assert set(job)=={'input','method','output','mass_kg','floor_m','floor_policy','phase'}
out=Path(job['output']);out.mkdir(parents=True,exist_ok=True)
try:
    motion=load(job['input'],job['method'])
    result=fit(job['input'],out/'work',job['mass_kg'],job['floor_m'])
    times,labels,coords=biomech.read_storage(out/'work/coordinates.mot')
    model=osim.Model(str(out/'work/model.osim'));state=model.initSystem()
    names=[item[2] for item in biomech.MARKERS.values()]
    raw=[];consistent=[]
    for row in coords:
        biomech._set(model,state,labels,row)
        raw.append([biomech._child_origin(model,state,joint) for joint,_,_ in biomech.MARKERS.values()])
        model.assemble(state);model.realizePosition(state)
        consistent.append([biomech._child_origin(model,state,joint) for joint,_,_ in biomech.MARKERS.values()])
    raw=np.asarray(raw);consistent=np.asarray(consistent)
    np.savez_compressed(out/'fk.npz',times=times+motion.times[0],names=names,
                        positions_raw=raw,positions_assembled=consistent,
                        coordinates=coords,labels=labels)
    status={'status':'completed','phase':job['phase'],'frames':len(times),
        'clock_origin_s':float(motion.times[0]),'head_neck_model_support':'no_independent_head_joint',
        'assembly_position_change_max_m':float(np.max(np.linalg.norm(raw-consistent,axis=-1))),
        'calibration':result['receipt']['calibration'],'floor_policy':job['floor_policy'],
        'scope':'model FK; raw and constraint-assembled results kept separately; no R accessed'}
except Exception as error:
    import traceback
    status={'status':'failed','error':str(error),'traceback':traceback.format_exc()}
(out/'status.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status),flush=True)
if status['status']!='completed':sys.exit(1)
