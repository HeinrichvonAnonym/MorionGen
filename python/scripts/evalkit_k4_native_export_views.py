"""D-only native FK and actual exported FK views for a frozen K implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('code-root','fit','input','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('window','method'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args()
    sys.path[:0]=[str(a.code_root/'utils'),str(a.code_root/'utils/motiogen')]
    import numpy as np
    import opensim as osim
    import biomech
    from motiogen.evalkit.k.motion import load
    from motiogen.evalkit.k.k4_fitting import model_fk
    from motiogen.evalkit.k.timebase import native_clock
    from motiogen.evalkit.k.root import compensate
    from motiogen.evalkit.k.head_neck import attach,combine
    from motiogen.evalkit.k.candidate_quality import exact_indices
    osim.Logger.setLevelString('error')
    motion=load(a.input,a.method,config=a.code_root/'config/evalkit_k4_20261117/input.json')
    receipt=json.loads((a.fit/'receipt.json').read_text())
    if receipt['key']['inputs']!=motion.provenance['inputs']:raise ValueError('Input identity changed')
    t,labels,q=biomech.read_storage(a.fit/'coordinates_refined_native.mot')
    t,clock=native_clock(t,motion.times);physical=t+motion.times[0]
    model=a.fit/'model.osim'
    _,names,q=model_fk(model,labels,q,direct_couplers=True,return_coordinates=True)
    after,root=compensate(model,physical,labels,q,motion)
    definition=json.loads((a.fit/'head_neck_model.json').read_text())
    _,head=attach(motion,model,labels,after,physical,fixed_definition=definition)
    positions,names=combine(root['after'],names,head)
    native=dict(times=physical,names=np.array(names),positions=positions,
                head_relative_thorax=head['head_relative_thorax'],head_rotation_valid=head['relative_rotation_valid'],
                head_world_rotation=head['world_rotations'][:,-1])
    with np.load(a.fit/'fit.npz') as z:export={k:z[k] for k in ('times','names','positions')}
    with np.load(a.fit/'head_neck.npz') as z:
        export.update(head_relative_thorax=z['head_relative_thorax'],head_rotation_valid=z['relative_rotation_valid'],
                      head_world_rotation=z['world_rotations'][:,-1])
    index=exact_indices(physical,export['times'])
    if list(native['names'])!=list(export['names']):raise ValueError('Native/export point definitions differ')
    finite=np.isfinite(native['positions']).all(-1)
    if not np.array_equal(finite,np.isfinite(export['positions'][index]).all(-1)):
        raise ValueError('Native/export observability changed')
    delta=export['positions'][index]-native['positions']
    discrepancy=float(abs(delta[finite]).max()) if finite.any() else None
    if discrepancy is None or discrepancy>1e-7:raise ValueError('Native-knot exported FK differs from independently rebuilt native FK')
    for name,data in [('native',native),('export',export)]:
        folder=a.output/name/a.window/a.method;folder.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(folder/'fk.npz',**data)
    result={'scope':'D-only native FK versus actual native-preserving export; R scoring is a separate process',
            'native_export_max_abs_position_m':discrepancy,'native_frames':len(physical),'export_frames':len(export['times']),
            'input_hashes':motion.provenance['inputs'],'native_clock':clock,
            'source_sha256':{n:hashlib.sha256((a.fit/n).read_bytes()).hexdigest() for n in
                ('receipt.json','model.osim','coordinates_refined_native.mot','fit.npz','head_neck.npz','head_neck_model.json')}}
    (a.output/'receipt.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
